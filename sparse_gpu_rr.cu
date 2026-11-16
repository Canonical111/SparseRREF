/*
	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// GPU (CUDA cores) multi-prime CRT + rational reconstruction; see sparse_gpu.h (rr_setup).
//
// Entry e has residues r[t][e] modulo k primes. The arithmetic (Garner's CRT, Wang's half extended
// Euclid with FLINT's bounds, Lehmer steps, quotients beyond 62 bits in pieces) is that of
// sparse_rr_core.h, run by one thread per entry (rr_kernel) up to 64 primes and 32 limbs; beyond that
// one warp per entry spreads the limbs over its lanes (rr_warp_kernel). Only an overflow of the
// kernel's limb capacity leaves an entry to the CPU (status 2).
//
// The host streams chunks of entries through two slots (stage residues, submit, wait, convert), so
// the CPU converts one chunk while the GPU works on the next. Results come back compacted: the 32-bit
// descriptor of sparse_qpack.h per entry and only the limbs each fraction needs.
#include "sparse_gpu.h"
#include "sparse_rr_core.h"
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace SparseRREF::gpu {
	using namespace rr_core; // the per-entry arithmetic of the thread kernel (sparse_rr_core.h)

	namespace {
		// meta (the descriptor of sparse_qpack.h): bits 0-13 numerator limbs, 14-27 denominator limbs,
		// 28 negative, 30-31 status;
		// num/den limb-major with stride: limb i of entry e at i * stride + e; size: limbs to keep
		template <int LM>
		__global__ void rr_kernel(size_t E, size_t stride, int k, const uint64_t* __restrict__ cst, int L, int W,
			const uint64_t* __restrict__ resid, uint64_t* __restrict__ num, uint64_t* __restrict__ den,
			uint32_t* __restrict__ meta, uint64_t* __restrict__ size) {
			const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
			if (e == 0)
				size[E] = 0;
			if (e >= E) return;
			const cst_view c(cst, k, L);
			big<LM> r1, s1;
			bool s1neg = false;
			int st = rr_entry<LM>(e, stride, k, c, L, resid, r1, s1, s1neg);
			if (st == 0 && (r1.n > W || s1.n > W || r1.n == 0))
				st = 2;
			if (st != 0) {
				meta[e] = (uint32_t)st << 30;
				size[e] = 0;
				return;
			}
			for (int i = 0; i < r1.n; i++) num[(size_t)i * stride + e] = r1.w[i];
			for (int i = 0; i < s1.n; i++) den[(size_t)i * stride + e] = s1.w[i];
			meta[e] = (uint32_t)r1.n | ((uint32_t)s1.n << 14) | ((s1neg ? 1u : 0u) << 28);
			size[e] = (uint64_t)(r1.n + s1.n);
		}

		__global__ void rr_gather(size_t E, size_t stride, const uint64_t* __restrict__ num, const uint64_t* __restrict__ den,
			const uint32_t* __restrict__ meta, const uint64_t* __restrict__ off, uint64_t* __restrict__ limbs) {
			const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
			if (e >= E) return;
			const uint32_t m = meta[e];
			if (m >> 30) return;
			const int nl = m & 0x3fff, dl = (m >> 14) & 0x3fff;
			uint64_t o = off[e];
			for (int i = 0; i < nl; i++) limbs[o + i] = num[(size_t)i * stride + e];
			for (int i = 0; i < dl; i++) limbs[o + nl + i] = den[(size_t)i * stride + e];
		}

		__global__ void rr_rowstart(size_t R, const uint64_t* __restrict__ rows, const uint64_t* __restrict__ off,
			uint64_t* __restrict__ row_limb) {
			const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
			if (r <= R) row_limb[r] = off[rows[r]];
		}

		// ---- the same reconstruction with one warp per entry, for moduli beyond 32 limbs ----
		// A number of up to 32 * LPL limbs: lane l holds limbs [l * LPL, (l + 1) * LPL). Within a lane the
		// limbs are processed with ordinary carries; the carries between lanes are resolved by a carry
		// lookahead across the warp. Every operation is warp-collective (all 32 lanes, same arguments).
		constexpr unsigned FULL = 0xffffffffu;
		template <int LPL>
		struct wbig { uint64_t w[LPL]; };

		__device__ __forceinline__ unsigned lane_id() { return threadIdx.x & 31; }

		template <int LPL>
		__device__ __forceinline__ uint64_t wsel(const wbig<LPL>& x, int i) {
			uint64_t r = 0;
#pragma unroll
			for (int t = 0; t < LPL; t++)
				r = (t == i) ? x.w[t] : r;
			return r;
		}
		// g: this lane's block sends a carry (or borrow) of one up; p: an incoming one passes through it.
		// Returns the one coming into this lane; top = the one leaving lane 31.
		__device__ __forceinline__ uint32_t w_lookahead(uint32_t g, uint32_t p, uint32_t& top) {
			const unsigned lane = lane_id();
			uint32_t G = g, Q = p;
#pragma unroll
			for (int o = 1; o < 32; o <<= 1) {
				const uint32_t g1 = __shfl_up_sync(FULL, G, o), q1 = __shfl_up_sync(FULL, Q, o);
				if (lane >= (unsigned)o) {
					G = G | (Q & g1);
					Q = Q & q1;
				}
			}
			top = __shfl_sync(FULL, G, 31);
			const uint32_t cin = __shfl_up_sync(FULL, G, 1);
			return lane == 0 ? 0u : cin;
		}
		// adds each lane's outgoing carry word C to the next lane's block and resolves what follows;
		// false if the number outgrows the 32 blocks
		template <int LPL>
		__device__ __forceinline__ bool w_settle(wbig<LPL>& x, uint64_t C) {
			const unsigned lane = lane_id();
			uint64_t c = __shfl_up_sync(FULL, C, 1);
			if (lane == 0) c = 0;
			const uint64_t topC = __shfl_sync(FULL, C, 31);
			bool ones = true;
#pragma unroll
			for (int i = 0; i < LPL; i++) {
				const uint64_t s = x.w[i] + c;
				c = s < c;
				x.w[i] = s;
				ones = ones && s == ~0ULL;
			}
			uint32_t top;
			uint32_t cin = w_lookahead((uint32_t)c, ones ? 1u : 0u, top);
#pragma unroll
			for (int i = 0; i < LPL; i++) {
				const uint64_t s = x.w[i] + cin;
				cin = (cin && s == 0) ? 1u : 0u;
				x.w[i] = s;
			}
			return topC == 0 && top == 0;
		}
		template <int LPL>
		__device__ __forceinline__ void w_zero(wbig<LPL>& x) {
#pragma unroll
			for (int i = 0; i < LPL; i++) x.w[i] = 0;
		}
		template <int LPL>
		__device__ __forceinline__ void w_load(wbig<LPL>& x, const uint64_t* __restrict__ src, int L) {
			const int base = (int)lane_id() * LPL;
#pragma unroll
			for (int i = 0; i < LPL; i++) x.w[i] = base + i < L ? src[base + i] : 0;
		}
		// x = x * a + b
		template <int LPL>
		__device__ __forceinline__ bool w_muladd(wbig<LPL>& x, uint64_t a, uint64_t b) {
			uint64_t c = lane_id() == 0 ? b : 0;
#pragma unroll
			for (int i = 0; i < LPL; i++) {
				const unsigned __int128 p = (unsigned __int128)x.w[i] * a + c;
				x.w[i] = (uint64_t)p;
				c = (uint64_t)(p >> 64);
			}
			return w_settle(x, c);
		}
		// out = x * a
		template <int LPL>
		__device__ __forceinline__ bool w_mul(wbig<LPL>& out, const wbig<LPL>& x, uint64_t a) {
			uint64_t c = 0;
#pragma unroll
			for (int i = 0; i < LPL; i++) {
				const unsigned __int128 p = (unsigned __int128)x.w[i] * a + c;
				out.w[i] = (uint64_t)p;
				c = (uint64_t)(p >> 64);
			}
			return w_settle(out, c);
		}
		// x = x + y * a
		template <int LPL>
		__device__ __forceinline__ bool w_addmul(wbig<LPL>& x, const wbig<LPL>& y, uint64_t a) {
			uint64_t c = 0;
#pragma unroll
			for (int i = 0; i < LPL; i++) {
				const unsigned __int128 p = (unsigned __int128)y.w[i] * a + x.w[i] + c;
				x.w[i] = (uint64_t)p;
				c = (uint64_t)(p >> 64);
			}
			return w_settle(x, c);
		}
		// x = x - y; false if y > x
		template <int LPL>
		__device__ __forceinline__ bool w_sub(wbig<LPL>& x, const wbig<LPL>& y) {
			uint64_t b = 0;
			bool zeros = true;
#pragma unroll
			for (int i = 0; i < LPL; i++) {
				const uint64_t xi = x.w[i], yi = y.w[i];
				const uint64_t d = xi - yi;
				const uint64_t b1 = xi < yi;
				const uint64_t d2 = d - b;
				const uint64_t b2 = d < b;
				x.w[i] = d2;
				b = b1 | b2;
				zeros = zeros && d2 == 0;
			}
			uint32_t top;
			uint32_t bin = w_lookahead((uint32_t)b, zeros ? 1u : 0u, top);
#pragma unroll
			for (int i = 0; i < LPL; i++) {
				const uint64_t xi = x.w[i];
				const uint32_t nb = (bin && xi == 0) ? 1u : 0u;
				x.w[i] = xi - bin;
				bin = nb;
			}
			return top == 0;
		}
		template <int LPL>
		__device__ __forceinline__ int w_cmp(const wbig<LPL>& x, const wbig<LPL>& y) {
			int r = 0;
#pragma unroll
			for (int i = LPL - 1; i >= 0; i--)
				if (r == 0 && x.w[i] != y.w[i]) r = x.w[i] < y.w[i] ? -1 : 1;
			const unsigned m = __ballot_sync(FULL, r != 0);
			if (!m) return 0;
			return __shfl_sync(FULL, r, 31 - __clz((int)m));
		}
		template <int LPL>
		__device__ __forceinline__ int w_bitlen(const wbig<LPL>& x) {
			int hi = -1;
#pragma unroll
			for (int i = 0; i < LPL; i++)
				if (x.w[i]) hi = i;
			const unsigned m = __ballot_sync(FULL, hi >= 0);
			if (!m) return 0;
			const int src = 31 - __clz((int)m);
			int bl = 0;
			if ((int)lane_id() == src)
				bl = ((int)lane_id() * LPL + hi) * 64 + (64 - __clzll((long long)wsel(x, hi)));
			return __shfl_sync(FULL, bl, src);
		}
		// limb li of x (0 beyond the top)
		template <int LPL>
		__device__ __forceinline__ uint64_t w_limb(const wbig<LPL>& x, int li) {
			const int owner = li / LPL;
			const uint64_t mine = wsel(x, li - (int)lane_id() * LPL);
			if (owner >= 32) return 0;
			return __shfl_sync(FULL, mine, owner);
		}
		template <int LPL>
		__device__ __forceinline__ uint64_t w_word(const wbig<LPL>& x, int h) {
			const int li = h >> 6, bo = h & 63;
			const uint64_t l0 = w_limb(x, li), l1 = w_limb(x, li + 1);
			return bo ? ((l0 >> bo) | (l1 << (64 - bo))) : l0;
		}
		template <int LPL>
		__device__ __forceinline__ unsigned __int128 w_bits128(const wbig<LPL>& x, int sh) {
			const int li = sh >> 6, bo = sh & 63;
			const uint64_t l0 = w_limb(x, li), l1 = w_limb(x, li + 1), l2 = w_limb(x, li + 2);
			const uint64_t lo = bo ? ((l0 >> bo) | (l1 << (64 - bo))) : l0;
			const uint64_t hi = bo ? ((l1 >> bo) | (l2 << (64 - bo))) : l1;
			return ((unsigned __int128)hi << 64) | lo;
		}
		// x mod p; bl[s] = 2^(64 LPL 2^s) mod p, blp[s] their Shoup factors
		template <int LPL>
		__device__ __forceinline__ uint64_t w_mod(const wbig<LPL>& x, uint64_t p, uint64_t one_pr, uint64_t b64, uint64_t b64_pr,
			const uint64_t* __restrict__ bl, const uint64_t* __restrict__ blp) {
			const unsigned lane = lane_id();
			uint64_t r = 0;
#pragma unroll
			for (int i = LPL - 1; i >= 0; i--) {
				const uint64_t t = mulmod_shoup(b64, r, b64_pr, p) + reduce_word(x.w[i], p, one_pr);
				r = t >= p ? t - p : t;
			}
#pragma unroll
			for (int s = 0; s < 5; s++) {
				const uint64_t up = __shfl_down_sync(FULL, r, 1 << s);
				if ((lane & ((2u << s) - 1)) == 0) {
					const uint64_t t = r + mulmod_shoup(bl[s], up, blp[s], p);
					r = t >= p ? t - p : t;
				}
			}
			return __shfl_sync(FULL, r, 0);
		}
		// limb lane * LPL + j - off of x (0 outside the number); off may be any shift
		template <int LPL>
		__device__ __forceinline__ uint64_t w_limb_rel(const wbig<LPL>& x, int j, int off) {
			const int t = j - off;
			const int d = t >= 0 ? t / LPL : -((-t + LPL - 1) / LPL); // floor(t / LPL): the source lane's offset
			const int jj = t - d * LPL;                                // the source slot, the same in every lane
			const uint64_t mine = wsel(x, jj);
			const int src = (int)lane_id() + d;
			const uint64_t v = __shfl_sync(FULL, mine, src & 31);
			return (src >= 0 && src < 32) ? v : 0;
		}
		// out = x * 2^s (the caller knows it fits)
		template <int LPL>
		__device__ __forceinline__ void w_shl(wbig<LPL>& out, const wbig<LPL>& x, int s) {
			const int ls = s >> 6, bs = s & 63;
#pragma unroll
			for (int j = 0; j < LPL; j++) {
				const uint64_t hi = w_limb_rel(x, j, ls);
				const uint64_t lo = bs ? w_limb_rel(x, j, ls + 1) : 0;
				out.w[j] = bs ? ((hi << bs) | (lo >> (64 - bs))) : hi;
			}
		}
		// floor(a / b) estimate as quot_estimate above (ba = bitlen(a))
		template <int LPL>
		__device__ __forceinline__ uint64_t w_quot_estimate(const wbig<LPL>& a, const wbig<LPL>& b, int ba) {
			if (ba <= 128) {
				const unsigned __int128 t0 = w_bits128(a, 0), t1 = w_bits128(b, 0);
				return (uint64_t)(t0 / t1);
			}
			const int sh = ba - 128;
			unsigned __int128 t0 = w_bits128(a, sh), t1 = w_bits128(b, sh) + 1;
			if (t1 == 0)
				return 0;
			if (t0 < (t1 << 3) || (t1 >> 125)) {
				uint64_t q = 0;
				while (t0 >= t1) { t0 -= t1; q++; }
				return q;
			}
			return (uint64_t)(t0 / t1);
		}

		// out = a x - |b| y if a > 0 >= b, b y - |a| x if b >= 0 >= a (the result is >= 0)
		template <int LPL>
		__device__ __forceinline__ bool w_combo_sub(wbig<LPL>& out, int64_t a, const wbig<LPL>& x, int64_t b, const wbig<LPL>& y, wbig<LPL>& t) {
			if (a > 0)
				return w_mul(out, x, (uint64_t)a) && w_mul(t, y, (uint64_t)(-b)) && w_sub(out, t);
			return w_mul(out, y, (uint64_t)b) && w_mul(t, x, (uint64_t)(-a)) && w_sub(out, t);
		}

		// the warp constants after cst: T[k * k] (T[i * k + j] = p_0 ... p_{j-1} mod p_i), TP[k * k] their
		// Shoup factors, BL[k * 5] (2^(64 LPL 2^s) mod p_i), BLP[k * 5]
		inline size_t wcst_words(int k) { return (size_t)k * (2 * (size_t)k + 10); }

		template <int LPL>
		__global__ void __launch_bounds__(128) rr_warp_kernel(size_t E, size_t stride, int k, const uint64_t* __restrict__ cst,
			const uint64_t* __restrict__ wcst, int L, int W, const uint64_t* __restrict__ resid,
			uint64_t* __restrict__ numden, uint32_t* __restrict__ meta, uint64_t* __restrict__ size) {
			extern __shared__ uint64_t sh_digits[];
			const unsigned lane = lane_id(), wib = threadIdx.x >> 5;
			uint64_t* v = sh_digits + (size_t)wib * k;
			if (blockIdx.x == 0 && threadIdx.x == 0)
				size[E] = 0;
			const cst_view c(cst, k, L);
			const uint64_t* T = wcst;
			const uint64_t* TP = T + (size_t)k * k;
			const uint64_t* BL = TP + (size_t)k * k;
			const uint64_t* BLP = BL + (size_t)k * 5;
			const size_t warps = (size_t)gridDim.x * (blockDim.x >> 5);
			for (size_t e = (size_t)blockIdx.x * (blockDim.x >> 5) + wib; e < E; e += warps) {
				// Garner: v_i = (r_i - sum_{j < i} v_j p_0 ... p_{j-1}) / (p_0 ... p_{i-1}) mod p_i, the sum
				// split over the lanes
				for (int i = 0; i < k; i++) {
					const uint64_t pi = c.P[i];
					uint64_t part = 0;
					for (int j = (int)lane; j < i; j += 32) {
						const uint64_t t = part + mulmod_shoup(T[(size_t)i * k + j], v[j], TP[(size_t)i * k + j], pi);
						part = t >= pi ? t - pi : t;
					}
#pragma unroll
					for (int o = 16; o > 0; o >>= 1) {
						const uint64_t t = part + __shfl_xor_sync(FULL, part, o);
						part = t >= pi ? t - pi : t;
					}
					uint64_t r = resid[(size_t)i * stride + e];
					if (r >= pi) r = reduce_word(r, pi, c.ONEP[i]);
					const uint64_t d = r >= part ? r - part : r + pi - part;
					const uint64_t vi = mulmod_shoup(c.INV[i], d, c.INVP[i], pi);
					__syncwarp();
					if (lane == 0) v[i] = vi;
					__syncwarp();
				}
				wbig<LPL> r0, r1, s0, s1, N, nr0, nr1, t;
				int st = 0;
				w_zero(r1);
				for (int i = k - 1; i >= 0 && st == 0; i--) // a = a * p_i + v_i
					if (!w_muladd(r1, c.P[i], v[i])) st = 2;
				w_load(r0, c.M, L);
				w_load(N, c.N, L);
				w_zero(s0);
				w_zero(s1);
				if (lane == 0) s1.w[0] = 1;
				bool s1neg = false;
				const int bN = w_bitlen(N);
				while (st == 0 && w_cmp(r1, N) > 0) {
					if (w_bitlen(r1) > bN + 32) {
						// several steps from the leading 62 bits (as lehmer() above)
						const int b0 = w_bitlen(r0);
						if (b0 > 64) {
							const int h = b0 - 62;
							int64_t uh = (int64_t)w_word(r0, h), vh = (int64_t)w_word(r1, h);
							int64_t A = 1, B = 0, C = 0, D = 1;
							int kk = 0;
							while (true) {
								const int64_t y1 = vh + C, y2 = vh + D, x1 = uh + A, x2 = uh + B;
								if (y1 <= 0 || y2 <= 0 || x1 < 0 || x2 < 0) break;
								uint64_t q;
								if ((uint64_t)x1 < 4 * (uint64_t)y1) {
									q = 0;
									uint64_t tt = (uint64_t)x1;
									while (tt >= (uint64_t)y1) { tt -= (uint64_t)y1; q++; }
								}
								else
									q = (uint64_t)x1 / (uint64_t)y1;
								if ((unsigned __int128)q * (uint64_t)y2 > (uint64_t)x2 || (unsigned __int128)(q + 1) * (uint64_t)y2 <= (uint64_t)x2)
									break;
								const int64_t iq = (int64_t)q;
								int64_t T2 = A - iq * C; A = C; C = T2;
								T2 = B - iq * D; B = D; D = T2;
								T2 = uh - iq * vh; uh = vh; vh = T2;
								kk++;
							}
							if (kk > 0 && B != 0) {
								if (!w_combo_sub(nr0, A, r0, B, r1, t) || !w_combo_sub(nr1, C, r0, D, r1, t)) { st = 2; break; }
								if (w_cmp(nr1, N) > 0) {
									const uint64_t aA = (uint64_t)(A < 0 ? -A : A), aB = (uint64_t)(B < 0 ? -B : B);
									const uint64_t aC = (uint64_t)(C < 0 ? -C : C), aD = (uint64_t)(D < 0 ? -D : D);
									r0 = nr0;
									r1 = nr1;
									if (!w_mul(t, s0, aA) || !w_addmul(t, s1, aB) || !w_mul(nr0, s0, aC) || !w_addmul(nr0, s1, aD)) { st = 2; break; }
									s0 = t;
									s1 = nr0;
									if (kk & 1) s1neg = !s1neg;
									continue;
								}
							}
						}
					}
					int b0 = w_bitlen(r0);
					const int b1 = w_bitlen(r1);
					// a quotient beyond 62 bits in pieces, as in rr_entry
					while (b0 - b1 >= 62) {
						const int sft = b0 - b1 - 61;
						w_shl(t, r1, sft);
						uint64_t qp = w_quot_estimate(r0, t, b0);
						if (!w_mul(nr0, t, qp) || !w_sub(r0, nr0)) { st = 2; break; }
						while (w_cmp(r0, t) >= 0) {
							if (!w_sub(r0, t)) { st = 2; break; }
							qp++;
						}
						w_shl(nr1, s1, sft);
						if (st != 0 || !w_addmul(s0, nr1, qp)) { st = 2; break; }
						b0 = w_bitlen(r0);
					}
					if (st != 0) break;
					uint64_t q;
					if (w_cmp(r0, r1) < 0) // the pieces took the whole quotient
						q = 0;
					else if (b0 <= 128) {
						const unsigned __int128 t0 = w_bits128(r0, 0), t1 = w_bits128(r1, 0);
						q = (uint64_t)(t0 / t1);
					}
					else {
						const int sh = b0 - 128;
						unsigned __int128 t0 = w_bits128(r0, sh), t1 = w_bits128(r1, sh) + 1;
						if (t1 == 0)
							q = 0;
						else if (t0 < (t1 << 3) || (t1 >> 125)) {
							q = 0;
							while (t0 >= t1) { t0 -= t1; q++; }
						}
						else
							q = (uint64_t)(t0 / t1);
					}
					// (r0, r1) = (r1, r0 - q r1), (s0, s1) = (s1, s0 + q s1)
					if (!w_mul(t, r1, q) || !w_sub(r0, t)) { st = 2; break; }
					while (w_cmp(r0, r1) >= 0) {
						if (!w_sub(r0, r1)) { st = 2; break; }
						q++;
					}
					if (st != 0 || !w_addmul(s0, s1, q)) { st = 2; break; }
					t = r0; r0 = r1; r1 = t;
					t = s0; s0 = s1; s1 = t;
					s1neg = !s1neg;
				}
				if (st == 0 && w_cmp(s1, N) > 0) st = 1;
				for (int i = 0; i < k && st == 0; i++)
					if (w_mod(s1, c.P[i], c.ONEP[i], c.B64[i], c.B64P[i], BL + (size_t)i * 5, BLP + (size_t)i * 5) == 0
						&& w_mod(r1, c.P[i], c.ONEP[i], c.B64[i], c.B64P[i], BL + (size_t)i * 5, BLP + (size_t)i * 5) == 0)
						st = 1;
				const int nl = (w_bitlen(r1) + 63) / 64, dl = (w_bitlen(s1) + 63) / 64;
				if (st == 0 && (nl > W || dl > W || nl == 0)) st = 2;
				if (st != 0) {
					if (lane == 0) { meta[e] = (uint32_t)st << 30; size[e] = 0; }
					continue;
				}
				const int base = (int)lane * LPL;
				uint64_t* out = numden + e * (size_t)(2 * W);
#pragma unroll
				for (int i = 0; i < LPL; i++) {
					if (base + i < nl) out[base + i] = r1.w[i];
					if (base + i < dl) out[W + base + i] = s1.w[i];
				}
				if (lane == 0) {
					meta[e] = (uint32_t)nl | ((uint32_t)dl << 14) | ((s1neg ? 1u : 0u) << 28);
					size[e] = (uint64_t)(nl + dl);
				}
			}
		}

		// the limbs of the entries with status 0, compacted (entry-major input, 2 W limbs per entry)
		__global__ void rr_gather_em(size_t E, int W, const uint64_t* __restrict__ numden, const uint32_t* __restrict__ meta,
			const uint64_t* __restrict__ off, uint64_t* __restrict__ limbs) {
			const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
			if (e >= E) return;
			const uint32_t m = meta[e];
			if (m >> 30) return;
			const int nl = m & 0x3fff, dl = (m >> 14) & 0x3fff;
			const uint64_t o = off[e];
			const uint64_t* in = numden + e * (size_t)(2 * W);
			for (int i = 0; i < nl; i++) limbs[o + i] = in[i];
			for (int i = 0; i < dl; i++) limbs[o + nl + i] = in[W + i];
		}

		struct rr_slot {
			cudaStream_t s = nullptr;
			cudaEvent_t ev[4] = {};
			uint64_t *d_res = nullptr, *d_num = nullptr, *d_den = nullptr, *d_size = nullptr, *d_off = nullptr;
			uint64_t *d_rows = nullptr, *d_rowlimb = nullptr, *d_limbs = nullptr;
			uint32_t* d_meta = nullptr;
			void* d_tmp = nullptr;
			size_t tmp_bytes = 0;
			uint64_t *h_stage = nullptr, *h_rows = nullptr, *h_rowlimb = nullptr, *h_limbs = nullptr;
			size_t h_limbs_cap = 0;
			uint32_t* h_meta = nullptr;
			size_t E = 0, R = 0;
		};
		struct rr_state_t {
			bool on = false;
			bool warp = false; // one warp per entry (rr_warp_kernel<lpl>), entry-major results
			int k = 0, L = 0, W = 0, lpl = 0;
			size_t maxE = 0, maxR = 0;
			uint64_t* d_cst = nullptr;
			uint64_t* d_wcst = nullptr;
			rr_slot sl[2];
		};
		rr_state_t g_rr;

		using clk = std::chrono::steady_clock;
		double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

	}

	bool rr_warp_path(int k, int L) {
#ifdef SPARSERREF_TEST_HOOKS
		static const bool force = [] { // testing: the warp kernel for every size
			const char* e = std::getenv("SPARSERREF_GPU_RR_WARP");
			return e && e[0] == '1';
		}();
		if (force)
			return true;
#endif
		return k > 64 || L > 32;
	}
	bool rr_supported(int k, int L) {
		return k >= 1 && L >= 1 && L + 2 <= 512;
	}
	size_t rr_max_entries(int k, int L) {
		const size_t W = (size_t)L / 2 + 2;
		// device: residues, results, compacted limbs, descriptor, sizes; page-locked: residues, descriptor,
		// limbs; two slots
		const size_t dev = (size_t)k * 8 + 4 * W * 8 + 4 + 16, host = (size_t)k * 8 + 4 + 2 * W * 8;
		size_t fr = 0, tot = 0;
		if (!available() || cudaMemGetInfo(&fr, &tot) != cudaSuccess)
			return 0;
		const size_t dev_budget = std::min<size_t>(fr / 2, (size_t)8 << 30);
		const size_t host_budget = get_options().rr_pinned_bytes;
		return std::max<size_t>(1024, std::min(dev_budget / (2 * dev), host_budget / (2 * host)));
	}

	void rr_end() {
		auto& g = g_rr;
		for (auto& S : g.sl) {
			if (S.s)
				cudaStreamSynchronize(S.s);
			for (void* p : { (void*)S.d_res, (void*)S.d_num, (void*)S.d_den, (void*)S.d_size, (void*)S.d_off,
				(void*)S.d_rows, (void*)S.d_rowlimb, (void*)S.d_limbs, (void*)S.d_meta, S.d_tmp })
				if (p) cudaFree(p);
			for (void* p : { (void*)S.h_stage, (void*)S.h_rows, (void*)S.h_rowlimb, (void*)S.h_limbs, (void*)S.h_meta })
				if (p) cudaFreeHost(p);
			for (auto& e : S.ev)
				if (e) cudaEventDestroy(e);
			if (S.s)
				cudaStreamDestroy(S.s);
			S = rr_slot();
		}
		if (g.d_cst)
			cudaFree(g.d_cst);
		if (g.d_wcst)
			cudaFree(g.d_wcst);
		g = rr_state_t();
	}

	bool rr_begin(const rr_setup& su) {
		rr_end();
		if (!available() || !rr_supported(su.k, su.L) || su.max_entries == 0)
			return false;
		auto& g = g_rr;
		g.k = su.k; g.L = su.L; g.W = su.L / 2 + 2;
		g.warp = rr_warp_path(su.k, su.L);
		g.lpl = su.L + 2 <= 128 ? 4 : su.L + 2 <= 256 ? 8 : 16;
		g.maxE = su.max_entries; g.maxR = su.max_rows;
		const int k = su.k, L = su.L;
		// constants
		std::vector<uint64_t> cst;
		if (!make_constants(k, su.primes, su.inv_prefix, L, su.m_limbs, su.N_limbs, cst))
			return false;
		bool ok = cudaMalloc(&g.d_cst, cst.size() * 8) == cudaSuccess
			&& cudaMemcpy(g.d_cst, cst.data(), cst.size() * 8, cudaMemcpyHostToDevice) == cudaSuccess;
		if (ok && g.warp) {
			// T[i][j] = p_0 ... p_{j-1} mod p_i, BL[i][s] = 2^(64 lpl 2^s) mod p_i, with Shoup factors
			std::vector<uint64_t> w(wcst_words(k), 0);
			uint64_t *T = w.data(), *TP = T + (size_t)k * k, *BL = TP + (size_t)k * k, *BLP = BL + (size_t)k * 5;
			for (int i = 0; i < k; i++) {
				const uint64_t pi = su.primes[i];
				uint64_t prod = 1 % pi;
				for (int j = 0; j < i; j++) {
					T[(size_t)i * k + j] = prod;
					TP[(size_t)i * k + j] = shoup(prod, pi);
					prod = (uint64_t)((unsigned __int128)prod * (su.primes[j] % pi) % pi);
				}
				uint64_t b = (uint64_t)(((unsigned __int128)1 << 64) % pi), B = 1 % pi;
				for (int t = 0; t < g.lpl; t++)
					B = (uint64_t)((unsigned __int128)B * b % pi);
				for (int t = 0; t < 5; t++) {
					BL[(size_t)i * 5 + t] = B;
					BLP[(size_t)i * 5 + t] = shoup(B, pi);
					B = (uint64_t)((unsigned __int128)B * B % pi);
				}
			}
			ok = cudaMalloc(&g.d_wcst, w.size() * 8) == cudaSuccess
				&& cudaMemcpy(g.d_wcst, w.data(), w.size() * 8, cudaMemcpyHostToDevice) == cudaSuccess;
		}
		const size_t E = g.maxE, R = g.maxR, W = g.W;
		for (auto& S : g.sl) {
			if (!ok) break;
			ok = cudaStreamCreateWithFlags(&S.s, cudaStreamNonBlocking) == cudaSuccess;
			for (auto& e : S.ev)
				ok = ok && cudaEventCreate(&e) == cudaSuccess;
			ok = ok && cudaMalloc(&S.d_res, (size_t)k * E * 8) == cudaSuccess
				&& cudaMalloc(&S.d_num, (g.warp ? 2 : 1) * W * E * 8) == cudaSuccess
				&& (g.warp || cudaMalloc(&S.d_den, W * E * 8) == cudaSuccess)
				&& cudaMalloc(&S.d_size, (E + 1) * 8) == cudaSuccess && cudaMalloc(&S.d_off, (E + 1) * 8) == cudaSuccess
				&& cudaMalloc(&S.d_rows, (R + 1) * 8) == cudaSuccess && cudaMalloc(&S.d_rowlimb, (R + 1) * 8) == cudaSuccess
				&& cudaMalloc(&S.d_limbs, 2 * W * E * 8 + 8) == cudaSuccess && cudaMalloc(&S.d_meta, E * 4 + 8) == cudaSuccess
				&& cub::DeviceScan::ExclusiveSum(nullptr, S.tmp_bytes, S.d_size, S.d_off, E + 1) == cudaSuccess
				&& cudaMalloc(&S.d_tmp, S.tmp_bytes + 8) == cudaSuccess
				&& cudaHostAlloc(&S.h_stage, (size_t)k * E * 8, cudaHostAllocDefault) == cudaSuccess
				&& cudaHostAlloc(&S.h_rows, (R + 1) * 8, cudaHostAllocDefault) == cudaSuccess
				&& cudaHostAlloc(&S.h_rowlimb, (R + 1) * 8, cudaHostAllocDefault) == cudaSuccess
				&& cudaHostAlloc(&S.h_meta, E * 4 + 8, cudaHostAllocDefault) == cudaSuccess;
		}
		if (!ok) {
			cudaGetLastError();
			rr_end();
			return false;
		}
		g.on = true;
		return true;
	}

	uint64_t* rr_stage(int slot) { return g_rr.on ? g_rr.sl[slot & 1].h_stage : nullptr; }
	uint64_t* rr_stage_rows(int slot) { return g_rr.on ? g_rr.sl[slot & 1].h_rows : nullptr; }

	bool rr_submit(int slot, size_t E, size_t R) {
		auto& g = g_rr;
		if (!g.on || E > g.maxE || R > g.maxR)
			return false;
		auto& S = g.sl[slot & 1];
		S.E = E; S.R = R;
		bool ok = cudaEventRecord(S.ev[0], S.s) == cudaSuccess;
		for (int t = 0; t < g.k && ok; t++)
			ok = E == 0 || cudaMemcpyAsync(S.d_res + (size_t)t * g.maxE, S.h_stage + (size_t)t * g.maxE, E * 8,
				cudaMemcpyHostToDevice, S.s) == cudaSuccess;
		ok = ok && cudaMemcpyAsync(S.d_rows, S.h_rows, (R + 1) * 8, cudaMemcpyHostToDevice, S.s) == cudaSuccess
			&& cudaEventRecord(S.ev[1], S.s) == cudaSuccess;
		if (!ok) return false;
		const int BSZ = 128;
		if (g.warp) {
			const unsigned grid = (unsigned)std::min<size_t>((E + 4) / 4, (size_t)1 << 20); // four warps per block
			const size_t sm = (size_t)4 * g.k * 8;
			auto launch = [&](auto kern) {
				if (sm > 48 * 1024)
					cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
				kern<<<grid, BSZ, sm, S.s>>>(E, g.maxE, g.k, g.d_cst, g.d_wcst, g.L, g.W, S.d_res, S.d_num, S.d_meta, S.d_size);
			};
			if (g.lpl == 4) launch(rr_warp_kernel<4>);
			else if (g.lpl == 8) launch(rr_warp_kernel<8>);
			else launch(rr_warp_kernel<16>);
		}
		else {
			const unsigned grid = (unsigned)((E + BSZ) / BSZ); // covers e = 0 even when E == 0
			auto launch = [&](auto kern) {
				kern<<<grid, BSZ, 0, S.s>>>(E, g.maxE, g.k, g.d_cst, g.L, g.W, S.d_res, S.d_num, S.d_den, S.d_meta, S.d_size);
			};
			if (g.L <= 6) launch(rr_kernel<8>);
			else if (g.L <= 14) launch(rr_kernel<16>);
			else launch(rr_kernel<34>);
		}
		size_t tb = S.tmp_bytes;
		ok = cudaGetLastError() == cudaSuccess
			&& cub::DeviceScan::ExclusiveSum(S.d_tmp, tb, S.d_size, S.d_off, E + 1, S.s) == cudaSuccess;
		if (!ok) return false;
		if (E > 0 && g.warp)
			rr_gather_em<<<(unsigned)((E + 255) / 256), 256, 0, S.s>>>(E, g.W, S.d_num, S.d_meta, S.d_off, S.d_limbs);
		else if (E > 0)
			rr_gather<<<(unsigned)((E + 255) / 256), 256, 0, S.s>>>(E, g.maxE, S.d_num, S.d_den, S.d_meta, S.d_off, S.d_limbs);
		rr_rowstart<<<(unsigned)((R + 1 + 255) / 256), 256, 0, S.s>>>(R, S.d_rows, S.d_off, S.d_rowlimb);
		ok = cudaGetLastError() == cudaSuccess && cudaEventRecord(S.ev[2], S.s) == cudaSuccess
			&& (E == 0 || cudaMemcpyAsync(S.h_meta, S.d_meta, E * 4, cudaMemcpyDeviceToHost, S.s) == cudaSuccess)
			&& cudaMemcpyAsync(S.h_rowlimb, S.d_rowlimb, (R + 1) * 8, cudaMemcpyDeviceToHost, S.s) == cudaSuccess
			&& cudaEventRecord(S.ev[3], S.s) == cudaSuccess;
		return ok;
	}

	bool rr_wait(int slot, rr_view& v) {
		auto& g = g_rr;
		if (!g.on)
			return false;
		auto& S = g.sl[slot & 1];
		if (cudaStreamSynchronize(S.s) != cudaSuccess || fail_inject("rr"))
			return false;
		const uint64_t total = S.h_rowlimb[S.R];
		if (total > S.h_limbs_cap) {
			if (S.h_limbs)
				cudaFreeHost(S.h_limbs);
			S.h_limbs = nullptr;
			S.h_limbs_cap = 0;
			size_t nc = total + total / 4 + 1024;
			if (cudaHostAlloc(&S.h_limbs, nc * 8, cudaHostAllocDefault) != cudaSuccess) {
				S.h_limbs = nullptr;
				return false;
			}
			S.h_limbs_cap = nc;
		}
		auto t0 = clk::now();
		if (total && (cudaMemcpyAsync(S.h_limbs, S.d_limbs, total * 8, cudaMemcpyDeviceToHost, S.s) != cudaSuccess
			|| cudaStreamSynchronize(S.s) != cudaSuccess))
			return false;
		float a = 0, b = 0, c = 0;
		cudaEventElapsedTime(&a, S.ev[0], S.ev[1]);
		cudaEventElapsedTime(&b, S.ev[1], S.ev[2]);
		cudaEventElapsedTime(&c, S.ev[2], S.ev[3]);
		v.E = S.E; v.R = S.R;
		v.meta = S.h_meta; v.row_limb = S.h_rowlimb; v.limbs = S.h_limbs;
		v.h2d = a * 1e-3; v.kernel = b * 1e-3; v.d2h = c * 1e-3 + since(t0);
		return true;
	}
}
