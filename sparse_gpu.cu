/*
	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// GPU (CUDA cores) Schur update over Z/p; see sparse_gpu.h for the contract.
#include "sparse_gpu.h"
#include <map>
#include <mutex>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace SparseRREF::gpu {

	namespace {
		constexpr int BS = 256; // threads per row

		struct dev_buf {
			void* p = nullptr;
			size_t cap = 0;
		};

		bool ensure(dev_buf& b, size_t bytes) {
			if (bytes <= b.cap)
				return true;
			if (b.p)
				cudaFree(b.p);
			b.p = nullptr;
			b.cap = 0;
			size_t nb = bytes + bytes / 4 + 4096;
			if (cudaMalloc(&b.p, nb) != cudaSuccess)
				return false;
			b.cap = nb;
			return true;
		}

		struct ctx_t {
			bool init = false, ok = false;
			int sms = 0;
			size_t smem_optin = 0, smem_per_sm = 0;
			cudaStream_t s = nullptr;
			dev_buf pv_off, pv_cid, pv_val, piv_cid, posU, cid_of_col, col_of_cid;
		};

		bool g_initialized = false;
		ctx_t& ctx() {
			static ctx_t c;
			if (!c.init) {
				c.init = true;
				g_initialized = true;
				int n = 0;
				if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
					return c;
				if (cudaSetDevice(0) != cudaSuccess)
					return c;
				cudaDeviceProp pr;
				if (cudaGetDeviceProperties(&pr, 0) != cudaSuccess)
					return c;
				c.sms = pr.multiProcessorCount;
				int v = 0;
				cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
				c.smem_optin = (size_t)v;
				cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
				c.smem_per_sm = (size_t)v;
				if (cudaStreamCreate(&c.s) != cudaSuccess)
					return c;
				c.ok = true;
			}
			return c;
		}

		size_t smem_bytes(uint32_t nU, uint32_t P) {
			size_t wU = (nU + 31) / 32, wP = (P + 31) / 32;
			return (size_t)nU * 8 + wU * 4 + wP * 4 + (wU + 1) * 4 + 16;
		}
		size_t smem_bytes_global(uint32_t nU, uint32_t P) {
			size_t wU = (nU + 31) / 32, wP = (P + 31) / 32;
			return wP * 4 + (wU + 1) * 4 + 16;
		}
		size_t slot_words(uint32_t nU) { // acc + presence bits, in 64-bit words, rounded to 32 words
			size_t w = (size_t)nU + ((nU + 31) / 32 + 1) / 2;
			return (w + 31) / 32 * 32;
		}
		constexpr size_t global_scratch_budget = (size_t)3 << 29; // bytes for the large-U accumulators of one slot

		__device__ __forceinline__ uint64_t mulmod_shoup(uint64_t a, uint64_t b, uint64_t a_pr, uint64_t p) {
			uint64_t q = __umul64hi(a_pr, b);
			uint64_t r = a * b - q * p;
			return r >= p ? r - p : r;
		}

		// GLOBAL = false: accumulator and presence bits in shared memory (small U);
		// GLOBAL = true: in this block's slice of a global scratch (large U), pivot bits still shared
		template <bool GLOBAL>
		__global__ void __launch_bounds__(BS) schur_kernel(
			uint64_t* __restrict__ scratch, size_t slot_words,
			uint64_t p, uint32_t nU, uint32_t P, uint32_t R,
			const uint64_t* __restrict__ pv_off, const uint32_t* __restrict__ pv_cid, const uint64_t* __restrict__ pv_val,
			const uint32_t* __restrict__ piv_cid, const int32_t* __restrict__ posU,
			const int32_t* __restrict__ cid_of_col, const uint32_t* __restrict__ col_of_cid,
			const uint64_t* __restrict__ in_off, const uint32_t* __restrict__ in_col, const uint64_t* __restrict__ in_val,
			uint32_t* __restrict__ tp_idx,
			unsigned long long* __restrict__ out_base, uint32_t* __restrict__ out_cnt,
			uint32_t* __restrict__ out_col, uint64_t* __restrict__ out_val, uint8_t* __restrict__ status,
			unsigned long long* __restrict__ top, unsigned long long cap) {

			extern __shared__ uint64_t smem[];
			const uint32_t wU = (nU + 31) / 32, wP = (P + 31) / 32;
			uint64_t* acc;
			uint32_t* pres;
			uint32_t* pbits;
			if constexpr (GLOBAL) {
				acc = scratch + (size_t)blockIdx.x * slot_words;
				pres = (uint32_t*)(acc + nU);
				pbits = (uint32_t*)smem;
			}
			else {
				acc = smem;
				pres = (uint32_t*)(acc + nU);
				pbits = pres + wU;
			}
			uint32_t* wpref = pbits + wP;
			__shared__ int s_j;
			__shared__ uint64_t s_e, s_epr;
			__shared__ unsigned long long s_base;
			__shared__ int s_over;
			__shared__ uint32_t s_wcnt[BS / 32];
			const int tid = threadIdx.x;

			if constexpr (!GLOBAL) {
				for (uint32_t i = tid; i < nU; i += BS) acc[i] = 0;
				for (uint32_t i = tid; i < wU; i += BS) pres[i] = 0;
			}
			for (uint32_t i = tid; i < wP; i += BS) pbits[i] = 0;
			__syncthreads();

			for (uint32_t row = blockIdx.x; row < R; row += gridDim.x) {
				// load the row: entries on U go to the accumulator, the others (pass-through, unchanged)
				// are listed in order in tp_idx[rb, rb + npass) by their offset in the row
				const uint64_t rb = in_off[row], re = in_off[row + 1];
				const int lane = tid & 31, warp = tid >> 5;
				uint32_t npass = 0;
				for (uint64_t b0 = rb; b0 < re; b0 += BS) {
					const uint64_t k = b0 + tid;
					bool pass = false;
					if (k < re) {
						const int32_t c = cid_of_col[in_col[k]];
						if (c >= 0) {
							acc[c] = in_val[k];
							atomicOr(&pres[c >> 5], 1u << (c & 31));
							int pos = posU[c];
							if (pos >= 0)
								atomicOr(&pbits[pos >> 5], 1u << (pos & 31));
						}
						else
							pass = true;
					}
					const unsigned m = __ballot_sync(0xffffffffu, pass);
					if (lane == 0)
						s_wcnt[warp] = __popc(m);
					__syncthreads();
					uint32_t off = npass, tot = 0;
					for (int w2 = 0; w2 < BS / 32; w2++) {
						if (w2 < warp)
							off += s_wcnt[w2];
						tot += s_wcnt[w2];
					}
					if (pass)
						tp_idx[rb + off + __popc(m & ((1u << lane) - 1u))] = (uint32_t)(k - rb);
					npass += tot;
					__syncthreads();
				}

				uint32_t w = 0;
				while (true) {
					if (tid < 32) {
						int found = -1;
						for (uint32_t base = w; base < wP; base += 32) {
							uint32_t idx = base + tid;
							uint32_t word = idx < wP ? pbits[idx] : 0u;
							unsigned m = __ballot_sync(0xffffffffu, word != 0);
							if (m) {
								int lane = __ffs(m) - 1;
								uint32_t wv = __shfl_sync(0xffffffffu, word, lane);
								found = (int)((base + lane) * 32 + (__ffs(wv) - 1));
								break;
							}
						}
						if (tid == 0) {
							s_j = found;
							if (found >= 0) {
								pbits[found >> 5] &= ~(1u << (found & 31));
								uint64_t ev = acc[piv_cid[found]];
								s_e = ev;
								if (ev)
									s_epr = (uint64_t)(((unsigned __int128)ev << 64) / p);
							}
						}
					}
					__syncthreads();
					const int j = s_j;
					if (j < 0)
						break;
					w = (uint32_t)j >> 5;
					const uint64_t ev = s_e;
					if (ev != 0) { // the row holds the pivot column: apply pivot j
						const uint64_t epr = s_epr;
						const uint64_t pb = pv_off[j], pe = pv_off[j + 1];
						for (uint64_t k = pb + tid; k < pe; k += BS) {
							uint32_t c = pv_cid[k];
							uint64_t old = acc[c];
							uint64_t t = mulmod_shoup(ev, pv_val[k], epr, p);
							uint64_t nw = old >= t ? old - t : old - t + p;
							acc[c] = nw;
							if (old == 0 && nw != 0) {
								atomicOr(&pres[c >> 5], 1u << (c & 31));
								int pos = posU[c];
								if (pos > j)
									atomicOr(&pbits[pos >> 5], 1u << (pos & 31));
							}
							else if (old != 0 && nw == 0) {
								atomicAnd(&pres[c >> 5], ~(1u << (c & 31)));
							}
						}
					}
					__syncthreads();
				}

				// exclusive prefix of the popcounts of the presence words (warp 0)
				if (tid < 32) {
					const uint32_t chunk = (wU + 31) / 32;
					const uint32_t b0 = tid * chunk, b1 = min(wU, b0 + chunk);
					uint32_t sum = 0;
					for (uint32_t i = b0; i < b1; i++) sum += __popc(pres[i]);
					uint32_t incl = sum;
					for (int o = 1; o < 32; o <<= 1) {
						uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
						if (tid >= o) incl += y;
					}
					uint32_t run = incl - sum;
					for (uint32_t i = b0; i < b1; i++) {
						wpref[i] = run;
						run += __popc(pres[i]);
					}
					if (tid == 31) wpref[wU] = incl;
				}
				__syncthreads();
				const uint32_t npres = wpref[wU];
				const uint32_t cnt = npres + npass;
				if (tid == 0) {
					unsigned long long b = atomicAdd(top, (unsigned long long)cnt);
					s_base = b;
					s_over = (b + cnt > cap) ? 1 : 0;
					status[row] = (uint8_t)s_over;
					out_base[row] = b;
					out_cnt[row] = cnt;
				}
				__syncthreads();
				if (!s_over) {
					// merge: the new entries on U and the pass-through entries, by global column
					const unsigned long long base = s_base;
					const uint32_t* tpi = tp_idx + rb;
					const uint32_t* rcol = in_col + rb;
					for (uint32_t wi = tid; wi < wU; wi += BS) {
						uint32_t m = pres[wi];
						uint32_t r = wpref[wi];
						while (m) {
							int bit = __ffs(m) - 1;
							m &= m - 1;
							uint32_t c = wi * 32 + bit;
							uint32_t gc = col_of_cid[c];
							uint32_t lo = 0, hi = npass; // pass-through entries before gc
							while (lo < hi) {
								uint32_t mid = (lo + hi) >> 1;
								if (rcol[tpi[mid]] < gc) lo = mid + 1; else hi = mid;
							}
							out_col[base + r + lo] = gc;
							out_val[base + r + lo] = acc[c];
							r++;
						}
					}
					for (uint32_t q = tid; q < npass; q += BS) {
						const uint32_t kk = tpi[q];
						const uint32_t gc = rcol[kk];
						uint32_t lo = 0, hi = nU; // U columns before gc
						while (lo < hi) {
							uint32_t mid = (lo + hi) >> 1;
							if (col_of_cid[mid] < gc) lo = mid + 1; else hi = mid;
						}
						uint32_t below = wpref[lo >> 5];
						if (lo & 31)
							below += __popc(pres[lo >> 5] & ((1u << (lo & 31)) - 1u));
						out_col[base + q + below] = gc;
						out_val[base + q + below] = in_val[rb + kk];
					}
				}
				__syncthreads();
				for (uint32_t wi = tid; wi < wU; wi += BS) {
					uint32_t m = pres[wi];
					while (m) {
						int bit = __ffs(m) - 1;
						m &= m - 1;
						acc[wi * 32 + bit] = 0;
					}
					pres[wi] = 0;
				}
				__syncthreads();
			}
		}

		template <typename X>
		bool up(dev_buf& b, const X* src, size_t n, cudaStream_t s) {
			if (!ensure(b, n * sizeof(X) + 8))
				return false;
			if (n == 0)
				return true;
			return cudaMemcpyAsync(b.p, src, n * sizeof(X), cudaMemcpyHostToDevice, s) == cudaSuccess;
		}
		using clk = std::chrono::steady_clock;
		double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
	}

	bool available() {
		static std::mutex m; // ctx() initialises once; a warm-up thread may run concurrently
		std::lock_guard<std::mutex> lock(m);
		return ctx().ok;
	}

	void warmup() {
		cudaFree(0); // creates the device's primary context; ctx() finds it ready
		cudaGetLastError();
	}

	bool initialized() {
		return g_initialized && ctx().ok;
	}

	bool fits(uint32_t nU, uint32_t P) {
		auto& c = ctx();
		if (!c.ok)
			return false;
		if (smem_bytes(nU, P) <= c.smem_optin)
			return true;
		// large U: at least one block per SM must fit the scratch budget (options::large_u)
		return get_options().large_u && smem_bytes_global(nU, P) <= c.smem_optin
			&& slot_words(nU) * 8 * (size_t)c.sms <= global_scratch_budget;
	}

	namespace {
		struct host_buf { void* p = nullptr; size_t cap = 0; };
		host_buf g_host[8]; // job inputs (host_buffer)

		void* grow_pinned(host_buf& b, size_t bytes) {
			if (bytes <= b.cap && b.p)
				return b.p;
			if (b.p)
				cudaFreeHost(b.p);
			b.p = nullptr;
			b.cap = 0;
			// allocating page-locked memory is slow: grow by a factor 2
			size_t nb = std::max(bytes + 4096, 2 * b.cap);
			if (cudaHostAlloc(&b.p, nb, cudaHostAllocDefault) != cudaSuccess) {
				b.p = nullptr;
				cudaGetLastError();
				if (cudaHostAlloc(&b.p, bytes + 4096, cudaHostAllocDefault) != cudaSuccess) {
					b.p = nullptr;
					cudaGetLastError();
					return nullptr;
				}
				nb = bytes + 4096;
			}
			b.cap = nb;
			return b.p;
		}

		// one of the two slots of the row stream: its own stream, device buffers and page-locked buffers
		struct slot_t {
			cudaStream_t s = nullptr;
			cudaEvent_t ev[4] = {};
			dev_buf in_off, in_col, in_val, tp_idx, out_base, out_cnt, out_col, out_val, status, top, scratch;
			host_buf in[3];
			host_buf h_top, h_base, h_cnt, h_status, h_col, h_val;
			uint32_t R = 0;
			uint64_t cap = 0;
			// the kernel leaves its accumulators zero, so the scratch stays zero from one part to the
			// next; it is cleared after an allocation or a failure
			bool scratch_zero = false;
		};
		struct stream_t {
			bool init = false, ok = false;
			slot_t sl[2];
			// the current pivot set
			uint64_t p = 0;
			uint32_t nU = 0, P = 0;
			bool global = false;
			size_t sm = 0, sw = 0;
			uint32_t blocks = 0;
		};
		stream_t& stm() {
			static stream_t g;
			if (!g.init) {
				g.init = true;
				if (!ctx().ok)
					return g;
				bool ok = true;
				for (auto& S : g.sl) {
					ok = ok && cudaStreamCreateWithFlags(&S.s, cudaStreamNonBlocking) == cudaSuccess;
					for (auto& e : S.ev)
						ok = ok && cudaEventCreate(&e) == cudaSuccess;
				}
				g.ok = ok;
			}
			return g;
		}
	}

	namespace {
		options g_options;
		std::mutex g_options_mutex;
	}
	void set_options(const options& o) {
		std::lock_guard<std::mutex> lock(g_options_mutex);
		g_options = o;
	}
	const options& get_options() { return g_options; }

	bool fail_inject(const char* site) {
#ifndef SPARSERREF_TEST_HOOKS
		(void)site;
		return false;
#else
		static const std::string spec = [] {
			const char* e = std::getenv("SPARSERREF_GPU_FAIL");
			return std::string(e ? e : "");
		}();
		if (spec.empty())
			return false;
		static std::mutex m;
		static std::map<std::string, long> calls;
		std::lock_guard<std::mutex> lock(m);
		const long n = ++calls[site];
		size_t at = 0;
		while (at < spec.size()) {
			size_t comma = spec.find_first_of(",/", at);
			if (comma == std::string::npos) comma = spec.size();
			const std::string item = spec.substr(at, comma - at);
			const size_t colon = item.find(':');
			if (colon != std::string::npos && item.substr(0, colon) == site && std::atol(item.c_str() + colon + 1) == n) {
				std::fprintf(stderr, "SPARSERREF_GPU_FAIL: failing %s call %ld\n", site, n);
				return true;
			}
			at = comma + 1;
		}
		return false;
#endif
	}

	void* host_buffer(int slot, size_t bytes) {
		if (slot < 0 || slot > 7 || !ctx().ok)
			return nullptr;
		return grow_pinned(g_host[slot], bytes);
	}

	bool schur_begin(const schur_common& cm) {
		auto& c = ctx();
		auto& g = stm();
		if (!g.ok || !fits(cm.nU, cm.P))
			return false;
		const uint64_t nPv = cm.pv_off[cm.P];
		cudaStream_t s0 = g.sl[0].s;
		bool ok = up(c.pv_off, cm.pv_off, (size_t)cm.P + 1, s0) && up(c.pv_cid, cm.pv_cid, nPv, s0)
			&& up(c.pv_val, cm.pv_val, nPv, s0) && up(c.piv_cid, cm.piv_cid, cm.P, s0)
			&& up(c.posU, cm.posU, cm.nU, s0) && up(c.cid_of_col, cm.cid_of_col, cm.ncol, s0)
			&& up(c.col_of_cid, cm.col_of_cid, cm.nU, s0) && cudaStreamSynchronize(s0) == cudaSuccess;
		if (!ok)
			return false;
		g.p = cm.p; g.nU = cm.nU; g.P = cm.P;
		g.global = smem_bytes(cm.nU, cm.P) > c.smem_optin;
		g.sm = g.global ? smem_bytes_global(cm.nU, cm.P) : smem_bytes(cm.nU, cm.P);
		auto kern = g.global ? schur_kernel<true> : schur_kernel<false>;
		if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g.sm) != cudaSuccess)
			return false;
		int occ = 0;
		if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, BS, g.sm) != cudaSuccess || occ < 1)
			return false;
		uint64_t blocks = (uint64_t)occ * c.sms;
		g.sw = 0;
		if (g.global) {
			g.sw = slot_words(cm.nU);
			blocks = std::min<uint64_t>(blocks, global_scratch_budget / (g.sw * 8));
			if (blocks < 1)
				return false;
		}
		g.blocks = (uint32_t)blocks;
		return true;
	}

	void* schur_stage(int slot, int which, size_t bytes) {
		auto& g = stm();
		if (!g.ok || which < 0 || which > 2)
			return nullptr;
		return grow_pinned(g.sl[slot & 1].in[which], bytes);
	}

	bool schur_submit(int slot, uint32_t R, uint64_t cap) {
		auto& g = stm();
		if (!g.ok)
			return false;
		auto& S = g.sl[slot & 1];
		S.R = R; S.cap = cap;
		const uint64_t* in_off = (const uint64_t*)S.in[0].p;
		if (!in_off)
			return false;
		const uint64_t nIn = in_off[R];
		bool ok = cudaEventRecord(S.ev[0], S.s) == cudaSuccess
			&& up(S.in_off, in_off, (size_t)R + 1, S.s)
			&& (nIn == 0 || (up(S.in_col, (const uint32_t*)S.in[1].p, nIn, S.s) && up(S.in_val, (const uint64_t*)S.in[2].p, nIn, S.s)))
			&& ensure(S.tp_idx, nIn * 4 + 8)
			&& ensure(S.out_base, (size_t)R * 8 + 8) && ensure(S.out_cnt, (size_t)R * 4 + 8)
			&& ensure(S.out_col, cap * 4 + 8) && ensure(S.out_val, cap * 8 + 8)
			&& ensure(S.status, (size_t)R + 8) && ensure(S.top, 8)
			&& cudaMemsetAsync(S.top.p, 0, 8, S.s) == cudaSuccess;
		if (ok && g.global) {
			void* before = S.scratch.p;
			ok = ensure(S.scratch, (size_t)g.blocks * g.sw * 8);
			if (ok && (S.scratch.p != before || !S.scratch_zero)) {
				ok = cudaMemsetAsync(S.scratch.p, 0, S.scratch.cap, S.s) == cudaSuccess;
				S.scratch_zero = ok;
			}
		}
		ok = ok && cudaEventRecord(S.ev[1], S.s) == cudaSuccess;
		if (!ok) {
			S.scratch_zero = false;
			return false;
		}
		auto& c = ctx();
		const uint32_t grid = std::min<uint32_t>(R, g.blocks);
		if (R > 0) {
			auto kern = g.global ? schur_kernel<true> : schur_kernel<false>;
			kern<<<grid, BS, g.sm, S.s>>>((uint64_t*)S.scratch.p, g.sw, g.p, g.nU, g.P, R,
				(const uint64_t*)c.pv_off.p, (const uint32_t*)c.pv_cid.p, (const uint64_t*)c.pv_val.p,
				(const uint32_t*)c.piv_cid.p, (const int32_t*)c.posU.p,
				(const int32_t*)c.cid_of_col.p, (const uint32_t*)c.col_of_cid.p,
				(const uint64_t*)S.in_off.p, (const uint32_t*)S.in_col.p, (const uint64_t*)S.in_val.p,
				(uint32_t*)S.tp_idx.p,
				(unsigned long long*)S.out_base.p, (uint32_t*)S.out_cnt.p,
				(uint32_t*)S.out_col.p, (uint64_t*)S.out_val.p, (uint8_t*)S.status.p,
				(unsigned long long*)S.top.p, (unsigned long long)cap);
			if (cudaGetLastError() != cudaSuccess) {
				S.scratch_zero = false;
				return false;
			}
		}
		auto* h_top = grow_pinned(S.h_top, 8);
		auto* h_base = grow_pinned(S.h_base, (size_t)R * 8 + 8);
		auto* h_cnt = grow_pinned(S.h_cnt, (size_t)R * 4 + 8);
		auto* h_status = grow_pinned(S.h_status, (size_t)R + 8);
		ok = h_top && h_base && h_cnt && h_status && cudaEventRecord(S.ev[2], S.s) == cudaSuccess
			&& cudaMemcpyAsync(h_top, S.top.p, 8, cudaMemcpyDeviceToHost, S.s) == cudaSuccess
			&& (R == 0 || (cudaMemcpyAsync(h_base, S.out_base.p, (size_t)R * 8, cudaMemcpyDeviceToHost, S.s) == cudaSuccess
				&& cudaMemcpyAsync(h_cnt, S.out_cnt.p, (size_t)R * 4, cudaMemcpyDeviceToHost, S.s) == cudaSuccess
				&& cudaMemcpyAsync(h_status, S.status.p, R, cudaMemcpyDeviceToHost, S.s) == cudaSuccess))
			&& cudaEventRecord(S.ev[3], S.s) == cudaSuccess;
		return ok;
	}

	bool schur_wait(int slot, schur_out& res) {
		auto& g = stm();
		if (!g.ok)
			return false;
		auto& S = g.sl[slot & 1];
		if (cudaStreamSynchronize(S.s) != cudaSuccess) {
			S.scratch_zero = false;
			return false;
		}
		const uint64_t valid = std::min<uint64_t>(*(const unsigned long long*)S.h_top.p, S.cap);
		auto* h_col = (uint32_t*)grow_pinned(S.h_col, valid * 4 + 8);
		auto* h_val = (uint64_t*)grow_pinned(S.h_val, valid * 8 + 8);
		if (!h_col || !h_val)
			return false;
		auto t0 = clk::now();
		if (valid && (cudaMemcpyAsync(h_col, S.out_col.p, valid * 4, cudaMemcpyDeviceToHost, S.s) != cudaSuccess
			|| cudaMemcpyAsync(h_val, S.out_val.p, valid * 8, cudaMemcpyDeviceToHost, S.s) != cudaSuccess
			|| cudaStreamSynchronize(S.s) != cudaSuccess))
			return false;
		float a = 0, b = 0, cc = 0;
		cudaEventElapsedTime(&a, S.ev[0], S.ev[1]);
		cudaEventElapsedTime(&b, S.ev[1], S.ev[2]);
		cudaEventElapsedTime(&cc, S.ev[2], S.ev[3]);
		res.h2d = a * 1e-3; res.kernel = b * 1e-3; res.d2h = cc * 1e-3 + since(t0);
		res.base = (const unsigned long long*)S.h_base.p; res.cnt = (const uint32_t*)S.h_cnt.p;
		res.status = (const uint8_t*)S.h_status.p; res.col = h_col; res.val = h_val;
		return true;
	}
}
