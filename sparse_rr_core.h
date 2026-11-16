/*
	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// Multi-prime CRT + rational reconstruction of one entry in fixed-size limb arrays, shared by the CPU
// (sparse_mat.h) and the GPU kernels (sparse_gpu_rr.cu).
//
// The entry has residues r_i modulo k primes p_i < 2^63. a = CRT(r) in [0, m) is formed by Garner's
// mixed radix and Horner in L limbs, then Wang's half extended Euclid runs on (m, a):
//   (r0, s0) = (m, 0), (r1, s1) = (a, 1); while r1 > N: q = r0 / r1, shift the pairs.
// Success iff |s1| <= N and gcd(r1, s1) = 1, giving n/d = sign(s1) r1 / |s1| with N = D =
// floor(sqrt((m-1)/2)), the bounds of FLINT's fmpq_reconstruct_fmpz. A fraction within these bounds
// congruent to a is unique, and the remainder sequence finds it iff it exists, so the result equals
// FLINT's. gcd(r1, s1) divides m (a product of distinct primes), so the gcd test is "no prime divides
// both". Lehmer steps take several quotients from the leading bits at once; a quotient beyond 62 bits
// is taken in 61-bit pieces (same remainder and cofactor). Status: 0 a fraction, 1 none within the
// bounds, 2 the limb capacity LM was exceeded (the caller uses FLINT).
//
// Needs unsigned __int128 (GCC, Clang, nvcc); elsewhere SPARSERREF_HAVE_RR_CORE is not defined and the
// CPU keeps FLINT's functions.

#ifndef SPARSE_RR_CORE_H
#define SPARSE_RR_CORE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define SPR_RR_HD __host__ __device__ __forceinline__
#else
#define SPR_RR_HD inline
#endif

#if defined(__SIZEOF_INT128__) || defined(__CUDACC__)
#define SPARSERREF_HAVE_RR_CORE 1

namespace SparseRREF::rr_core {

	// the largest k and L of rr_entry (Garner's digits live in a local array of 64 words)
	inline constexpr int max_primes = 64;

	SPR_RR_HD int clz64(uint64_t x) { // x != 0
#if defined(__CUDA_ARCH__)
		return __clzll((long long)x);
#else
		return __builtin_clzll(x);
#endif
	}
	SPR_RR_HD uint64_t mulhi64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
		return __umul64hi(a, b);
#else
		return (uint64_t)(((unsigned __int128)a * b) >> 64);
#endif
	}
	// floor(a 2^64 / p), a < p
	inline uint64_t shoup(uint64_t a, uint64_t p) { return (uint64_t)(((unsigned __int128)a << 64) / p); }

	template <int LM>
	struct big {
		uint64_t w[LM];
		int n; // used limbs (no leading zero limb), 0 for zero
	};

	template <int LM>
	SPR_RR_HD void norm(big<LM>& x) {
		while (x.n > 0 && x.w[x.n - 1] == 0) x.n--;
	}
	template <int LM>
	SPR_RR_HD int cmp(const big<LM>& a, const big<LM>& b) {
		if (a.n != b.n) return a.n < b.n ? -1 : 1;
		for (int i = a.n - 1; i >= 0; i--)
			if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
		return 0;
	}
	template <int LM>
	SPR_RR_HD int bitlen(const big<LM>& a) {
		return a.n == 0 ? 0 : (a.n - 1) * 64 + (64 - clz64(a.w[a.n - 1]));
	}
	// the 128 bits of a starting at bit sh (bits beyond the top read as 0)
	template <int LM>
	SPR_RR_HD unsigned __int128 bits128(const big<LM>& a, int sh) {
		int li = sh >> 6, bo = sh & 63;
		uint64_t l0 = li < a.n ? a.w[li] : 0, l1 = li + 1 < a.n ? a.w[li + 1] : 0, l2 = li + 2 < a.n ? a.w[li + 2] : 0;
		uint64_t lo = bo ? ((l0 >> bo) | (l1 << (64 - bo))) : l0;
		uint64_t hi = bo ? ((l1 >> bo) | (l2 << (64 - bo))) : l1;
		return ((unsigned __int128)hi << 64) | lo;
	}
	// x = x - q * y (requires x >= q * y), returns false if it would go negative
	template <int LM>
	SPR_RR_HD bool submul(big<LM>& x, const big<LM>& y, uint64_t q) {
		uint64_t borrow = 0, carry = 0;
		int n = x.n;
		for (int i = 0; i < n; i++) {
			unsigned __int128 p = (unsigned __int128)(i < y.n ? y.w[i] : 0) * q + carry;
			carry = (uint64_t)(p >> 64);
			uint64_t pl = (uint64_t)p;
			uint64_t xi = x.w[i];
			uint64_t d = xi - pl;
			uint64_t b1 = xi < pl;
			uint64_t d2 = d - borrow;
			uint64_t b2 = d < borrow;
			x.w[i] = d2;
			borrow = b1 + b2;
		}
		if (carry || borrow) return false;
		norm(x);
		return true;
	}
	// x = x + y
	template <int LM>
	SPR_RR_HD bool add(big<LM>& x, const big<LM>& y) {
		int n = x.n > y.n ? x.n : y.n;
		uint64_t c = 0;
		for (int i = 0; i < n; i++) {
			uint64_t a = i < x.n ? x.w[i] : 0, b = i < y.n ? y.w[i] : 0;
			uint64_t s = a + b;
			uint64_t c1 = s < a;
			uint64_t s2 = s + c;
			uint64_t c2 = s2 < s;
			x.w[i] = s2;
			c = c1 + c2;
		}
		x.n = n;
		if (c) {
			if (n >= LM) return false;
			x.w[n] = c;
			x.n = n + 1;
		}
		return true;
	}
	// x = x + y * q
	template <int LM>
	SPR_RR_HD bool addmul(big<LM>& x, const big<LM>& y, uint64_t q) {
		int n = x.n > y.n ? x.n : y.n;
		uint64_t carry = 0;
		for (int i = 0; i < n; i++) {
			unsigned __int128 p = (unsigned __int128)(i < y.n ? y.w[i] : 0) * q + (i < x.n ? x.w[i] : 0) + carry;
			x.w[i] = (uint64_t)p;
			carry = (uint64_t)(p >> 64);
		}
		x.n = n;
		if (carry) {
			if (n >= LM) return false;
			x.w[n] = carry;
			x.n = n + 1;
		}
		return true;
	}
	SPR_RR_HD uint64_t mulmod_shoup(uint64_t a, uint64_t b, uint64_t a_pr, uint64_t p) {
		// a < p < 2^63, a_pr = floor(a 2^64 / p), any 64-bit b
		uint64_t q = mulhi64(a_pr, b);
		uint64_t r = a * b - q * p;
		return r >= p ? r - p : r;
	}
	SPR_RR_HD uint64_t reduce_word(uint64_t w, uint64_t p, uint64_t one_pr) {
		return mulmod_shoup(1, w, one_pr, p);
	}
	// a mod p, with b64 = 2^64 mod p
	template <int LM>
	SPR_RR_HD uint64_t mod_big(const big<LM>& a, uint64_t p, uint64_t one_pr, uint64_t b64, uint64_t b64_pr) {
		uint64_t r = 0;
		for (int i = a.n - 1; i >= 0; i--) {
			uint64_t x = mulmod_shoup(b64, r, b64_pr, p) + reduce_word(a.w[i], p, one_pr);
			r = x >= p ? x - p : x;
		}
		return r;
	}

	// floor(a / 2^h) for 0 <= h, assuming it fits 64 bits
	template <int LM>
	SPR_RR_HD uint64_t shr_word(const big<LM>& a, int h) {
		int li = h >> 6, bo = h & 63;
		uint64_t l0 = li < a.n ? a.w[li] : 0, l1 = li + 1 < a.n ? a.w[li + 1] : 0;
		return bo ? ((l0 >> bo) | (l1 << (64 - bo))) : l0;
	}
	// out = x * a (a word)
	template <int LM>
	SPR_RR_HD bool mul_word(big<LM>& out, const big<LM>& x, uint64_t a) {
		uint64_t carry = 0;
		for (int i = 0; i < x.n; i++) {
			unsigned __int128 p = (unsigned __int128)x.w[i] * a + carry;
			out.w[i] = (uint64_t)p;
			carry = (uint64_t)(p >> 64);
		}
		out.n = x.n;
		if (carry) {
			if (x.n >= LM) return false;
			out.w[x.n] = carry;
			out.n = x.n + 1;
		}
		norm(out);
		return true;
	}
	// x = x - y (requires x >= y)
	template <int LM>
	SPR_RR_HD bool sub_big(big<LM>& x, const big<LM>& y) {
		if (cmp(x, y) < 0) return false;
		uint64_t borrow = 0;
		for (int i = 0; i < x.n; i++) {
			uint64_t yi = i < y.n ? y.w[i] : 0;
			uint64_t d = x.w[i] - yi;
			uint64_t b1 = x.w[i] < yi;
			uint64_t d2 = d - borrow;
			uint64_t b2 = d < borrow;
			x.w[i] = d2;
			borrow = b1 + b2;
		}
		norm(x);
		return true;
	}
	// out = a x - |b| y when a > 0 >= b, b y - |a| x when b >= 0 >= a (the result is >= 0)
	template <int LM>
	SPR_RR_HD bool combo_sub(big<LM>& out, int64_t a, const big<LM>& x, int64_t b, const big<LM>& y) {
		big<LM> t;
		if (a > 0) {
			if (!mul_word(out, x, (uint64_t)a) || !mul_word(t, y, (uint64_t)(-b))) return false;
		}
		else {
			if (!mul_word(out, y, (uint64_t)b) || !mul_word(t, x, (uint64_t)(-a))) return false;
		}
		return sub_big(out, t);
	}
	// out = a x + b y (words)
	template <int LM>
	SPR_RR_HD bool combo_add(big<LM>& out, uint64_t a, const big<LM>& x, uint64_t b, const big<LM>& y) {
		if (!mul_word(out, x, a)) return false;
		return addmul(out, y, b);
	}

	// out = x * 2^s (the caller knows it fits)
	template <int LM>
	SPR_RR_HD void shl(big<LM>& out, const big<LM>& x, int s) {
		const int ls = s >> 6, bs = s & 63;
		int n = x.n + ls + 1;
		if (n > LM) n = LM;
		for (int i = n - 1; i >= 0; i--) {
			const int j = i - ls;
			const uint64_t hi = (j >= 0 && j < x.n) ? x.w[j] : 0;
			const uint64_t lo = (j - 1 >= 0 && j - 1 < x.n) ? x.w[j - 1] : 0;
			out.w[i] = bs ? ((hi << bs) | (lo >> (64 - bs))) : hi;
		}
		out.n = n;
		norm(out);
	}
	// floor(a / b) for b <= a < b * 2^62 from the top bits (ba = bitlen(a)): the quotient or a little
	// less; the caller corrects upwards
	template <int LM>
	SPR_RR_HD uint64_t quot_estimate(const big<LM>& a, const big<LM>& b, int ba) {
		if (ba <= 128) {
			const unsigned __int128 t0 = bits128(a, 0), t1 = bits128(b, 0);
			return (uint64_t)(t0 / t1);
		}
		const int sh = ba - 128;
		unsigned __int128 t0 = bits128(a, sh), t1 = bits128(b, sh) + 1;
		if (t1 == 0)
			return 0;
		if (t0 < (t1 << 3) || (t1 >> 125)) {
			uint64_t q = 0;
			while (t0 >= t1) { t0 -= t1; q++; }
			return q;
		}
		return (uint64_t)(t0 / t1);
	}

	// Several Euclid steps at once from the leading 62 bits (Knuth, TAOCP 4.5.2, Algorithm L):
	// the steps are taken while the quotients from both bracketing approximations agree, so the
	// 2x2 cofactor matrix is that of the true remainder sequence. Applied only if the new r1 stays
	// above N, so Wang's stopping point is never passed. Returns 1 if it advanced, 0 if the caller
	// should take a single step, 2 on overflow.
	template <int LM>
	SPR_RR_HD int lehmer(big<LM>& r0, big<LM>& r1, big<LM>& s0, big<LM>& s1, bool& s1neg, const big<LM>& N) {
		const int b0 = bitlen(r0);
		if (b0 <= 64) return 0;
		const int h = b0 - 62;
		int64_t uh = (int64_t)shr_word(r0, h), vh = (int64_t)shr_word(r1, h);
		// a step is kept only while the new remainder is certainly above N: with r_j / 2^h = v_j + C_j e0
		// + D_j e1 (e0, e1 in [0, 1) the truncated bits of r0 and r1), r_j > N when v_j - |C_j| - |D_j| >
		// floor(N / 2^h), so the block can run up to the stopping point instead of leaving the last
		// stretch to single steps
		const int64_t hN = (int64_t)shr_word(N, h);
		int64_t A = 1, B = 0, C = 0, D = 1;
		int k = 0;
		while (true) {
			const int64_t y1 = vh + C, y2 = vh + D, x1 = uh + A, x2 = uh + B;
			if (y1 <= 0 || y2 <= 0 || x1 < 0 || x2 < 0) break;
			uint64_t q;
			if ((uint64_t)x1 < 4 * (uint64_t)y1) {
				q = 0;
				uint64_t t = (uint64_t)x1;
				while (t >= (uint64_t)y1) { t -= (uint64_t)y1; q++; }
			}
			else
				q = (uint64_t)x1 / (uint64_t)y1;
			if ((unsigned __int128)q * (uint64_t)y2 > (uint64_t)x2 || (unsigned __int128)(q + 1) * (uint64_t)y2 <= (uint64_t)x2)
				break; // floor(x2 / y2) != q
			const int64_t iq = (int64_t)q;
			const int64_t nC = A - iq * C, nD = B - iq * D, nv = uh - iq * vh;
			if (nv - (nC < 0 ? -nC : nC) - (nD < 0 ? -nD : nD) <= hN)
				break;
			A = C; C = nC;
			B = D; D = nD;
			uh = vh; vh = nv;
			k++;
		}
		if (k == 0 || B == 0) return 0;
		big<LM> nr0, nr1, ns0, ns1;
		if (!combo_sub(nr0, A, r0, B, r1) || !combo_sub(nr1, C, r0, D, r1)) return 2;
		if (cmp(nr1, N) <= 0) return 0; // would pass the stopping point: single steps from here
		const uint64_t aA = (uint64_t)(A < 0 ? -A : A), aB = (uint64_t)(B < 0 ? -B : B);
		const uint64_t aC = (uint64_t)(C < 0 ? -C : C), aD = (uint64_t)(D < 0 ? -D : D);
		if (!combo_add(ns0, aA, s0, aB, s1) || !combo_add(ns1, aC, s0, aD, s1)) return 2;
		r0 = nr0; r1 = nr1; s0 = ns0; s1 = ns1;
		if (k & 1) s1neg = !s1neg;
		return 1;
	}

	// constants (uint64 words): primes[k], C[k*k] (C[i*k+j] = p_j mod p_i), CP[k*k] its Shoup factors,
	// INV[k] = (p_0 ... p_{i-1})^{-1} mod p_i, INVP[k], ONEP[k] = floor(2^64 / p_i), B64[k] = 2^64 mod p_i,
	// B64P[k], M[L], N[L]
	struct cst_view {
		const uint64_t *P, *C, *CP, *INV, *INVP, *ONEP, *B64, *B64P, *M, *N;
		SPR_RR_HD cst_view(const uint64_t* c, int k, int L) {
			P = c; C = P + k; CP = C + k * k; INV = CP + k * k; INVP = INV + k; ONEP = INVP + k;
			B64 = ONEP + k; B64P = B64 + k; M = B64P + k; N = M + L;
		}
	};
	inline size_t cst_words(int k, int L) { return (size_t)k * (2 * k + 6) + 2 * (size_t)L; }

	// a = CRT of the residues resid[i * stride + e], i < k, in [0, m): Garner's mixed radix digits, then
	// Horner; 0, or 2 if a does not fit LM limbs
	template <int LM>
	SPR_RR_HD int crt_value(size_t e, size_t stride, int k, const cst_view& c, const uint64_t* __restrict__ resid, big<LM>& a) {
		// Garner: mixed radix digits v_i, a = v_0 + v_1 p_0 + v_2 p_0 p_1 + ...
		uint64_t v[64];
		for (int i = 0; i < k; i++) {
			const uint64_t pi = c.P[i];
			uint64_t t = 0; // value of the digits so far modulo p_i (Horner from the top digit)
			for (int j = i - 1; j >= 0; j--) {
				uint64_t vj = v[j];
				if (vj >= pi) vj = reduce_word(vj, pi, c.ONEP[i]);
				uint64_t x = mulmod_shoup(c.C[i * k + j], t, c.CP[i * k + j], pi) + vj;
				t = x >= pi ? x - pi : x;
			}
			uint64_t r = resid[(size_t)i * stride + e];
			if (r >= pi) r = reduce_word(r, pi, c.ONEP[i]);
			uint64_t d = r >= t ? r - t : r + pi - t;
			v[i] = mulmod_shoup(c.INV[i], d, c.INVP[i], pi);
		}
		// Horner in place: a = a p_i + v_i from the top digit
		a.w[0] = v[k - 1];
		a.n = v[k - 1] ? 1 : 0;
		for (int i = k - 2; i >= 0; i--) {
			const uint64_t pi = c.P[i];
			uint64_t carry = v[i];
			for (int j = 0; j < a.n; j++) {
				const unsigned __int128 t = (unsigned __int128)a.w[j] * pi + carry;
				a.w[j] = (uint64_t)t;
				carry = (uint64_t)(t >> 64);
			}
			if (carry) {
				if (a.n >= LM) return 2;
				a.w[a.n++] = carry;
			}
		}
		return 0;
	}

	// Wang's half extended Euclid on (m, a) for the CRT value a of k primes: the status; on 0 the
	// fraction is sign(s1) r1 / |s1| (s1neg: negative)
	template <int LM>
	SPR_RR_HD int wang(const big<LM>& a, int k, const cst_view& c, int L, big<LM>& r1, big<LM>& s1, bool& s1neg) {
		big<LM> r0, s0, N;
		r0.n = L; for (int i = 0; i < L; i++) r0.w[i] = c.M[i]; norm(r0);
		N.n = L; for (int i = 0; i < L; i++) N.w[i] = c.N[i]; norm(N);
		r1 = a;
		s0.n = 0; s1.w[0] = 1; s1.n = 1;
		s1neg = false;
		const int bN = bitlen(N);
		while (cmp(r1, N) > 0) {
			if (bitlen(r1) > bN + 1) {
				const int lr = lehmer<LM>(r0, r1, s0, s1, s1neg, N);
				if (lr == 2) return 2;
				if (lr == 1) continue;
			}
			int b0 = bitlen(r0), b1 = bitlen(r1);
			// a quotient beyond 62 bits goes in pieces q 2^s (61 bits each, s decreasing): r0 -= q 2^s r1,
			// s0 += q 2^s s1, until the rest of the quotient fits; the remainder and the cofactor are those of
			// the whole quotient
			while (b0 - b1 >= 62) {
				const int sft = b0 - b1 - 61;
				big<LM> t, u;
				shl(t, r1, sft);
				uint64_t qp = quot_estimate(r0, t, b0);
				if (!submul(r0, t, qp)) return 2;
				while (cmp(r0, t) >= 0) {
					if (!submul(r0, t, 1)) return 2;
					qp++;
				}
				shl(u, s1, sft);
				if (!addmul(s0, u, qp)) return 2;
				b0 = bitlen(r0);
			}
			// the pieces may have taken the whole quotient (r0 < r1 now: the rest is 0); otherwise b0 >= b1
			uint64_t q;
			if (cmp(r0, r1) < 0)
				q = 0;
			else if (b0 <= 128) { // both fit 128 bits: exact
				unsigned __int128 t0 = bits128(r0, 0), t1 = bits128(r1, 0);
				q = (uint64_t)(t0 / t1);
			}
			else {
				// top 128 bits at the same shift; t1 >= 2^65, so floor(t0 / (t1 + 1)) is the true
				// quotient or one less, and never more
				int sh = b0 - 128;
				unsigned __int128 t0 = bits128(r0, sh), t1 = bits128(r1, sh) + 1;
				if (t1 == 0) // the top 128 bits of both are all ones: the correction below finds q
					q = 0;
				else if (t0 < (t1 << 3) || (t1 >> 125)) { // small quotient: subtract
					q = 0;
					while (t0 >= t1) { t0 -= t1; q++; }
				}
				else
					q = (uint64_t)(t0 / t1);
			}
			big<LM> rn = r0;
			if (!submul(rn, r1, q)) return 2;
			while (cmp(rn, r1) >= 0) { // correct upwards
				if (!submul(rn, r1, 1)) return 2;
				q++;
			}
			// (r0, r1) = (r1, r0 - q r1); |s| magnitudes: s_new = s0 + q s1 (signs alternate)
			big<LM> sn = s0;
			if (!addmul(sn, s1, q)) return 2;
			r0 = r1; r1 = rn;
			s0 = s1; s1 = sn;
			s1neg = !s1neg;
		}
		if (cmp(s1, N) > 0) return 1;
		for (int i = 0; i < k; i++)
			if (mod_big(s1, c.P[i], c.ONEP[i], c.B64[i], c.B64P[i]) == 0
				&& mod_big(r1, c.P[i], c.ONEP[i], c.B64[i], c.B64P[i]) == 0)
				return 1;
		return 0;
	}

	// the reconstruction of entry e (residues resid[i * stride + e]): the status; on 0 the fraction is in
	// (r1, s1, s1neg)
	template <int LM>
	SPR_RR_HD int rr_entry(size_t e, size_t stride, int k, const cst_view& c, int L,
		const uint64_t* __restrict__ resid, big<LM>& r1, big<LM>& s1, bool& s1neg) {
		big<LM> a;
		if (crt_value<LM>(e, stride, k, c, resid, a) != 0)
			return 2;
		return wang<LM>(a, k, c, L, r1, s1, s1neg);
	}

	// the constant table of cst_view for k primes (inv_prefix[i] = (p_0 ... p_{i-1})^{-1} mod p_i) and
	// m, N of L limbs; false if a prime is 2^63 or more
	inline bool make_constants(int k, const uint64_t* primes, const uint64_t* inv_prefix, int L,
		const uint64_t* m_limbs, const uint64_t* N_limbs, std::vector<uint64_t>& cst) {
		cst.assign(cst_words(k, L), 0);
		uint64_t *P = cst.data(), *C = P + k, *CP = C + (size_t)k * k, *INV = CP + (size_t)k * k, *INVP = INV + k,
			*ONEP = INVP + k, *B64 = ONEP + k, *B64P = B64 + k, *M = B64P + k, *N = M + L;
		for (int i = 0; i < k; i++) {
			const uint64_t pi = primes[i];
			if (pi >= ((uint64_t)1 << 63))
				return false;
			P[i] = pi;
			for (int j = 0; j < i; j++) {
				C[(size_t)i * k + j] = primes[j] % pi;
				CP[(size_t)i * k + j] = shoup(C[(size_t)i * k + j], pi);
			}
			INV[i] = inv_prefix[i] % pi;
			INVP[i] = shoup(INV[i], pi);
			ONEP[i] = shoup(1, pi);
			B64[i] = (uint64_t)(((unsigned __int128)1 << 64) % pi);
			B64P[i] = shoup(B64[i], pi);
		}
		for (int i = 0; i < L; i++) { M[i] = m_limbs[i]; N[i] = N_limbs[i]; }
		return true;
	}

} // namespace SparseRREF::rr_core

#endif // __int128
#endif // SPARSE_RR_CORE_H
