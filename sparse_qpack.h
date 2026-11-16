/*
	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// Packed rationals: the result of the multi-modular reconstruction before (or instead of) becoming a
// sparse_mat<rat_t>. An entry is a 32-bit descriptor and the limbs of its numerator and denominator
// (absolute values), nothing else: a rational of a few words costs a few words, where an fmpq whose
// parts exceed 62 bits costs two heap-allocated mpz (about 48 bytes each) on top of its 16 bytes.
//
// Descriptor: bits 0-13 numerator limbs, 14-27 denominator limbs (the value 16383 means "the count is
// the next limb of the stream", for parts of 16383 limbs or more), bit 28 negative, bits 30-31 a
// status used by the GPU reconstruction (0 in stored entries).

#ifndef SPARSE_QPACK_H
#define SPARSE_QPACK_H

#include <gmp.h>
#include "sparse_type.h"
#include "sparse_rr_core.h"

namespace SparseRREF::qpack {

	constexpr uint32_t LESC = (1u << 14) - 1;

	struct entry_view {
		const uint64_t* num = nullptr;
		const uint64_t* den = nullptr;
		size_t nl = 0, dl = 0;
		bool neg = false;
	};

	inline size_t entry_limbs(size_t nl, size_t dl) {
		return nl + dl + (nl >= LESC) + (dl >= LESC);
	}
	inline uint32_t meta_make(size_t nl, size_t dl, bool neg) {
		return (uint32_t)std::min<size_t>(nl, LESC) | ((uint32_t)std::min<size_t>(dl, LESC) << 14) | ((uint32_t)neg << 28);
	}
	// limbs of the entry with descriptor m (the stream is needed only for escaped counts)
	inline size_t meta_limbs(uint32_t m, const uint64_t* p) {
		size_t nl = m & LESC, n = 0;
		if (nl == LESC) { nl = p[0]; n++; }
		size_t dl = (m >> 14) & LESC;
		if (dl == LESC) { dl = p[n + nl]; n++; }
		return n + nl + dl;
	}
	// reads the entry at p, returns the start of the next one
	inline const uint64_t* read_entry(const uint64_t* p, uint32_t m, entry_view& v) {
		size_t nl = m & LESC;
		if (nl == LESC) nl = *p++;
		v.num = p; v.nl = nl; p += nl;
		size_t dl = (m >> 14) & LESC;
		if (dl == LESC) dl = *p++;
		v.den = p; v.dl = dl; p += dl;
		v.neg = (m >> 28) & 1;
		return p;
	}
	// writes an entry at p (advanced), returns its descriptor
	inline uint32_t write_entry(uint64_t*& p, const uint64_t* num, size_t nl, const uint64_t* den, size_t dl, bool neg) {
		if (nl >= LESC) *p++ = nl;
		std::memcpy(p, num, nl * 8); p += nl;
		if (dl >= LESC) *p++ = dl;
		std::memcpy(p, den, dl * 8); p += dl;
		return meta_make(nl, dl, neg);
	}

	// the limbs of |f| without copying (tmp holds a small value), and its sign
	inline void fmpz_view(const fmpz* f, const uint64_t*& d, size_t& n, bool& neg, uint64_t& tmp) {
		const fmpz v = *f;
		if (!COEFF_IS_MPZ(v)) {
			neg = v < 0;
			tmp = neg ? (uint64_t)(-v) : (uint64_t)v;
			d = &tmp;
			n = v != 0;
			return;
		}
#ifdef FMPZ_TO_ZZ
		zz_srcptr z = FMPZ_TO_ZZ(v);
		n = (size_t)std::abs(z->size); neg = z->size < 0; d = (const uint64_t*)z->ptr;
#else
		mpz_srcptr z = COEFF_TO_PTR(v);
		n = (size_t)std::abs(z->_mp_size); neg = z->_mp_size < 0; d = (const uint64_t*)z->_mp_d;
#endif
	}
	inline size_t fmpq_limbs(const fmpq* q) {
		return entry_limbs(fmpz_size(fmpq_numref(q)), fmpz_size(fmpq_denref(q)));
	}
	inline uint32_t write_fmpq(uint64_t*& p, const fmpq* q) {
		const uint64_t *nd, *dd;
		size_t nl, dl;
		bool nneg, dneg;
		uint64_t t1, t2;
		fmpz_view(fmpq_numref(q), nd, nl, nneg, t1);
		fmpz_view(fmpq_denref(q), dd, dl, dneg, t2);
		return write_entry(p, nd, nl, dd, dl, nneg);
	}
	inline void set_fmpq(fmpq* q, const entry_view& v) {
		fmpz_set_ui_array(fmpq_numref(q), (const ulong*)v.num, (slong)v.nl);
		if (v.neg)
			fmpz_neg(fmpq_numref(q), fmpq_numref(q));
		fmpz_set_ui_array(fmpq_denref(q), (const ulong*)v.den, (slong)v.dl);
	}

	// x mod p for a little-endian limb array (b64 = 2^64 mod p)
	inline ulong limbs_mod(const uint64_t* x, size_t n, const nmod_t& mod, ulong b64) {
		ulong r = 0;
		for (size_t i = n; i-- > 0;) {
			r = nmod_mul(r, b64, mod);
			r = nmod_add(r, (x[i] >= mod.n ? nmod_set_ui(x[i], mod) : x[i]), mod);
		}
		return r;
	}

	// appends the decimal digits of the limb array (n >= 1, top limb nonzero); scratch is clobbered
	inline void append_decimal(std::string& out, const uint64_t* x, size_t n, std::vector<mp_limb_t>& scratch,
		std::vector<unsigned char>& digits) {
		while (n > 0 && x[n - 1] == 0)
			n--;
		if (n == 0) {
			out.push_back('0');
			return;
		}
		if (n == 1) {
			char buf[24];
			auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), (unsigned long long)x[0]);
			out.append(buf, ptr);
			return;
		}
		scratch.assign(x, x + n);
		scratch.push_back(0);
		digits.resize(n * 20 + 2);
		const size_t len = mpn_get_str(digits.data(), 10, scratch.data(), (mp_size_t)n);
		size_t s = 0;
		while (s + 1 < len && digits[s] == 0)
			s++;
		const size_t at = out.size();
		out.resize(at + len - s);
		for (size_t i = s; i < len; i++)
			out[at + i - s] = (char)('0' + digits[i]);
	}
	// "n" or "n/d", the text of fmpq_get_str
	inline void append_rational(std::string& out, const entry_view& v, std::vector<mp_limb_t>& scratch,
		std::vector<unsigned char>& digits) {
		if (v.neg)
			out.push_back('-');
		append_decimal(out, v.num, v.nl, scratch, digits);
		if (!(v.dl == 1 && v.den[0] == 1)) {
			out.push_back('/');
			append_decimal(out, v.den, v.dl, scratch, digits);
		}
	}

	// malloc'd limb storage that can be shrunk in place
#ifdef SPARSERREF_HAVE_RR_CORE
	static_assert(sizeof(mp_limb_t) == 8 && GMP_NUMB_BITS == 64, "64-bit GMP limbs expected");

	// a running common denominator of consecutive entries (see fast_rr::entry_cd): n limbs, 0 for none;
	// always coprime to the modulus. At most three limbs: the denominators of a row usually divide one of
	// that size, and a larger D makes every test pay a long gcd
	struct common_den {
		static constexpr int max_limbs = 3;
		uint64_t w[max_limbs];
		int n = 0;
#ifdef SPARSERREF_HAVE_RR_CORE
		uint64_t dm[rr_core::max_primes], dmp[rr_core::max_primes]; // D mod p_i and its Shoup factor
#endif
	};

	// CRT + rational reconstruction of one entry from its residues modulo k primes with the fixed-size
	// arithmetic of sparse_rr_core.h (up to 64 primes and 32 limbs), written straight into packed form.
	// The result is FLINT's fmpq_reconstruct_fmpz of the CRT value (same bounds, same fraction).
	//
	// entry_cd() first tries a common denominator D of the entries before (the entries of a row of a
	// reduced echelon form, and often of many rows, share a small one): with a = CRT value, y = D a mod m
	// in (-m/2, m/2), g = gcd(y, D), the fraction (y/g) / (D/g) is in lowest terms and congruent to a
	// (D is coprime to m); if both parts are within FLINT's bounds N it is the reconstruction of a, since
	// a fraction within the bounds is unique. Otherwise the Euclidean reconstruction runs and D takes
	// the new denominator in. Same results, no Euclid steps for most entries.
	class fast_rr {
	public:
		// primes[0..k) with product mod; false if k or the size of mod is beyond the fixed-size path
		bool init(const ulong* primes, size_t k_, const int_t& mod) {
			lm = 0;
			k = (int)k_;
			L = (int)fmpz_size(mod.data());
			if (k < 1 || k > rr_core::max_primes || L < 1 || L > 32)
				return false;
			std::vector<uint64_t> m_limbs(L, 0), N_limbs(L, 0), inv_prefix(k, 1);
			fmpz_get_ui_array((ulong*)m_limbs.data(), L, mod.data());
			int_t N = mod - 1;
			fmpz_fdiv_q_2exp(N.data(), N.data(), 1);
			fmpz_sqrt(N.data(), N.data());
			fmpz_get_ui_array((ulong*)N_limbs.data(), L, N.data());
			for (int i = 1; i < k; i++) {
				const ulong pi = primes[i], pinv = n_preinvert_limb(pi);
				ulong prod = 1;
				for (int j = 0; j < i; j++)
					prod = n_mulmod2_preinv(prod, primes[j] % pi, pi, pinv);
				inv_prefix[i] = n_invmod(prod, pi);
			}
			if (!rr_core::make_constants(k, (const uint64_t*)primes, inv_prefix.data(), L, m_limbs.data(), N_limbs.data(), cst))
				return false;
			M.assign(m_limbs.begin(), m_limbs.end());
			Nl.assign(N_limbs.begin(), N_limbs.end());
			Nn = L;
			while (Nn > 0 && Nl[Nn - 1] == 0) Nn--;
			H.assign(L, 0); // floor(m / 2), m odd
			mpn_rshift((mp_ptr)H.data(), (mp_srcptr)M.data(), L, 1);
			Hn = L;
			while (Hn > 0 && H[Hn - 1] == 0) Hn--;
			P.assign((const uint64_t*)primes, (const uint64_t*)primes + k);
			bN = bits_n(Nl.data(), Nn);
			lm = L <= 6 ? 8 : L <= 14 ? 16 : 34;
			return true;
		}
		bool on() const { return lm != 0; }
		// the entry with residues rk[0..k): 0 written at p (advanced), descriptor in meta; 1 no fraction
		// within the bounds; 2 beyond the limb capacity (FLINT decides)
		int entry(const uint64_t* rk, uint64_t*& p, uint32_t& meta) const {
			switch (lm) {
			case 8: return run<8>(rk, p, meta);
			case 16: return run<16>(rk, p, meta);
			case 34: return run<34>(rk, p, meta);
			default: return 2;
			}
		}
		// as entry(), trying the common denominator D first; D is updated when the Euclidean path ran
		int entry_cd(const uint64_t* rk, common_den& D, uint64_t*& p, uint32_t& meta) const {
			switch (lm) {
			case 8: return run_cd<8>(rk, D, p, meta);
			case 16: return run_cd<16>(rk, D, p, meta);
			case 34: return run_cd<34>(rk, D, p, meta);
			default: return 2;
			}
		}

	private:
		int k = 0, L = 0, lm = 0, Nn = 0, Hn = 0, bN = 0;
		std::vector<uint64_t> cst, M, Nl, H, P;

		static int bits_n(const uint64_t* x, int n) {
			return n == 0 ? 0 : 64 * (n - 1) + 64 - rr_core::clz64(x[n - 1]);
		}
		static int norm_n(const uint64_t* x, int n) {
			while (n > 0 && x[n - 1] == 0) n--;
			return n;
		}
		static int cmp_n(const uint64_t* x, int xn, const uint64_t* y, int yn) {
			if (xn != yn) return xn < yn ? -1 : 1;
			return mpn_cmp((mp_srcptr)x, (mp_srcptr)y, xn);
		}
		// g = gcd(x, y) for x, y > 0 of at most common_den::max_limbs + 32 limbs (clobbers x and y);
		// returns the limbs of g
		static int gcd_n(uint64_t* g, uint64_t* x, int xn, uint64_t* y, int yn) {
			// mpn_gcd wants one odd operand: take out the powers of two first
			const mp_bitcnt_t tx = mpn_scan1((mp_srcptr)x, 0), ty = mpn_scan1((mp_srcptr)y, 0), t = std::min(tx, ty);
			auto shr = [](uint64_t* z, int zn, mp_bitcnt_t b) {
				const int ls = (int)(b / 64), bs = (int)(b % 64);
				if (ls) { std::memmove(z, z + ls, (size_t)(zn - ls) * 8); zn -= ls; }
				if (bs) mpn_rshift((mp_ptr)z, (mp_srcptr)z, zn, bs);
				return norm_n(z, zn);
			};
			xn = shr(x, xn, tx);
			yn = shr(y, yn, ty);
			int gn;
			if (xn >= yn) gn = (int)mpn_gcd((mp_ptr)g, (mp_ptr)x, xn, (mp_ptr)y, yn);
			else gn = (int)mpn_gcd((mp_ptr)g, (mp_ptr)y, yn, (mp_ptr)x, xn);
			if (t) { // g <<= t
				const int ls = (int)(t / 64), bs = (int)(t % 64);
				if (bs) {
					const uint64_t c = mpn_lshift((mp_ptr)g, (mp_srcptr)g, gn, bs);
					if (c) g[gn++] = c;
				}
				if (ls) { std::memmove(g + ls, g, (size_t)gn * 8); std::memset(g, 0, (size_t)ls * 8); gn += ls; }
			}
			return gn;
		}
		// q = x / g (exact division; q has room for xn limbs); returns the limbs of q
		static int div_n(uint64_t* q, const uint64_t* x, int xn, const uint64_t* g, int gn) {
			if (gn == 1) {
				if (g[0] == 1) { std::memcpy(q, x, (size_t)xn * 8); return xn; }
				mpn_divrem_1((mp_ptr)q, 0, (mp_srcptr)x, xn, g[0]);
				return norm_n(q, xn);
			}
			uint64_t r[common_den::max_limbs + 34];
			mpn_tdiv_qr((mp_ptr)q, (mp_ptr)r, 0, (mp_srcptr)x, xn, (mp_srcptr)g, gn);
			return norm_n(q, xn - gn + 1);
		}
		// D = lcm(D, d), or d if the lcm does not fit; none if it would not be coprime to m
		void absorb(common_den& D, const uint64_t* d, int dn) const {
			constexpr int CM = common_den::max_limbs;
			if (dn > CM) // too large to share: D stays for the entries after it
				return;
			if (D.n > 0 && dn == 1 && d[0] == 1) // an integer: D stays
				return;
			uint64_t nd[CM];
			int ndn = dn;
			std::memcpy(nd, d, (size_t)dn * 8);
			if (D.n > 0) {
				uint64_t x[CM], y[CM], g[CM], q[CM];
				std::memcpy(x, D.w, (size_t)D.n * 8);
				std::memcpy(y, d, (size_t)dn * 8);
				const int gn = gcd_n(g, x, D.n, y, dn);
				const int qn = div_n(q, d, dn, g, gn); // d / gcd
				if (D.n + qn <= CM) {
					uint64_t t[2 * CM];
					if (D.n >= qn) mpn_mul((mp_ptr)t, (mp_srcptr)D.w, D.n, (mp_srcptr)q, qn);
					else mpn_mul((mp_ptr)t, (mp_srcptr)q, qn, (mp_srcptr)D.w, D.n);
					const int tn = norm_n(t, D.n + qn);
					if (tn <= CM) { std::memcpy(nd, t, (size_t)tn * 8); ndn = tn; }
				}
			}
			for (int i = 0; i < k; i++) {
				D.dm[i] = mpn_mod_1((mp_srcptr)nd, ndn, P[i]);
				if (D.dm[i] == 0) { D.n = 0; return; }
				D.dmp[i] = rr_core::shoup(D.dm[i], P[i]);
			}
			std::memcpy(D.w, nd, (size_t)ndn * 8);
			D.n = ndn;
		}
		template <int LM>
		int run_cd(const uint64_t* rk, common_den& D, uint64_t*& p, uint32_t& meta) const {
			constexpr int CM = common_den::max_limbs;
			const rr_core::cst_view c(cst.data(), k, L);
			if (D.n > 0) {
				// y = D a mod m, symmetric: the CRT of the residues times D mod p_i
				uint64_t ys[rr_core::max_primes];
				for (int i = 0; i < k; i++)
					ys[i] = rr_core::mulmod_shoup(D.dm[i], rk[i], D.dmp[i], P[i]);
				rr_core::big<LM> yv;
				uint64_t r[LM + CM];
				int rn = 0;
				if (rr_core::crt_value<LM>(0, 1, k, c, ys, yv) == 0) {
					rn = yv.n;
					std::memcpy(r, yv.w, (size_t)rn * 8);
				}
				bool neg = false;
				if (cmp_n(r, rn, H.data(), Hn) > 0) { // y = -(m - r)
					uint64_t u[LM + CM];
					mpn_sub((mp_ptr)u, (mp_srcptr)M.data(), L, (mp_srcptr)r, rn);
					rn = norm_n(u, L);
					std::memcpy(r, u, (size_t)rn * 8);
					neg = true;
				}
				// |y| / g <= N needs |y| <= N D: by bit lengths first (most failures stop here, before the gcd)
				if (rn > 0 && bits_n(r, rn) <= bN + bits_n(D.w, D.n)) {
					uint64_t g[CM], num[LM + CM], den[CM];
					int gn;
					if (D.n == 1) { // one word: gcd(y mod D, D)
						g[0] = n_gcd(mpn_mod_1((mp_srcptr)r, rn, D.w[0]), D.w[0]);
						gn = 1;
					}
					else {
						uint64_t x[LM + CM], y[CM];
						std::memcpy(x, r, (size_t)rn * 8);
						std::memcpy(y, D.w, (size_t)D.n * 8);
						gn = gcd_n(g, x, rn, y, D.n);
					}
					const int nn = div_n(num, r, rn, g, gn);
					const int dn = div_n(den, D.w, D.n, g, gn);
					if (cmp_n(num, nn, Nl.data(), Nn) <= 0 && cmp_n(den, dn, Nl.data(), Nn) <= 0) {
						meta = write_entry(p, num, (size_t)nn, den, (size_t)dn, neg);
						return 0;
					}
				}
			}
			rr_core::big<LM> a;
			if (rr_core::crt_value<LM>(0, 1, k, c, rk, a) != 0 || a.n == 0)
				return 2;
			rr_core::big<LM> r1, s1;
			bool neg = false;
			const int st = rr_core::wang<LM>(a, k, c, L, r1, s1, neg);
			if (st != 0)
				return st;
			if (r1.n == 0)
				return 2;
			meta = write_entry(p, r1.w, (size_t)r1.n, s1.w, (size_t)s1.n, neg);
			absorb(D, s1.w, s1.n);
			return 3;
		}
		template <int LM>
		int run(const uint64_t* rk, uint64_t*& p, uint32_t& meta) const {
			const rr_core::cst_view c(cst.data(), k, L);
			rr_core::big<LM> r1, s1;
			bool neg = false;
			const int st = rr_core::rr_entry<LM>(0, 1, k, c, L, rk, r1, s1, neg);
			if (st != 0)
				return st;
			if (r1.n == 0) // a zero residue vector (no stored entry is zero): FLINT's 0/1
				return 2;
			meta = write_entry(p, r1.w, (size_t)r1.n, s1.w, (size_t)s1.n, neg);
			return 0;
		}
	};
#endif

	struct limb_buf {
		uint64_t* p = nullptr;
		size_t n = 0;
		limb_buf() = default;
		explicit limb_buf(size_t m) { if (m) { p = (uint64_t*)std::malloc(m * 8); if (!p) throw std::bad_alloc(); n = m; } }
		limb_buf(const limb_buf&) = delete;
		limb_buf& operator=(const limb_buf&) = delete;
		limb_buf(limb_buf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
		limb_buf& operator=(limb_buf&& o) noexcept { if (this != &o) { std::free(p); p = o.p; n = o.n; o.p = nullptr; o.n = 0; } return *this; }
		~limb_buf() { std::free(p); }
		void shrink(size_t m) {
			if (m >= n) return;
			if (m == 0) { std::free(p); p = nullptr; n = 0; return; }
			if (auto* q = (uint64_t*)std::realloc(p, m * 8)) p = q;
			n = m;
		}
	};

	// a run of consecutive rows (li0..li1 of the row list) whose entries are stored in one limb array
	struct segment {
		size_t li0 = 0, li1 = 0;
		limb_buf limbs;
		std::vector<uint64_t> row_limb; // li1 - li0 + 1 offsets into limbs
	};

	// A range of rows (whole rows) and its entries. A chunk holds either the residues of every prime
	// used so far (one array per prime) or, once all of its entries were reconstructed, the
	// reconstructed values (descriptors + segments), which stand for the residues: a value n/d agrees
	// with every residue, so they can be recomputed from it (n d^-1 mod p).
	struct chunk {
		size_t li0 = 0, li1 = 0, e0 = 0, e1 = 0;
		std::vector<std::unique_ptr<ulong[]>> resid;
		std::unique_ptr<uint32_t[]> meta;
		std::vector<segment> segs;

		size_t entries() const { return e1 - e0; }
		bool packed() const { return meta != nullptr; }
		void drop_packed() { meta.reset(); segs.clear(); segs.shrink_to_fit(); }
		void drop_resid() { resid.clear(); resid.shrink_to_fit(); }
		// visit(li, j, e, view) for every entry, rows in order, from segment s (one thread per segment)
		template <typename Fn>
		void for_entries(const segment& s, const std::vector<size_t>& eoff, Fn&& visit) const {
			for (size_t li = s.li0; li < s.li1; li++) {
				const uint64_t* p = s.limbs.p + s.row_limb[li - s.li0];
				for (size_t e = eoff[li]; e < eoff[li + 1]; e++) {
					entry_view v;
					p = read_entry(p, meta[e - e0], v);
					visit(li, e - eoff[li], e, v);
				}
			}
		}
	};
}

namespace SparseRREF {
	// entries per formatted block of the writers below
	constexpr size_t write_block_entries = (size_t)1 << 16;

	// Formats blocks 0..nb-1 in parallel (format(b, out) appends block b's text to out) and writes
	// them in order. At most a few blocks per thread are in flight, so the text held in memory stays
	// bounded (it used to be up to the whole output while the first blocks were still being formatted).
	template <typename S, typename Fmt>
	void write_blocks_ordered(S& st, thread_pool& pool, size_t nb, Fmt&& format) {
		const size_t window = 3 * pool.get_thread_count();
		std::deque<std::future<std::string>> inflight;
		size_t next = 0;
		auto submit = [&]() {
			const size_t b = next++;
			inflight.push_back(pool.submit_task([&format, b]() {
				std::string out;
				format(b, out);
				return out;
				}));
			};
		while (next < nb && inflight.size() < window)
			submit();
		while (!inflight.empty()) {
			std::string part = inflight.front().get();
			inflight.pop_front();
			if (next < nb)
				submit();
			st.write(part.data(), part.size());
		}
	}

	// A rational sparse matrix in packed form (see sparse_qpack.h): the rows with entries, in
	// increasing order, their columns, and their values in chunks of whole rows.
	template <typename index_t>
	struct sparse_qmat_packed {
		size_t nrow = 0, ncol = 0;
		std::vector<size_t> rows;           // the rows with entries, increasing
		std::vector<size_t> eoff;           // rows.size() + 1 entry offsets
		std::unique_ptr<index_t[]> col;     // the column of each entry
		std::vector<qpack::chunk> chunks;

		size_t nnz() const { return eoff.empty() ? 0 : eoff.back(); }

		void clear() {
			rows.clear(); rows.shrink_to_fit();
			eoff.clear(); eoff.shrink_to_fit();
			col.reset();
			chunks.clear(); chunks.shrink_to_fit();
		}

		// the plain text format of sparse_mat_write (same bytes as writing the materialized matrix)
		template <typename S>
		void write(S& st, thread_pool* pool) const {
			st << nrow << ' ' << ncol << ' ' << nnz() << '\n';
			// blocks of whole rows inside one segment
			struct blk { size_t c, s, li0, li1; };
			std::vector<blk> blocks;
			for (size_t c = 0; c < chunks.size(); c++)
				for (size_t si = 0; si < chunks[c].segs.size(); si++) {
					const auto& sg = chunks[c].segs[si];
					size_t li = sg.li0;
					while (li < sg.li1) {
						size_t l1 = li, n = 0;
						while (l1 < sg.li1 && (l1 == li || n < write_block_entries_q))
							n += eoff[l1 + 1] - eoff[l1], l1++;
						blocks.push_back({ c, si, li, l1 });
						li = l1;
					}
				}
			auto format = [&](size_t b, std::string& out) {
				const auto& B = blocks[b];
				const auto& ch = chunks[B.c];
				const auto& sg = ch.segs[B.s];
				std::vector<mp_limb_t> scratch;
				std::vector<unsigned char> digits;
				out.reserve((eoff[B.li1] - eoff[B.li0]) * 48);
				char buf[24];
				for (size_t li = B.li0; li < B.li1; li++) {
					const uint64_t* p = sg.limbs.p + sg.row_limb[li - sg.li0];
					for (size_t e = eoff[li]; e < eoff[li + 1]; e++) {
						qpack::entry_view v;
						p = qpack::read_entry(p, ch.meta[e - ch.e0], v);
						auto [p1, e1] = std::to_chars(buf, buf + sizeof(buf), rows[li] + 1);
						out.append(buf, p1);
						out.push_back(' ');
						auto [p2, e2] = std::to_chars(buf, buf + sizeof(buf), (size_t)col[e] + 1);
						out.append(buf, p2);
						out.push_back(' ');
						qpack::append_rational(out, v, scratch, digits);
						out.push_back('\n');
					}
				}
				};
			if (pool != nullptr && pool->get_thread_count() > 1)
				write_blocks_ordered(st, *pool, blocks.size(), format);
			else {
				std::string out;
				for (size_t b = 0; b < blocks.size(); b++) {
					out.clear();
					format(b, out);
					st.write(out.data(), out.size());
				}
			}
		}

		// the result as a sparse_mat<rat_t> (rows not listed are empty); the packed storage is released
		// chunk by chunk as it is converted
		void materialize(sparse_mat<rat_t, index_t>& mat, thread_pool& pool) {
			mat = sparse_mat<rat_t, index_t>(nrow, ncol);
			for (auto& ch : chunks) {
				for (auto& sg : ch.segs) {
					pool.detach_loop<size_t>(sg.li0, sg.li1, [&](size_t li) {
						auto& row = mat[rows[li]];
						const size_t n = eoff[li + 1] - eoff[li];
						row.reserve(n);
						row.resize(n);
						const uint64_t* p = sg.limbs.p + sg.row_limb[li - sg.li0];
						for (size_t j = 0; j < n; j++) {
							const size_t e = eoff[li] + j;
							row(j) = col[e];
							qpack::entry_view v;
							p = qpack::read_entry(p, ch.meta[e - ch.e0], v);
							qpack::set_fmpq(row[j].data(), v);
						}
						});
					pool.wait();
				}
				ch.drop_packed();
				ch.drop_resid();
			}
			clear();
		}

		static constexpr size_t write_block_entries_q = (size_t)1 << 15;
	};
}

#endif
