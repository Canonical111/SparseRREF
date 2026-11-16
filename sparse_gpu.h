/*
	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// GPU (CUDA cores) Schur update over Z/p: the host side, included by sparse_mat.h when
// SPARSERREF_USE_CUDA is defined; the kernels are in sparse_gpu.cu (compiled by nvcc).
//
// One call updates a set of rows by an ordered pivot list, exactly as schur_complete does on the
// CPU: pivot j is applied to a row iff the row holds the pivot's column when j is reached, with
// the same modular arithmetic, so the result is identical. The columns that occur in the pivot
// rows (U) are renumbered 0..nU-1 in increasing order; each row keeps a dense accumulator over U
// (shared memory, or a global slice when U is large); its entries outside U pass through unchanged
// and the kernel merges them back, so rows travel whole and are copied in and out as they are.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

namespace SparseRREF::gpu {
	// the pivot set of one Schur loop
	struct schur_common {
		uint64_t p = 0;
		uint32_t nU = 0, P = 0;
		// pivot rows over U (compact column ids), CSR, normalised (pivot entry 1)
		const uint64_t* pv_off = nullptr; const uint32_t* pv_cid = nullptr; const uint64_t* pv_val = nullptr;
		const uint32_t* piv_cid = nullptr;    // compact column of pivot j
		const int32_t* posU = nullptr;        // pivot position of each compact column, -1 if none
		uint32_t ncol = 0;
		const int32_t* cid_of_col = nullptr;  // compact id of each global column, -1 outside U
		const uint32_t* col_of_cid = nullptr; // global column of each compact id (increasing)
	};
	struct schur_out {
		// row r's new entries (global columns, sorted) occupy col/val[base[r], base[r] + cnt[r]) when
		// status[r] == 0; status 1: arena overflow, cnt[r] is still the exact count (rerun with a
		// larger arena). The arrays live in page-locked buffers of the slot.
		const unsigned long long* base = nullptr; const uint32_t* cnt = nullptr;
		const uint32_t* col = nullptr; const uint64_t* val = nullptr; const uint8_t* status = nullptr;
		double h2d = 0, kernel = 0, d2h = 0; // seconds
	};

	// limits of the kernels on the current device (nU columns, P pivots)
	bool fits(uint32_t nU, uint32_t P);
	bool available();
	// grow-only page-locked host buffer number slot (0..7) for the pivot rows; nullptr on failure.
	// Growing discards the old content.
	void* host_buffer(int slot, size_t bytes);
	// The rows go through two slots, so the host packs and unpacks one part while the GPU works on
	// the other: schur_begin uploads the pivot set; then per part, stage whole rows (which 0: R + 1
	// offsets, 1: global columns, 2: values; page-locked, grow-only, growing discards),
	// submit with the output arena capacity, and wait for the result (valid until the slot is
	// submitted again). Any false: a CUDA failure, the caller computes the rows not yet written back.
	bool schur_begin(const schur_common& c);
	void* schur_stage(int slot, int which, size_t bytes);
	bool schur_submit(int slot, uint32_t R, uint64_t cap);
	bool schur_wait(int slot, schur_out& res);

	// multi-prime CRT + rational reconstruction (see sparse_gpu_rr.cu) over a stream of chunks in two
	// slots: stage a chunk's residues in a slot, submit it, wait for it, read the view, and reuse the
	// slot, so the host converts one chunk while the GPU works on the other.
	struct rr_setup {
		int k = 0;                          // number of primes, each < 2^63 (see rr_supported)
		const uint64_t* primes = nullptr;
		const uint64_t* inv_prefix = nullptr; // (p_0 ... p_{i-1})^{-1} mod p_i, 1 for i = 0
		int L = 0;                          // limbs of m = prod p_i
		const uint64_t* m_limbs = nullptr;
		const uint64_t* N_limbs = nullptr;  // floor(sqrt((m - 1) / 2)), L limbs
		size_t max_entries = 0, max_rows = 0; // largest chunk
	};
	struct rr_view {
		size_t E = 0, R = 0;
		// per entry the descriptor of sparse_qpack.h (bits 0-13 numerator limbs, 14-27 denominator
		// limbs, 28 negative) and in bits 30-31 a status (0 ok, 1 no fraction within the bounds, 2 do it
		// on the CPU)
		const uint32_t* meta = nullptr;
		const uint64_t* row_limb = nullptr; // per row (R + 1): index of its first limb in limbs
		const uint64_t* limbs = nullptr;    // the entries with status 0 in order: numerator limbs, then denominator limbs
		double h2d = 0, kernel = 0, d2h = 0; // seconds on the GPU's clock (d2h includes the limb download)
	};
	// whether k primes with a product of L limbs can be done here (one thread per entry up to 64 primes
	// and 32 limbs, one warp per entry up to 510 limbs), and the largest chunk one submission takes
	bool rr_supported(int k, int L);
	bool rr_warp_path(int k, int L);
	size_t rr_max_entries(int k, int L);
	bool rr_begin(const rr_setup& s);
	uint64_t* rr_stage(int slot);      // page-locked: prime t's residues of the chunk at t * max_entries
	uint64_t* rr_stage_rows(int slot); // page-locked: R + 1 row starts relative to the chunk (last = E)
	bool rr_submit(int slot, size_t E, size_t R);
	bool rr_wait(int slot, rr_view& v); // the view is valid until the slot is submitted again
	void rr_end();                      // also after a failure; waits for both slots

	// A matrix kept on the GPU (Z/p) while an elimination whose pivots are known runs on it: upload
	// the rows once, rescale and update level by level, download once (see sparse_gpu_res.cu).
	// Rows are addressed by their index in the host matrix (< nrow); only uploaded rows may be used.
	// Any false: a CUDA failure or no room; call res_end and compute on the CPU.
	// rows[0..R) with off (R + 1 entry offsets); fill(r0, r1, col, val) writes the entries of rows
	// [r0, r1) (columns sorted within a row) at col / val, which have room for off[r1] - off[r0]; it is
	// called piece by piece (a bounded page-locked buffer), in order
	using res_fill_fn = std::function<void(uint32_t r0, uint32_t r1, uint32_t* col, uint64_t* val)>;
	bool res_begin(uint64_t p, uint32_t nrow, uint32_t ncol, uint32_t R, const uint32_t* rows,
		const uint64_t* off, const res_fill_fn& fill);
	// divide each pivot row by its pivot entry
	bool res_rescale(uint32_t P, const uint32_t* prow, const uint32_t* pcol);
	// update rows[0..R) by the ordered pivots (prow, pcol)[0..P), as schur_complete does
	bool res_schur(uint32_t P, const uint32_t* prow, const uint32_t* pcol, uint32_t R, const uint32_t* rows);
	// back substitution over the ordered pivots (prow, pcol)[0..P) (pivot k's row holds no pivot
	// column before k), on all uploaded rows, in batches of n_split pivots from the end
	bool res_backsub(uint32_t P, const uint32_t* prow, const uint32_t* pcol, uint32_t n_split);
	// the current entries of rows[0..R) (columns only unless with_val), piece by piece in order:
	// take(r0, r1, off, col, val) gets rows [r0, r1) with off[0..r1 - r0] relative to col / val (val
	// null without values; the buffers are valid during the call); take returning false stops
	using res_take_fn = std::function<bool(uint32_t r0, uint32_t r1, const uint64_t* off, const uint32_t* col, const uint64_t* val)>;
	bool res_download_pieces(uint32_t R, const uint32_t* rows, bool with_val, const res_take_fn& take);
	// whether every row of the pivots (prow, pcol)[0..P) holds its pivot column and none holds the column
	// of an earlier pivot of the list (what one pass of res_rescale + res_schur needs): 1 yes, 0 no, -1 a
	// CUDA failure
	int res_level_order(uint32_t P, const uint32_t* prow, const uint32_t* pcol);
	// the rows written by the last res_schur (the others did not hold a pivot column)
	const std::vector<uint32_t>& res_changed();
	// whether the matrix on the GPU can still be read after a failed step (no error that loses the
	// device's state)
	bool res_alive();
	void res_end();
	// The pivot search of one round (pivots_search, method 0) on the rows on the GPU: leftrows and
	// leftcols in the host's order, the column weights set once (res_search_weights). Gives the pivots
	// in the order of pivots_search, whether each row of leftrows is empty, and their entries in all.
	bool res_search_weights(const int64_t* w);
	bool res_search(uint32_t R, const uint32_t* leftrows, uint32_t L, const uint32_t* leftcols,
		std::vector<uint32_t>& prow, std::vector<uint32_t>& pcol, std::vector<uint8_t>& empty, uint64_t& nnz);
	struct res_stats { double rescale = 0, prep = 0, kernel = 0, compact = 0, download = 0, backsub = 0, struct_dl = 0, search = 0, s_count = 0, s_cand = 0, s_greedy = 0; size_t compactions = 0; };
	res_stats& res_profile();

	// whether CUDA has been initialised in this process (does not initialise it)
	bool initialized();
	// creates the CUDA context ahead of use (for a background thread at start-up: the context takes a
	// tenth of a second or more); does not count as initialised above
	void warmup();

	// testing (builds with SPARSERREF_TEST_HOOKS only): SPARSERREF_GPU_FAIL="site:n,..." (or "/" between
	// items) makes the n-th call (from 1) at that site report a failure (sites: rescale, schur, backsub,
	// lost, replay, rr, search); always false otherwise
	bool fail_inject(const char* site);

	// the module's tunables (the library sets them from its exec_policy before a GPU path runs)
	struct options {
		uint64_t transfer_piece = (uint64_t)1 << 22; // entries per page-locked host-device transfer piece
		size_t rr_pinned_bytes = (size_t)512 << 20;  // page-locked memory of the reconstruction
		bool large_u = true;                         // per-call Schur: large U through a global scratch
	};
	void set_options(const options& o);
	const options& get_options();
}
