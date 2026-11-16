/*
	Copyright (C) 2024-2026 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"
#include "sparserref_profile.h"
#include "sparse_qpack.h"
#if defined(__GLIBC__) && !defined(USE_MIMALLOC)
#include <malloc.h>
#elif defined(__APPLE__) && !defined(USE_MIMALLOC)
#include <malloc/malloc.h>
#endif
#ifdef SPARSERREF_USE_CUDA
#include "sparse_gpu.h"
#endif

namespace SparseRREF {

	// first look for rows with only one nonzero value and eliminate them
	// we assume that mat is canonical, i.e. each index is sorted
	// and the result is also canonical
	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz(sparse_mat_subview<T, index_t> mat, std::vector<index_t>& donelist,
		rref_option_t opt) {
		auto localcounter = 0;
		std::unordered_map<size_t, index_t> pivlist;
		bit_array collist(mat.ncol());
		for (size_t a = 0; a < mat.nrow(); a++) {
			size_t i = mat(a);
			if (donelist[i] != index_sval<index_t>())
				continue;
			if (mat[i].nnz() == 1) {
				// a column of negative weight is never a pivot column (see pivots_search);
				// using it here and discarding the pivot afterwards left a row that the
				// rational reconstruction could not agree on across primes
				if (opt->col_weight(mat[i](0)) < 0)
					continue;
				if (!collist[mat[i](0)]) {
					localcounter++;
					pivlist[i] = mat[i](0);
					collist.insert(mat[i](0));
				}
			}
		}

		if (localcounter == 0)
			return localcounter;

		opt->pool.detach_loop(0, mat.nrow(), [&](size_t i) {
			bool is_changed = false;
			auto row = mat(i);
			for (auto [col, val] : mat[i]) {
				if (collist[col]) {
					if (pivlist.contains(row) && pivlist[row] == col)
						val = 1;
					else {
						val = 0;
						is_changed = true;
					}
				}
			}
			if (is_changed) {
				mat[i].canonicalize();
			}
			});

		for (auto [a, b] : pivlist)
			donelist[a] = b;

		opt->pool.wait();

		return localcounter;
	}

	template <typename index_t>
	size_t eliminate_row_with_one_nnz(sparse_mat_subview<bool, index_t> mat, std::vector<index_t>& donelist,
		rref_option_t opt) {
		auto localcounter = 0;
		std::unordered_map<size_t, index_t> pivlist;
		bit_array collist(mat.ncol());
		for (size_t a = 0; a < mat.nrow(); a++) {
			size_t i = mat(a);
			if (donelist[i] != index_sval<index_t>())
				continue;
			if (mat[i].nnz() == 1) {
				// a column of negative weight is never a pivot column (see pivots_search);
				// using it here and discarding the pivot afterwards left a row that the
				// rational reconstruction could not agree on across primes
				if (opt->col_weight(mat[i](0)) < 0)
					continue;
				if (!collist[mat[i](0)]) {
					localcounter++;
					pivlist[i] = mat[i](0);
					collist.insert(mat[i](0));
				}
			}
		}

		if (localcounter == 0)
			return localcounter;

		constexpr index_t sv = index_sval<index_t>();

		opt->pool.detach_loop(0, mat.nrow(), [&](size_t i) {
			auto row = mat(i);
			bool is_changed = false;
			for (size_t j = 0; j < mat[i].nnz(); j++) {
				if (collist[mat[i](j)]) {
					if (!(pivlist.contains(row) && pivlist[row] == mat[i](j))) {
						mat[i](j) = sv; // mark as deleted
						is_changed = true;
					}
				}
			}
			if (is_changed) {
				size_t new_nnz = 0;
				for (size_t j = 0; j < mat[i].nnz(); j++) {
					if (mat[i](j) != sv) {
						if (new_nnz != j)
							mat[i](new_nnz) = mat[i](j);
						new_nnz++;
					}
				}
				mat[i].resize(new_nnz);
			}
			});

		for (auto [a, b] : pivlist)
			donelist[a] = b;

		opt->pool.wait();

		return localcounter;
	}

	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz(sparse_mat<T, index_t>& mat, std::vector<index_t>& donelist,
		rref_option_t opt) {
		return eliminate_row_with_one_nnz(sparse_mat_subview<T, index_t>(mat), donelist, opt);
	}

	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz_rec(sparse_mat_subview<T, index_t> mat, std::vector<index_t>& donelist,
		rref_option_t opt, int max_depth = 1024) {
		int depth = 0;
		size_t localcounter = 0;
		size_t count = 0;
		bool verbose = opt->verbose;

		size_t ndir = mat.ncol();

		size_t oldnnz = mat.nnz();
		int bitlen_nnz = num_digits(oldnnz) + 1;
		int bitlen_ndir = num_digits(ndir);

		// if the number of newly eliminated rows is less than 
		// 0.1% of the total number of eliminated rows, we stop
		do {
			localcounter = eliminate_row_with_one_nnz(mat, donelist, opt);
			count += localcounter;
			if (verbose) {
				oldnnz = mat.nnz();
				progress(opt, "Col")
					.add("%*zu/%zu", bitlen_ndir, count, ndir)
					.add("  rank: %*zu", bitlen_ndir, count)
					.add("  nnz: %*zu", bitlen_nnz, oldnnz)
					.add("  density: %8.6g%%", density_percent(oldnnz, mat.nrow(), mat.ncol()))
					.add("    ");
			}
			depth++;
			if (opt->abort)
				return count;

		} while (localcounter > 0 && depth < max_depth && localcounter * 1000 > count);
		return count;
	}

	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz_rec(sparse_mat<T, index_t>& mat, std::vector<index_t>& donelist,
		rref_option_t opt, int max_depth = 1024) {
		return eliminate_row_with_one_nnz_rec(sparse_mat_subview<T, index_t>(mat), donelist, opt, max_depth);
	}

	// first choose the pivot with minimal col_weight
	// if the col_weight is negative, we do not choose it
	// The pivots of one round: first a left look over the columns (in leftcols order: a column whose
	// rows are all still free takes its row of fewest entries), then a right look over the rows (in
	// leftrows order: a row holding no column taken so far takes its column of fewest rows). Each look
	// is greedy, but what a column or row would take does not depend on the earlier choices, only
	// whether it may take it: a column may if no taken row holds it, a row may if it holds no taken
	// column. So each look goes in chunks: marks record what the pivots taken so far block (the columns
	// of a taken row, the rows of a taken column; the transposes hold exactly the rows of leftrows by
	// column); the candidates of a chunk's unblocked items are found in parallel, and a sequential pass
	// over the chunk takes those still unblocked, marking as it goes. The pivots are those of one
	// sequential pass, in the same order.
	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> pivots_search(
		const sparse_mat<T, index_t>& mat, const std::vector<sparse_mat<bool, index_t>>& tranmat_vec,
		const std::vector<size_t>& leftrows, const std::vector<index_t>& leftcols,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; },
		thread_pool* pool = nullptr, const int64_t* col_w_given = nullptr) {

		const size_t ncol = mat.ncol;
		const bool par_ok = pool != nullptr && pool->get_thread_count() > 1;
		auto par = [&](size_t b0, size_t b1, auto&& body) { // body(begin, end) over [b0, b1)
			if (par_ok && b1 - b0 >= 1024) {
				pool->detach_blocks<size_t>(b0, b1, body, 2 * pool->get_thread_count());
				pool->wait();
			}
			else
				body(b0, b1);
			};
		// chunks start small, so that the first pivots set the marks cheaply, and double up to an eighth
		auto next_end = [](size_t i0, size_t n, size_t& c) {
			const size_t i1 = std::min(n, i0 + c);
			c = std::min(2 * c, std::max<size_t>(8192, n / 8));
			return i1;
			};

		// col_weight is a pure function of the column index, queried for nearly every scanned element
		std::vector<int64_t> col_w_own;
		const int64_t* col_w = col_w_given;
		if (col_w == nullptr) {
			col_w_own.resize(ncol);
			for (index_t i = 0; i < ncol; i++)
				col_w_own[i] = col_weight(i);
			col_w = col_w_own.data();
		}

		std::vector<size_t> tranmat_nnz(ncol, 0);
		par(0, ncol, [&](size_t s, size_t e) {
			for (auto& tranmat : tranmat_vec)
				for (size_t col = s; col < e && col < tranmat.nrow; col++)
					tranmat_nnz[col] += tranmat[col].nnz();
			});

		// left look: the row each unblocked column would take (fewest entries, then smallest index)
		const size_t L = leftcols.size();
		std::vector<index_t> lrow(L);
		std::vector<char> lok(L);
		std::deque<pivot_t<index_t>> pivots;
		bit_array col_blocked(ncol + 1); // columns holding a row taken in the left look
		for (size_t i0 = 0, c = 256; i0 < L;) {
			const size_t i1 = next_end(i0, L, c);
			par(i0, i1, [&](size_t s, size_t e) {
				for (size_t i = s; i < e; i++) {
					lok[i] = 0;
					const index_t col = leftcols[i];
					if (tranmat_nnz[col] == 0 || col_w[col] < 0 || col_blocked.test(col)) // negative weight: never
						continue;
					index_t row = 0;
					size_t mnnz = SIZE_MAX;
					for (auto& tranmat : tranmat_vec)
						for (auto r : tranmat[col].index_span()) {
							const size_t newnnz = mat[r].nnz();
							if (newnnz < mnnz || (newnnz == mnnz && r < row)) {
								row = r;
								mnnz = newnnz;
							}
						}
					if (mnnz != SIZE_MAX) {
						lrow[i] = row;
						lok[i] = 1;
					}
				}
				});
			for (size_t i = i0; i < i1; i++) {
				if (!lok[i] || col_blocked.test(leftcols[i]))
					continue;
				const index_t row = lrow[i];
				pivots.emplace_front(row, leftcols[i]);
				for (auto cc : mat[row].index_span())
					col_blocked.insert(cc);
			}
			i0 = i1;
		}

		// right look: the column each unblocked row would take (fewest rows, then least weight, then
		// smallest index); a row is blocked once it holds a taken column (the left look's to begin with)
		bit_array row_blocked((size_t)mat.nrow + 1);
		auto block_rows_of = [&](index_t col) {
			for (auto& tranmat : tranmat_vec)
				for (auto r : tranmat[col].index_span())
					row_blocked.insert(r);
			};
		for (auto [r, c] : pivots)
			block_rows_of(c);
		const size_t R = leftrows.size();
		std::vector<index_t> rcol(R);
		std::vector<char> rok(R);
		for (size_t i0 = 0, c = 256; i0 < R;) {
			const size_t i1 = next_end(i0, R, c);
			par(i0, i1, [&](size_t s, size_t e) {
				for (size_t i = s; i < e; i++) {
					rok[i] = 0;
					if (row_blocked.test(leftrows[i]))
						continue;
					index_t col = 0;
					size_t mnnz = SIZE_MAX;
					int64_t wcol = 0;
					for (auto c : mat[leftrows[i]].index_span()) {
						const int64_t wc = col_w[c];
						if (wc < 0)
							continue;
						if (tranmat_nnz[c] < mnnz) {
							mnnz = tranmat_nnz[c];
							col = c;
							wcol = wc;
						}
						else if (tranmat_nnz[c] == mnnz) {
							if (wc < wcol) {
								col = c;
								wcol = wc;
							}
							else if (wc == wcol && c < col)
								col = c;
						}
					}
					if (mnnz != SIZE_MAX) {
						rcol[i] = col;
						rok[i] = 1;
					}
				}
				});
			for (size_t i = i0; i < i1; i++) {
				const size_t row = leftrows[i];
				if (!rok[i] || row_blocked.test(row))
					continue;
				pivots.emplace_back(row, rcol[i]);
				block_rows_of(rcol[i]);
			}
			i0 = i1;
		}

		std::vector<pivot_t<index_t>> result(pivots.begin(), pivots.end());
		return result;
	}

	// first choose the pivot with minimal col_weight
	// if the col_weight is negative, we do not choose it
	// only right search version, do not need the full tranpose
	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> pivots_search_right(const sparse_mat<T, index_t>& mat,
		const std::vector<size_t>& leftrows,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; }) {

		std::vector<pivot_t<index_t>> pivots;
		bit_array c_dict((size_t)mat.ncol + 1);

		// col_weight is a pure function of the column index and is queried for nearly
		// every scanned element below, so evaluate it once per column
		std::vector<int64_t> col_w(mat.ncol);
		for (index_t i = 0; i < mat.ncol; i++)
			col_w[i] = col_weight(i);

		for (auto row : leftrows) {
			index_t col = 0;
			size_t mnnz = SIZE_MAX;
			int64_t wcol = 0;
			bool flag = true;

			for (auto c : mat[row].index_span()) {
				int64_t wc = col_w[c];
				// negative weight means that we do not want to select this column
				if (wc < 0)
					continue;
				flag = !c_dict.test(c);
				if (!flag)
					break;
				if (mnnz > 0) {
					mnnz = 0;
					col = c;
					wcol = wc;
				}
				// make the result stable
				else if (mnnz == 0) {
					if (wc < wcol) {
						col = c;
						wcol = wc;
					}
					else if (wc == wcol && c < col)
						col = c;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.emplace_back(row, col);
				c_dict.insert(col);
			}
		}

		return pivots;
	}


	// For a pivot column, the rows holding a nonzero in it. Only the pivot columns are built:
	// the pivot rows are counted into them over one static split, then written in the same
	// split. The counters are indexed by block, not by thread -- the pool may schedule the two
	// passes differently.
	template <typename index_t>
	struct pivot_transpose_t {
		std::vector<index_t> cols; // the pivot columns, increasing
		std::vector<index_t> data; // the rows holding cols[o], in cols order
		std::vector<size_t> ptr;   // ptr[o] .. ptr[o + 1] is the range of cols[o]

		// the rows holding column c, empty when c is not a pivot column
		std::span<const index_t> rows_of(const index_t c) const {
			const auto it = std::lower_bound(cols.begin(), cols.end(), c);
			if (it == cols.end() || *it != c)
				return {};
			const size_t o = (size_t)(it - cols.begin());
			return std::span<const index_t>(data.data() + ptr[o], ptr[o + 1] - ptr[o]);
		}
	};

	template <typename T, typename index_t>
	pivot_transpose_t<index_t> build_pivot_transpose(const sparse_mat<T, index_t>& mat,
		const std::vector<pivot_t<index_t>>& pivots, thread_pool& pool) {

		const size_t rank = pivots.size();
		pivot_transpose_t<index_t> res;
		res.ptr.assign(rank + 1, 0);
		if (rank == 0)
			return res;

		res.cols.resize(rank);
		for (size_t i = 0; i < rank; i++)
			res.cols[i] = pivots[i].c;
		std::sort(res.cols.begin(), res.cols.end());

		// Every entry looks its column up once, so which map pays depends on how many columns are
		// pivot columns. A bitmap plus a search over the pivot columns rejects the other entries
		// with one test, but when most columns are pivots nothing is rejected and the search is
		// paid for nearly every entry; a dense column -> ordinal table is one access, at ncol
		// entries per call (measured over 12.3M entries with 83% pivot columns: 22 ms against
		// 168 ms). The table is only taken when it is bounded by the pivot count.
		const bool dense_ord = rank * 64 >= mat.ncol;
		std::vector<index_t> ord;
		std::vector<uint64_t> bits;
		if (dense_ord) {
			ord.assign(mat.ncol, index_sval<index_t>());
			for (size_t o = 0; o < rank; o++)
				ord[(size_t)res.cols[o]] = (index_t)o;
		}
		else {
			bits.assign((mat.ncol >> 6) + 1, 0);
			for (auto c : res.cols)
				bits[(size_t)c >> 6] |= mask_table[(size_t)c & 63];
		}

		const entry_blocks_t blks(0, rank, pool.get_thread_count());
		const size_t nblk = blks.nblk;
		// counters per (block, pivot column), turned into cursors by the scan below
		std::vector<size_t> cursor(nblk * rank, 0);

		// both passes are written once and handed a monomorphic lookup, so the inner loop carries
		// no test on which map is in use
		auto count_block = [&](const auto& lookup, const size_t blk) {
			auto* my = cursor.data() + blk * rank;
			for (size_t i = blks.start(blk); i < blks.end(blk); i++)
				for (auto c : mat[pivots[i].r].index_span()) {
					const size_t o = lookup(c);
					if (o < rank)
						my[o]++;
				}
			};
		auto write_block = [&](const auto& lookup, const size_t blk) {
			for (size_t i = blks.start(blk); i < blks.end(blk); i++) {
				const auto r = pivots[i].r;
				for (auto c : mat[r].index_span()) {
					const size_t o = lookup(c);
					if (o < rank)
						res.data[cursor[blk * rank + o]++] = r;
				}
			}
			};
		auto prefix = [&]() {
			size_t nnz = 0;
			for (size_t o = 0; o < rank; o++) {
				res.ptr[o] = nnz;
				for (size_t blk = 0; blk < nblk; blk++) {
					const size_t n = cursor[blk * rank + o];
					cursor[blk * rank + o] = nnz;
					nnz += n;
				}
			}
			res.ptr[rank] = nnz;
			res.data.resize(nnz);
			};

		if (dense_ord) {
			auto lookup = [&](const index_t c) -> size_t {
				const auto o = ord[(size_t)c];
				return (o == index_sval<index_t>()) ? rank : (size_t)o;
				};
			pool.detach_sequence(0, nblk, [&](const size_t blk) { count_block(lookup, blk); });
			pool.wait();
			prefix();
			pool.detach_sequence(0, nblk, [&](const size_t blk) { write_block(lookup, blk); });
			pool.wait();
		}
		else {
			auto lookup = [&](const index_t c) -> size_t {
				if (!((bits[(size_t)c >> 6] >> ((size_t)c & 63)) & 1ull))
					return rank; // not a pivot column, never queried
				return (size_t)(std::lower_bound(res.cols.begin(), res.cols.end(), c) - res.cols.begin());
				};
			pool.detach_sequence(0, nblk, [&](const size_t blk) { count_block(lookup, blk); });
			pool.wait();
			prefix();
			pool.detach_sequence(0, nblk, [&](const size_t blk) { write_block(lookup, blk); });
			pool.wait();
		}

		return res;
	}

	// upper solver : ordering = -1
	// lower solver : ordering = 1
	// tranmat in this version is precomputed
	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat,
		const pivot_transpose_t<index_t>& tranmat,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {
		bool verbose = opt->verbose;
		// a non-positive print_step would be an undefined modulo below; take it as
		// "print every row", which is what the other progress sites do
		size_t printstep = opt->print_step > 0 ? (size_t)opt->print_step : 1;
		auto& pool = opt->pool;

		bit_array rowlist(mat.nrow);
		for (auto [r, c] : pivots)
			rowlist.insert(r);

		size_t count = 0;
		bool printed = false;
		size_t nthreads = pool.get_thread_count();
		std::vector<index_t> thecol;
		// the reference time is kept across the whole loop, so that the reported
		// speed is the average over the rows handled since the last report
		auto start = SparseRREF::clocknow();
		for (size_t i = 0; i < pivots.size(); i++) {
			size_t index = i;
			if (ordering < 0)
				index = pivots.size() - 1 - i;
			if (opt->abort) {
				pool.purge();
				pool.wait();
				return;
			}
			auto [row, col] = pivots[index];
			thecol.clear();
			for (auto r : tranmat.rows_of(col)) {
				if (rowlist[r] && r != row)
					thecol.push_back(r);
			}

			// every row of thecol is reduced by the same pivot row and only reads
			// mat[row], so the updates are independent and may run in any order
			auto eliminate = [&](auto j) {
				const auto r = thecol[j];
				if constexpr (std::is_same_v<T, bool>) {
					sparse_vec_add(mat[r], mat[row], F);
				}
				else {
					sparse_vec_sub_mul(mat[r], mat[row], *mat.find(r, col), F);
				}
			};

			if (thecol.size() < 20 * nthreads) {
				// narrow columns are the common case, and dispatching a handful of
				// updates through the pool costs far more in wake-ups and
				// synchronization than the elimination itself (a 3 element batch was
				// measured at ~16us of pool overhead against ~0.5us of work)
				for (size_t j = 0; j < thecol.size(); j++)
					eliminate(j);
			}
			else {
				pool.detach_loop<index_t>(0, thecol.size(), eliminate,
					thecol.size() / 10);
			}
			pool.wait();

			count++;
			if (verbose && (i % printstep == 0 || i == pivots.size() - 1) && thecol.size() > 1) {
				auto end = SparseRREF::clocknow();
				auto now_nnz = mat.nnz();
				progress(opt, "Row")
					.add("%zu/%zu", i + 1, pivots.size())
					.add("  row to eliminate: %zu", thecol.size() - 1)
					.add("  nnz: %zu", now_nnz)
					.add("  density: %g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
					.add("  speed: %g row/s", rate_per_second((double)count, SparseRREF::usedtime(start, end)));
				start = end;
				count = 0;
				printed = true;
			}
		}
		if (printed)
			progress_end(opt);
	}
	
	// in this version, we compute the transpose inside
	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {

		auto& pool = opt->pool;

		if (opt->abort)
			return;

		const auto tranmat = build_pivot_transpose(mat, pivots, pool);
		triangular_solver(mat, tranmat, pivots, F, opt, ordering);
	}

	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat,
		std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {
		std::vector<pivot_t<index_t>> n_pivots;
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		triangular_solver(mat, n_pivots, F, opt, ordering);
	}

	// dot_product
	template <typename T, typename index_t>
	sparse_vec<T, index_t> sparse_mat_dot_sparse_vec(
		const sparse_mat<T, index_t>& mat,
		const sparse_vec<T, index_t>& vec, const field_t& F) {

		sparse_vec<T, index_t> result;

		if (vec.nnz() == 0 || mat.nnz() == 0)
			return result;

		T tmp;
		for (size_t i = 0; i < mat.nrow; i++) {
			tmp = sparse_vec_dot(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return result;
	}

	template <typename T, typename index_t>
	sparse_vec<T, index_t> sparse_mat_dot_dense_vec(
		const sparse_mat<T, index_t>& mat, const T* vec, const field_t& F) {

		sparse_vec<T, index_t> result;

		if (mat.nnz() == 0)
			return result;

		T tmp = 0;
		for (size_t i = 0; i < mat.nrow; i++) {
			tmp = sparse_vec_dot_dense_vec(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return result;
	}

	// A set of column indices supporting O(1) insert/erase/test/enumeration. On top
	// of the plain bit_array we keep an exact counter (so nnz() is O(1)) and a bitmap
	// with one bit per *word* of the bit_array that may be non-empty, so that the
	// final enumeration only visits the words that were touched instead of the whole
	// ncol/64 range. The word summary is only ever set, never cleared (clearing it
	// would need a branch on every erase); a stale bit is harmless because the
	// enumeration skips empty words and clears the summary on the way out.
	struct alignas(64) nonzero_cols {
		bit_array bits;
		bit_array word_bits;
		size_t nz = 0;

		explicit nonzero_cols(size_t ncol) : bits(ncol), word_bits((ncol >> 6) + 1) {}

		void clear() {
			bits.clear();
			word_bits.clear();
			nz = 0;
		}

		void insert(const size_t val) {
			auto idx = val >> 6;
			bits.data[idx] |= mask_table[val & 63];
			word_bits.data[idx >> 6] |= mask_table[idx & 63];
			nz++;
		}

		void erase(const size_t val) {
			bits.erase(val);
			nz--;
		}

		void xor_insert(const size_t val) {
			if (bits.test(val)) {
				bits.erase(val);
				nz--;
			}
			else {
				insert(val);
			}
		}

		bool test(const size_t val) const {
			return bits.test(val);
		}

		size_t nnz() const {
			return nz;
		}

		// visit(pos, index) in increasing index order, clearing the structure on the way out
		template <typename Fn>
		void enumerate_and_clear(Fn&& visit) {
			size_t pos = 0;
			auto& wdata = word_bits.data;
			for (size_t i = 0; i < wdata.size(); i++) {
				uint64_t wb = wdata[i];
				if (wb == 0)
					continue;
				wdata[i] = 0;
				while (wb) {
					auto idx = (i << 6) + ctz(wb);
					wb &= wb - 1;
					uint64_t word = bits.data[idx];
					bits.data[idx] = 0;
					auto base = idx << 6;
					while (word) {
						visit(pos, base + ctz(word));
						pos++;
						word &= word - 1;
					}
				}
			}
			nz = 0;
		}

		// write the set indices in increasing order into ptr, then clear the structure
		template <typename T>
		void nonzero_and_clear(T* ptr) {
			enumerate_and_clear([&](const size_t pos, const size_t idx) { ptr[pos] = (T)idx; });
		}

		// the same, picking the values of src up in the same pass and resetting those slots: the
		// reset is a store to a line the read just brought in, so it is free, and it keeps the
		// rule "the cache is zero outside the set" for every caller
		template <typename T, typename V>
		void nonzero_and_clear(T* ptr, V* src, V* vals) {
			enumerate_and_clear([&](const size_t pos, const size_t idx) {
				ptr[pos] = (T)idx;
				vals[pos] = src[idx];
				src[idx] = 0;
				});
		}
	};

	// The per thread dense cache of one output row: a slot is always written before it is read
	// (an entry that the set does not hold stores a 0 in its slot), so a trivially constructible
	// T is taken raw -- the ncol * nthreads zeros used to dominate a wide product. Over QQ the
	// slot has to be constructed before it can be assigned, so that field keeps the vector.
	template <typename T>
	struct dense_cache_t {
		T* ptr = nullptr;
		std::vector<T> zeroed;

		explicit dense_cache_t(const size_t n) {
			if constexpr (std::is_trivially_constructible_v<T>)
				ptr = s_malloc<T>(n);
			else {
				zeroed = std::vector<T>(n);
				ptr = zeroed.data();
			}
		}
		~dense_cache_t() {
			if constexpr (std::is_trivially_constructible_v<T>)
				s_free(ptr);
		}
		dense_cache_t(const dense_cache_t&) = delete;
		dense_cache_t& operator=(const dense_cache_t&) = delete;

		T* data() { return ptr; }
	};

	// A = B * C
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_mul(
		const sparse_mat<T, index_t>& B, const sparse_mat<T, index_t>& C,
		const field_t& F, thread_pool* pool = nullptr) {

		sparse_mat<T, index_t> A(B.nrow, C.ncol);
		// The pool costs a task and a scratch per thread, and the scratch is sized by ncol, so
		// short rows are better done serially; the crossover is roughly a fixed number of
		// products per thread.
		const size_t products = B.nnz() * (C.nrow ? C.nnz() / C.nrow : 0);
		if (pool != nullptr && products < mul_parallel_threshold * pool->get_thread_count())
			pool = nullptr;

		auto nthreads = 1;
		if (pool)
			nthreads = pool->get_thread_count();

		dense_cache_t<T> cache_densed_mat(A.ncol * nthreads);
		std::vector<nonzero_cols> nonzero_c(nthreads, nonzero_cols(A.ncol));

		auto pp = F.get_prime();

		auto method = [&](size_t id, size_t i) {
			auto& therow = B[i];
			if (therow.nnz() == 0)
				return;
			if (therow.nnz() == 1) {
				A[i] = C[therow(0)];
				sparse_vec_rescale(A[i], therow[0], F);
				return;
			}
			auto cache_dense_vec = cache_densed_mat.data() + id * A.ncol;
			// the set is empty here: nonzero_and_clear emptied it at the end of the row above
			auto& nonzero_c_vec = nonzero_c[id];

			T scalar = therow[0];
			ulong e_pr;
			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(scalar, pp);
			}
			for (auto [ind, val] : C[therow(0)]) {
				nonzero_c_vec.insert(ind);
				if constexpr (std::is_same_v<T, ulong>) {
					cache_dense_vec[ind] = n_mulmod_shoup(scalar, val, e_pr, pp);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					cache_dense_vec[ind] = scalar * val;
				}
			}

			for (size_t j = 1; j < therow.nnz(); j++) {
				scalar = therow[j];
				if constexpr (std::is_same_v<T, ulong>) {
					e_pr = n_mulmod_precomp_shoup(scalar, pp);
				}
				for (auto [ind, val] : C[therow(j)]) {
					if (!nonzero_c_vec.test(ind)) {
						nonzero_c_vec.insert(ind);
						cache_dense_vec[ind] = 0;
					}
					if constexpr (std::is_same_v<T, ulong>) {
						cache_dense_vec[ind] = _nmod_add(cache_dense_vec[ind],
							n_mulmod_shoup(scalar, val, e_pr, pp), F.mod);
					}
					else if constexpr (std::is_same_v<T, rat_t>) {
						cache_dense_vec[ind] += scalar * val;
					}
					if (cache_dense_vec[ind] == 0)
						nonzero_c_vec.erase(ind);
				}
			}

			const size_t nz = nonzero_c_vec.nnz();
			A[i].reserve(nz);
			A[i].resize(nz);
			nonzero_c_vec.nonzero_and_clear(A[i].indices, cache_dense_vec, A[i].entries);
			};

		if (pool) {
			pool->detach_loop(0, B.nrow, [&](size_t i) {
				method(SparseRREF::thread_id(), i);
				}, ((B.nrow < 20 * nthreads) ? 0 : B.nrow / 10));
			pool->wait();
		}
		else {
			for (size_t i = 0; i < B.nrow; i++)
				method(0, i);
		}

		return A;
	}

	// a helper class to store the buffer for schur_complete
	// it is used to avoid frequent memory allocation and deallocation
	template <typename T, typename index_t>
	struct schur_helper_buffer {
		size_t nthreads = 0;
		size_t ncol = 0;
		std::vector<T> cache_densed_mat;
		std::vector<nonzero_cols> nonzero_cs;
		std::vector<std::vector<index_t>> add_lists;
		std::vector<std::vector<index_t>> remove_lists;
		// position of each pivot column in the current pivot list (-1: not a pivot column), set by
		// index_pivots() around one parallel Schur loop; lets a row visit only the pivots it meets
		std::vector<int32_t> pivot_pos;
		std::vector<std::vector<int32_t>> heaps;
		bool indexed = false;

		schur_helper_buffer(size_t nthreads_, size_t ncol_) : nthreads(nthreads_), ncol(ncol_) {
			cache_densed_mat = std::vector<T>(nthreads * ncol);
			nonzero_cs = std::vector<nonzero_cols>(nthreads, nonzero_cols(ncol));
			add_lists = std::vector<std::vector<index_t>>(nthreads);
			remove_lists = std::vector<std::vector<index_t>>(nthreads);
			heaps = std::vector<std::vector<int32_t>>(nthreads);
		}

		// the pivots must have distinct columns; call unindex_pivots with the same list afterwards
		void index_pivots(const std::vector<pivot_t<index_t>>& pivots) {
			if (pivots.size() > (size_t)INT32_MAX)
				return; // stays unindexed: the scanning update is used
			if (pivot_pos.size() != ncol)
				pivot_pos.assign(ncol, -1);
			for (size_t j = 0; j < pivots.size(); j++)
				pivot_pos[pivots[j].c] = (int32_t)j;
			indexed = true;
		}
		void unindex_pivots(const std::vector<pivot_t<index_t>>& pivots) {
			if (!indexed)
				return;
			for (auto& p : pivots)
				pivot_pos[p.c] = -1;
			indexed = false;
		}

		~schur_helper_buffer() = default;
		schur_helper_buffer(const schur_helper_buffer&) = delete;
		schur_helper_buffer& operator=(const schur_helper_buffer&) = delete;
		schur_helper_buffer(schur_helper_buffer&&) = default;
		schur_helper_buffer& operator=(schur_helper_buffer&&) = default;
	};

	// define a helper struct to store temporary variables for single-threaded schur_complete
	// pointers are used to avoid copying large data structures
	// but do not manage the memory of the pointers directly, the caller should manage the memory of the pointers
	template <typename T, typename index_t>
	struct schur_helper {
		T* cache_densed_vec = nullptr;
		nonzero_cols* nonzero_c = nullptr;
		std::vector<index_t>* add_list = nullptr;
		std::vector<index_t>* remove_list = nullptr;
		const int32_t* pivot_pos = nullptr; // null unless the caller indexed the pivot list
		std::vector<int32_t>* heap = nullptr;

		schur_helper(schur_helper_buffer<T, index_t>& buffer, size_t thread_id) {
			cache_densed_vec = buffer.cache_densed_mat.data() + thread_id * buffer.ncol;
			nonzero_c = &buffer.nonzero_cs[thread_id];
			add_list = &buffer.add_lists[thread_id];
			remove_list = &buffer.remove_lists[thread_id];
			if (buffer.indexed)
				pivot_pos = buffer.pivot_pos.data();
			heap = &buffer.heaps[thread_id];
		}
	};

	// indexes a pivot list in the buffer for the lifetime of one parallel Schur loop
	template <typename T, typename index_t>
	struct schur_index_guard {
		schur_helper_buffer<T, index_t>& buffer;
		const std::vector<pivot_t<index_t>>& pivots;
		schur_index_guard(schur_helper_buffer<T, index_t>& b, const std::vector<pivot_t<index_t>>& p)
			: buffer(b), pivots(p) {
			if constexpr (!std::is_same_v<T, bool>)
				buffer.index_pivots(pivots);
		}
		~schur_index_guard() { buffer.unindex_pivots(pivots); }
		schur_index_guard(const schur_index_guard&) = delete;
		schur_index_guard& operator=(const schur_index_guard&) = delete;
	};

	// The strategy of the Schur update below. The dense one asks the set of the columns present in the
	// row about every pivot column; the buffered one keeps a histogram of that set, so that an absent
	// pivot column is rejected with one read of a small array before the set is consulted. The dense
	// case uses an empty filter, so its loop compiles to exactly what it was before.
	struct no_pivot_filter {
		bool absent(const size_t) const { return false; }
		void add(const size_t) {}
		void remove(const size_t) {}
	};

	template <size_t buffer_bit>
	struct pivot_histogram {
		static constexpr size_t buffer_size = (size_t)1 << buffer_bit;
		std::array<size_t, buffer_size> buffer = { 0 };

		bool absent(const size_t c) const { return buffer[c % buffer_size] == 0; }
		void add(const size_t ind) { buffer[ind % buffer_size]++; }
		void remove(const size_t ind) { buffer[ind % buffer_size]--; }
	};

	// make sure nonzero_c and tmpvec are cleared before calling this function
	// after this function, nonzero_c and tmpvec are also cleared
	// the pivots must have distinct rows and distinct columns, none of them row k: a column given twice
	// would be inserted into and erased from the set twice, which desynchronises its count
	template <typename T, typename index_t, typename Filter = no_pivot_filter>
		requires (!std::is_same_v<T, bool>)
	void schur_complete(sparse_mat<T, index_t>& mat, size_t k,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, schur_helper<T, index_t>& helper) {

		if (mat[k].nnz() == 0)
			return;

		T* tmpvec = helper.cache_densed_vec;
		auto& nonzero_c = *helper.nonzero_c;
		auto& add_list = *helper.add_list;
		auto& remove_list = *helper.remove_list;

		Filter filter;
		for (auto [ind, val] : mat[k]) {
			nonzero_c.insert(ind);
			tmpvec[ind] = val;
			filter.add(ind);
		}

		ulong e_pr;
		auto pp = F.get_prime();
		for (auto [r, c] : pivots) {
			if (filter.absent(c) || !nonzero_c.test(c))
				continue;

			T entry = tmpvec[c];
			add_list.clear();
			remove_list.clear();

			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(entry, pp);
			}
			for (auto [ind, val] : mat[r]) {
				bool old_c = tmpvec[ind] == 0;
				if constexpr (std::is_same_v<T, ulong>) {
					tmpvec[ind] = _nmod_sub(tmpvec[ind],
						n_mulmod_shoup(entry, val, e_pr, pp), F.mod);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					tmpvec[ind] -= entry * val;
				}
				else {
					tmpvec[ind] = scalar_sub(tmpvec[ind], scalar_mul(entry, val, F), F);
				}
				if (tmpvec[ind] == 0) {
					remove_list.push_back(ind);
				}
				else {
					if (old_c)
						add_list.push_back(ind);
				}
			}
			for (auto ind : add_list) {
				nonzero_c.insert(ind);
				filter.add(ind);
			}
			for (auto ind : remove_list) {
				nonzero_c.erase(ind);
				filter.remove(ind);
			}
		}

		auto nnz = nonzero_c.nnz();
		if (mat[k].alloc() < nnz) {
			mat[k].reserve(nnz, false);
		}
		mat[k].resize(nnz);
		nonzero_c.nonzero_and_clear(mat[k].indices, tmpvec, mat[k].entries);
	}

	// tempvec is of useless, only nonzero_c is used
	template <typename index_t>
	void schur_complete(sparse_mat<bool, index_t>& mat, size_t k,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, schur_helper<bool, index_t>& helper) {

		auto& nonzero_c = *helper.nonzero_c;

		auto nk = mat[k].nnz();
		if (nk == 0)
			return;

		for (size_t i = 0; i < nk; i++)
			nonzero_c.insert(mat[k](i));

		for (auto [r, c] : pivots) {
			if (!nonzero_c.test(c))
				continue;

			auto nr = mat[r].nnz();
			for (size_t i = 0; i < nr; i++)
				nonzero_c.xor_insert(mat[r](i));
		}

		auto nnz = nonzero_c.nnz();
		if (mat[k].alloc() < nnz) {
			mat[k].reserve(nnz, false);
		}
		mat[k].resize(nnz);
		nonzero_c.nonzero_and_clear(mat[k].indices);
	}

	// the buffered strategy: a histogram of the row rejects an absent pivot column with one array read
	template <typename T, typename index_t, size_t buffer_bit>
	void schur_complete_buffer(sparse_mat<T, index_t>& mat, size_t k,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, schur_helper<T, index_t>& helper) {

		if constexpr (std::is_same_v<T, bool>) {
			schur_complete(mat, k, pivots, F, helper); // a bool row has no values to update
		}
		else {
			schur_complete<T, index_t, pivot_histogram<buffer_bit>>(mat, k, pivots, F, helper);
		}
	}

	// The same update, visiting only the pivots whose column the row holds. The scanning version
	// tests every pivot of the list for every row, which is O(#pivots) per row even when the row
	// meets a handful of them (a first round of 10^5 pivots over 10^5 short rows is 10^10 tests).
	// Here the row's pivot columns sit in a min-heap of pivot positions. Pivot j is applied iff the
	// row holds column c_j when j is reached, exactly as in the scan: a column of a later pivot that
	// becomes nonzero is pushed, one that becomes zero again is skipped when popped, and fill in the
	// column of an earlier pivot is not revisited (the scan never goes back either). Same operations
	// in the same order, so the result is identical.
	template <typename T, typename index_t>
		requires (!std::is_same_v<T, bool>)
	void schur_complete_indexed(sparse_mat<T, index_t>& mat, size_t k,
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, schur_helper<T, index_t>& helper) {

		if (helper.pivot_pos == nullptr) {
			schur_complete<T, index_t>(mat, k, pivots, F, helper);
			return;
		}
		if (mat[k].nnz() == 0)
			return;

		T* tmpvec = helper.cache_densed_vec;
		auto& nonzero_c = *helper.nonzero_c;
		auto& add_list = *helper.add_list;
		auto& remove_list = *helper.remove_list;
		const int32_t* pos = helper.pivot_pos;
		auto& heap = *helper.heap;
		heap.clear();
		constexpr auto later = std::greater<int32_t>{}; // min-heap

		for (auto [ind, val] : mat[k]) {
			nonzero_c.insert(ind);
			tmpvec[ind] = val;
			if (pos[ind] >= 0)
				heap.push_back(pos[ind]);
		}
		std::make_heap(heap.begin(), heap.end(), later);

		ulong e_pr;
		auto pp = F.get_prime();
		int32_t last = -1;
		while (!heap.empty()) {
			std::pop_heap(heap.begin(), heap.end(), later);
			const int32_t j = heap.back();
			heap.pop_back();
			if (j <= last)
				continue; // a column pushed twice (it became zero and nonzero again)
			last = j;
			auto [r, c] = pivots[j];
			if (!nonzero_c.test(c))
				continue;

			T entry = tmpvec[c];
			add_list.clear();
			remove_list.clear();

			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(entry, pp);
			}
			for (auto [ind, val] : mat[r]) {
				bool old_c = tmpvec[ind] == 0;
				if constexpr (std::is_same_v<T, ulong>) {
					tmpvec[ind] = _nmod_sub(tmpvec[ind],
						n_mulmod_shoup(entry, val, e_pr, pp), F.mod);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					tmpvec[ind] -= entry * val;
				}
				else {
					tmpvec[ind] = scalar_sub(tmpvec[ind], scalar_mul(entry, val, F), F);
				}
				if (tmpvec[ind] == 0) {
					remove_list.push_back(ind);
				}
				else {
					if (old_c)
						add_list.push_back(ind);
				}
			}
			for (auto ind : add_list) {
				nonzero_c.insert(ind);
				if (pos[ind] > j) {
					heap.push_back(pos[ind]);
					std::push_heap(heap.begin(), heap.end(), later);
				}
			}
			for (auto ind : remove_list)
				nonzero_c.erase(ind);
		}

		auto nnz = nonzero_c.nnz();
		if (mat[k].alloc() < nnz) {
			mat[k].reserve(nnz, false);
		}
		mat[k].resize(nnz);
		nonzero_c.nonzero_and_clear(mat[k].indices, tmpvec, mat[k].entries);
	}

	// Gives the free pages of the heap back to the system (glibc: malloc_trim; macOS: the zones'
	// pressure relief). The rows of an elimination grow and shrink every round and the freed space is
	// scattered over the per-thread arenas, so without this the resident set keeps growing after the
	// matrix has passed its largest size (exec_policy::trim_heap).
	inline void trim_heap(const exec_policy& pol) {
#if !defined(USE_MIMALLOC) && (defined(__GLIBC__) || defined(__APPLE__))
		if (pol.trim_heap) {
#if defined(__GLIBC__)
			malloc_trim(0);
#else
			malloc_zone_pressure_relief(nullptr, 0);
#endif
		}
#else
		(void)pol;
#endif
	}

#ifdef SPARSERREF_USE_CUDA
	// the GPU tunables of a policy, for the GPU module
	inline void gpu_configure(const gpu_policy& gp) {
		gpu::options o;
		o.transfer_piece = gp.transfer_piece;
		o.rr_pinned_bytes = gp.rr_pinned_mb << 20;
		o.large_u = gp.call_large_u;
		gpu::set_options(o);
	}
	// whether a GPU path may start for work of nnz entries: the policy enables it, the work is large
	// enough (CUDA is not even initialised below min_nnz: small systems are faster on the CPU, except
	// once CUDA runs anyway) and a device is there
	inline bool gpu_go(const gpu_policy& gp, size_t nnz) {
		if (!gp.enabled || (!gpu::initialized() && nnz < gp.min_nnz) || !gpu::available())
			return false;
		gpu_configure(gp);
		return true;
	}
#endif

#ifdef SPARSERREF_USE_CUDA
	// Run one parallel Schur loop (rows updated by the ordered pivot list) on the GPU. Returns false
	// when the GPU is not used (disabled, not Z/p, too small, does not fit); the caller then runs the
	// CPU loop for all rows. On true every row is updated except those listed in redo, which the
	// caller must update on the CPU (a CUDA failure after some rows were written back).
	template <typename T, typename index_t, typename RowVec>
	bool schur_gpu(sparse_mat<T, index_t>& mat, const RowVec& rows,
		const std::vector<pivot_t<index_t>>& pivots, const field_t& F, thread_pool& pool,
		const gpu_policy& gp, std::vector<size_t>& redo) {
		if constexpr (!std::is_same_v<T, ulong>) {
			return false;
		}
		else {
			const size_t min_rows = gp.call_min_rows, min_piv = gp.call_min_pivots;
			const size_t max_cap = gp.call_max_arena, chunk = gp.call_chunk;
			redo.clear();
			if (rows.size() < min_rows || pivots.size() < min_piv || rows.size() > UINT32_MAX || F.get_prime() >= (1ULL << 63))
				return false;
			{
				size_t nnz = 0;
				for (size_t i = 0; i < rows.size(); i++)
					nnz += mat[rows[i]].nnz();
				if (!gpu_go(gp, nnz))
					return false;
			}
			SPR_T0(spr_g);
			const size_t P = pivots.size(), R = rows.size();
			// U: the columns of the pivot rows, renumbered in increasing order
			std::vector<uint8_t> inU(mat.ncol, 0);
			pool.detach_loop<size_t>(0, P, [&](size_t j) {
				for (auto col : mat[pivots[j].r].index_span())
					std::atomic_ref<uint8_t>(inU[col]).store(1, std::memory_order_relaxed);
				});
			pool.wait();
			std::vector<int32_t> cid_of_col(mat.ncol, -1);
			std::vector<uint32_t> col_of_cid;
			for (size_t col = 0; col < mat.ncol; col++)
				if (inU[col]) {
					cid_of_col[col] = (int32_t)col_of_cid.size();
					col_of_cid.push_back((uint32_t)col);
				}
			const uint32_t nU = (uint32_t)col_of_cid.size();
			if (!gpu::fits(nU, (uint32_t)P))
				return false;
			std::vector<int32_t> posU(nU, -1);
			std::vector<uint32_t> piv_cid(P);
			for (size_t j = 0; j < P; j++) {
				piv_cid[j] = (uint32_t)cid_of_col[pivots[j].c];
				posU[piv_cid[j]] = (int32_t)j;
			}
			// a row changes only if it holds a pivot column
			std::vector<uint8_t> take(R);
			pool.detach_loop<size_t>(0, R, [&](size_t i) {
				bool hp = false;
				for (auto col : mat[rows[i]].index_span()) {
					const int32_t cid = cid_of_col[col];
					if (cid >= 0 && posU[cid] >= 0) {
						hp = true;
						break;
					}
				}
				take[i] = hp;
				});
			pool.wait();
			std::vector<uint32_t> sel;
			sel.reserve(R);
			for (size_t i = 0; i < R; i++)
				if (take[i])
					sel.push_back((uint32_t)i);
			SPR_DO(SparseRREF::prof::d().gpu_rows += sel.size());
			SPR_DO(SparseRREF::prof::d().gpu_skip += R - sel.size());
			if (sel.empty()) {
				SPR_DO(SparseRREF::prof::d().gpu_pack += SparseRREF::prof::since(spr_g));
				return true;
			}
			// pivot rows over U (page-locked: slots 0, 1)
			std::vector<uint64_t> pv_off(P + 1, 0);
			for (size_t j = 0; j < P; j++)
				pv_off[j + 1] = pv_off[j] + mat[pivots[j].r].nnz();
			auto* pv_cid = (uint32_t*)gpu::host_buffer(0, pv_off[P] * 4 + 8);
			auto* pv_val = (uint64_t*)gpu::host_buffer(1, pv_off[P] * 8 + 8);
			if (!pv_cid || !pv_val)
				return false;
			pool.detach_loop<size_t>(0, P, [&](size_t j) {
				auto& v = mat[pivots[j].r];
				size_t o = pv_off[j];
				for (size_t k = 0; k < v.nnz(); k++) {
					pv_cid[o + k] = (uint32_t)cid_of_col[v(k)];
					pv_val[o + k] = v[k];
				}
				});
			pool.wait();
			gpu::schur_common cm;
			cm.p = F.get_prime(); cm.nU = nU; cm.P = (uint32_t)P;
			cm.pv_off = pv_off.data(); cm.pv_cid = pv_cid; cm.pv_val = pv_val;
			cm.piv_cid = piv_cid.data(); cm.posU = posU.data();
			cm.ncol = (uint32_t)mat.ncol; cm.cid_of_col = cid_of_col.data(); cm.col_of_cid = col_of_cid.data();
			const bool begun = gpu::schur_begin(cm);
			SPR_DO(SparseRREF::prof::d().gpu_pack += SparseRREF::prof::since(spr_g));
			if (!begun)
				return false;

			// stage the whole rows list[0..S) (indices into rows) in a slot and submit them
			auto submit = [&](int slot, const uint32_t* list, size_t S, uint64_t cap) -> bool {
				SPR_T0(spr_p);
				auto* in_off = (uint64_t*)gpu::schur_stage(slot, 0, (S + 1) * 8);
				if (!in_off)
					return false;
				in_off[0] = 0;
				for (size_t s = 0; s < S; s++)
					in_off[s + 1] = in_off[s] + mat[rows[list[s]]].nnz();
				auto* in_col = (uint32_t*)gpu::schur_stage(slot, 1, in_off[S] * 4 + 8);
				auto* in_val = (uint64_t*)gpu::schur_stage(slot, 2, in_off[S] * 8 + 8);
				if (!in_col || !in_val)
					return false;
				pool.detach_loop<size_t>(0, S, [&](size_t s) {
					auto& v = mat[rows[list[s]]];
					const size_t a = in_off[s], n = v.nnz();
					if constexpr (sizeof(index_t) == 4)
						std::memcpy(in_col + a, v.indices, n * 4);
					else
						for (size_t k = 0; k < n; k++)
							in_col[a + k] = (uint32_t)v.indices[k];
					std::memcpy(in_val + a, v.entries, n * 8);
					});
				pool.wait();
				if (cap == 0) {
					for (size_t s = 0; s < S; s++) {
						const uint64_t n = in_off[s + 1] - in_off[s];
						cap += std::min<uint64_t>(n + nU, 2 * n + 64);
					}
					cap = std::min<uint64_t>(cap, max_cap);
				}
				SPR_DO(SparseRREF::prof::d().gpu_pack += SparseRREF::prof::since(spr_p));
				SPR_DO(SparseRREF::prof::d().gpu_in += in_off[S]);
				return gpu::schur_submit(slot, (uint32_t)S, cap);
			};
			// wait for a slot and copy its rows back; rows whose result did not fit the arena are
			// appended to over with their exact output sizes
			auto collect = [&](int slot, const uint32_t* list, size_t S, std::vector<uint32_t>& over,
				std::vector<uint64_t>& over_cnt) -> bool {
				SPR_T0(spr_w);
				gpu::schur_out res;
				if (!gpu::schur_wait(slot, res))
					return false;
				SPR_DO(SparseRREF::prof::d().gpu_wait += SparseRREF::prof::since(spr_w));
				SPR_T0(spr_u);
				for (size_t s = 0; s < S; s++)
					if (res.status[s]) {
						over.push_back(list[s]);
						over_cnt.push_back(res.cnt[s]);
					}
				pool.detach_loop<size_t>(0, S, [&](size_t s) {
					if (res.status[s])
						return;
					auto& v = mat[rows[list[s]]];
					const size_t n = res.cnt[s], b = res.base[s];
					if (v.alloc() < n)
						v.reserve(n, false);
					v.resize(n);
					if constexpr (sizeof(index_t) == 4)
						std::memcpy(v.indices, res.col + b, n * 4);
					else
						for (size_t k = 0; k < n; k++)
							v.indices[k] = (index_t)res.col[b + k];
					std::memcpy(v.entries, res.val + b, n * 8);
					});
				pool.wait();
				SPR_DO(SparseRREF::prof::d().gpu_unpack += SparseRREF::prof::since(spr_u));
				SPR_DO(SparseRREF::prof::d().gpu_h2d += res.h2d);
				SPR_DO(SparseRREF::prof::d().gpu_kernel += res.kernel);
				SPR_DO(SparseRREF::prof::d().gpu_d2h += res.d2h);
				SPR_DO(SparseRREF::prof::d().gpu_calls++);
				return true;
			};

			// parts of about chunk entries, through the two slots: part c + 1 is on the GPU while part c
			// is copied back and part c + 2 is staged
			std::vector<size_t> pb{ 0 };
			{
				size_t acc = 0;
				for (size_t s = 0; s < sel.size(); s++) {
					acc += mat[rows[sel[s]]].nnz();
					if (acc >= chunk && s + 1 < sel.size()) {
						pb.push_back(s + 1);
						acc = 0;
					}
				}
				pb.push_back(sel.size());
			}
			const size_t nparts = pb.size() - 1;
			auto part = [&](size_t c) { return std::pair<const uint32_t*, size_t>(sel.data() + pb[c], pb[c + 1] - pb[c]); };
			std::vector<uint32_t> over;
			std::vector<uint64_t> over_cnt;
			if (!submit(0, part(0).first, part(0).second, 0))
				return false; // nothing written back: the caller runs the CPU loop
			if (nparts > 1 && !submit(1, part(1).first, part(1).second, 0)) {
				// part 0 is in flight, nothing written back yet: wait for it and give up
				gpu::schur_out drop;
				gpu::schur_wait(0, drop);
				return false;
			}
			for (size_t c = 0; c < nparts; c++) {
				const int slot = (int)(c & 1);
				const bool merged = collect(slot, part(c).first, part(c).second, over, over_cnt);
				bool ok = merged;
				if (ok && c + 2 < nparts)
					ok = submit(slot, part(c + 2).first, part(c + 2).second, 0);
				if (!ok) {
					// CUDA failure: the rows of the parts not merged yet go to the CPU (the rows that
					// overflowed in merged parts stay in over and are handled below)
					for (size_t s = pb[merged ? c + 1 : c]; s < sel.size(); s++)
						redo.push_back(rows[sel[s]]);
					if (c + 1 < nparts) {
						gpu::schur_out drop;
						gpu::schur_wait(1 - slot, drop);
					}
					break;
				}
			}
			// rows that did not fit: again with the exact size, in arena-sized batches through slot 0
			size_t pos = 0;
			while (pos < over.size()) {
				std::vector<uint32_t> batch, o2;
				std::vector<uint64_t> c2;
				uint64_t need = 0;
				while (pos < over.size() && (batch.empty() || need + over_cnt[pos] <= max_cap)) {
					need += over_cnt[pos];
					batch.push_back(over[pos++]);
				}
				SPR_DO(SparseRREF::prof::d().gpu_redo += batch.size());
				const bool ok = submit(0, batch.data(), batch.size(), need) && collect(0, batch.data(), batch.size(), o2, c2);
				if (!ok || !o2.empty()) {
					// CUDA failure: the remaining rows go to the CPU
					for (auto i : (ok ? o2 : batch))
						redo.push_back(rows[i]);
					for (; pos < over.size(); pos++)
						redo.push_back(rows[over[pos]]);
				}
			}
			return true;
		}
	}
#endif

	// runs the Schur loop of rows on the GPU when it can; returns true when all rows are done
	template <typename T, typename index_t, typename RowVec>
	bool schur_try_gpu(sparse_mat<T, index_t>& mat, const RowVec& rows,
		const std::vector<pivot_t<index_t>>& pivots, const field_t& F, thread_pool& pool,
		schur_helper_buffer<T, index_t>& g_helper, const exec_policy& pol) {
#ifdef SPARSERREF_USE_CUDA
		std::vector<size_t> redo;
		if (!schur_gpu(mat, rows, pivots, F, pool, pol.gpu, redo))
			return false;
		if constexpr (!std::is_same_v<T, bool>) {
			schur_helper<T, index_t> helper(g_helper, 0); // no pool task is running here
			for (auto r : redo)
				schur_complete_indexed<T, index_t>(mat, r, pivots, F, helper);
		}
		return true;
#else
		(void)mat; (void)rows; (void)pivots; (void)F; (void)pool; (void)g_helper; (void)pol;
		return false;
#endif
	}

	// The strategy for a matrix of this density: once a fair share of the pivot columns is present in a
	// row the set test is answered by the cache anyway, and below that the histogram of the buffered
	// strategy rejects most pivot columns without touching the set.
	template <typename T, typename index_t>
	auto schur_strategy(const double density) {
		using func_t = void (*)(sparse_mat<T, index_t>&, size_t, const std::vector<pivot_t<index_t>>&,
			const field_t&, schur_helper<T, index_t>&);
		if constexpr (!std::is_same_v<T, bool>)
			return func_t(&schur_complete_indexed<T, index_t>); // scans like schur_complete if not indexed
		else {
			if (100 * density > 0.01)
				return func_t(&schur_complete<T, index_t>);
			return func_t(&schur_complete_buffer<T, index_t, 10>);
		}
	}

	// Progress bookkeeping of the triangular back substitution, shared by all
	// recursion levels: every report covers the work completed since the previous
	// one, so a report issued at a level entry shows the real average speed of
	// the levels that came before instead of 0.
	struct triangular_progress_t {
		std::chrono::system_clock::time_point last_time; // time of the previous report
		double last_count;                               // completed pivots at the previous report
		bool reported;                                   // whether any line was already printed
	};

	// Back substitution in batches of n_split pivots, from the last pivot backwards: each batch
	// first eliminates its pivot columns from the rows above it (Schur updates, in parallel),
	// then reduces itself with triangular_solver. This used to recurse once per batch, copying
	// the remaining pivot list and keeping every level's row set alive until the innermost
	// level returned (O(rank^2 / n_split) bytes: ~70 MiB at rank 1.4e5); it is now a loop over a
	// shrinking span that reuses the same containers, with the rows of a batch listed through a
	// mark array instead of an unordered_set. Same batches, same order, same Schur updates:
	// the result is identical.
	template <typename T, typename index_t>
	void triangular_solver_2_rec(sparse_mat<T, index_t>& mat,
		const pivot_transpose_t<index_t>& tranmat,
		const std::vector<pivot_t<index_t>>& all_pivots,
		const field_t& F, rref_option_t opt,
		schur_helper_buffer<T, index_t>& g_helper, size_t n_split, size_t rank, size_t& process,
		triangular_progress_t& tprogress) {

		const size_t printstep = opt->print_step > 0 ? (size_t)opt->print_step : 1;
		bool verbose = opt->verbose;
		auto& pool = opt->pool;
		std::span<const pivot_t<index_t>> pivots(all_pivots);
		std::vector<char> mark(mat.nrow, 0); // 1: row already listed, 2: a pivot row of this batch
		std::vector<pivot_t<index_t>> sub_pivots;
		std::vector<index_t> leftrows;
		sub_pivots.reserve(n_split);

		while (pivots.size() >= n_split) {
			if (opt->abort)
				return;
			opt->verbose = false;
			sub_pivots.assign(pivots.end() - n_split, pivots.end());
			leftrows.clear();
			for (auto [r, c] : sub_pivots)
				mark[r] = 2;
			for (auto [r, c] : sub_pivots) {
				for (auto rr : tranmat.rows_of(c)) {
					if (mark[rr] == 0) {
						mark[rr] = 1;
						leftrows.push_back(rr);
					}
				}
			}
			for (auto rr : leftrows)
				mark[rr] = 0;
			for (auto [r, c] : sub_pivots)
				mark[r] = 0;

			// for printing (the count at this barrier; the tasks below resize rows)
			size_t now_nnz = mat.nnz();
			int bitlen_nnz = num_digits(now_nnz) + 1;
			int bitlen_nrow = num_digits(rank);

			double density = (double)now_nnz / (mat.nrow * mat.ncol);
			auto schur_complete_func = schur_strategy<T, index_t>(density);

			SPR_T0(spr_bs);
			SPR_DO(SparseRREF::prof::d().bs_batches++);
			SPR_DO(SparseRREF::prof::d().bs_schur_rows += leftrows.size());
			std::atomic<size_t> cc = 0;
			schur_index_guard<T, index_t> spr_index(g_helper, sub_pivots);
			const bool spr_gpu = schur_try_gpu(mat, leftrows, sub_pivots, F, pool, g_helper, opt->policy);
			if (!spr_gpu)
			pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
				auto id = SparseRREF::thread_id();
				schur_helper<T, index_t> helper(g_helper, id);
				for (size_t i = s; i < e; i++) {
					if (opt->abort)
						return;
					schur_complete_func(mat, leftrows[i], sub_pivots, F, helper);
					cc++;
				}
				}, ((n_split < 20 * pool.get_thread_count()) ? 0 : leftrows.size() / 10));
			if (spr_gpu)
				cc = leftrows.size();

			if (verbose) {
				while (cc.load(std::memory_order_relaxed) < leftrows.size()) {
					if (opt->abort)
						break; // the wait below drains the pool
					double status = 1.0 * sub_pivots.size() * cc.load(std::memory_order_relaxed) / leftrows.size();
					double done = (double)process + status;
					if (done - tprogress.last_count <= (double)printstep) {
						if (pool.wait_for(progress_poll_interval))
							break; // pool idle, no further progress can be reported
						continue;
					}
					auto now = SparseRREF::clocknow();
					progress(opt, "Row")
						.add("%*zu/%zu", bitlen_nrow, (size_t)done, rank)
						.add("  nnz: %*zu", bitlen_nnz, now_nnz)
						.add("  density: %8.6g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
						.add("  speed: %6.6g row/s", rate_per_second(done - tprogress.last_count, SparseRREF::usedtime(tprogress.last_time, now)))
						.add("    ");
					tprogress.last_time = now;
					tprogress.last_count = done;
					tprogress.reported = true;
				}
			}

			pool.wait();
			SPR_ADD(bs_schur, spr_bs);
			if (opt->abort) {
				opt->verbose = verbose;
				return;
			}

			SPR_T0(spr_tri);
			triangular_solver(mat, tranmat, sub_pivots, F, opt, -1);
			SPR_ADD(bs_tri, spr_tri);
			opt->verbose = verbose;
			process += sub_pivots.size();
			pivots = pivots.first(pivots.size() - n_split);
		}

		// the remainder (fewer than n_split pivots) performs the elimination itself, so it is
		// the only one that can report it; keep the caller's verbosity
		if (opt->abort)
			return;
		sub_pivots.assign(pivots.begin(), pivots.end());
		SPR_T0(spr_tri);
		triangular_solver(mat, tranmat, sub_pivots, F, opt, -1);
		SPR_ADD(bs_tri, spr_tri);
		process += sub_pivots.size();
	}

	// The back substitution takes the pivots from the last one backwards and needs pivot k's row to
	// hold no pivot column of an earlier pivot (the pivot block upper triangular in this order): then
	// eliminating a pivot column never brings in a pivot column still to be processed. The forward
	// elimination yields such an order. For any other order whose pivot block is triangular after a
	// permutation, this finds one (Kahn's algorithm, earliest given position first, so a valid order
	// is kept as it is). The RREF for a given pivot set is unique, so the order does not change the
	// result. row_cols(r, visit) visits the columns of row r. Returns false (pivots unchanged) if the
	// pivot block is not triangular in any order.
	template <typename index_t, typename RowCols>
	bool backsub_order(std::vector<pivot_t<index_t>>& piv, size_t ncol, RowCols&& row_cols, thread_pool* pool = nullptr) {
		const size_t P = piv.size();
		if (P < 2)
			return true;
		std::vector<int64_t> pos(ncol, -1);
		for (size_t k = 0; k < P; k++)
			pos[piv[k].c] = (int64_t)k;
		std::atomic<bool> valid = true;
		auto check = [&](size_t k0, size_t k1) {
			for (size_t k = k0; k < k1 && valid.load(std::memory_order_relaxed); k++)
				row_cols(piv[k].r, [&](size_t c) {
					const int64_t j = pos[c];
					if (j >= 0 && j < (int64_t)k)
						valid.store(false, std::memory_order_relaxed);
					});
			};
		if (pool != nullptr && P > 4096) {
			pool->detach_blocks<size_t>(0, P, check);
			pool->wait();
		}
		else
			check(0, P);
		if (valid)
			return true;
		// edges k -> j (k before j) for each pivot column c_j in row k
		std::vector<size_t> indeg(P, 0), eoff(P + 1, 0);
		for (size_t k = 0; k < P; k++)
			row_cols(piv[k].r, [&](size_t c) {
				const int64_t j = pos[c];
				if (j >= 0 && j != (int64_t)k) {
					indeg[j]++;
					eoff[k + 1]++;
				}
				});
		for (size_t k = 0; k < P; k++)
			eoff[k + 1] += eoff[k];
		std::vector<size_t> adj(eoff[P]), cur(eoff.begin(), eoff.end() - 1);
		for (size_t k = 0; k < P; k++)
			row_cols(piv[k].r, [&](size_t c) {
				const int64_t j = pos[c];
				if (j >= 0 && j != (int64_t)k)
					adj[cur[k]++] = (size_t)j;
				});
		std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
		for (size_t k = 0; k < P; k++)
			if (indeg[k] == 0)
				ready.push(k);
		std::vector<pivot_t<index_t>> out;
		out.reserve(P);
		while (!ready.empty()) {
			const size_t k = ready.top();
			ready.pop();
			out.push_back(piv[k]);
			for (size_t e = eoff[k]; e < eoff[k + 1]; e++)
				if (--indeg[adj[e]] == 0)
					ready.push(adj[e]);
		}
		if (out.size() != P)
			return false;
		piv = std::move(out);
		return true;
	}

	// A batch of a level that cannot go in one pass (its pivot block is not triangular in any order,
	// or some pivot rows do not hold their pivot column yet: merged levels, where an entry appears only
	// once other pivots were applied). A batch holds only pivots whose row holds its column, and no row
	// of a batch holds the column of an earlier pivot of the batch, so it goes in one pass; applied to
	// the rest of the level it removes the batch's columns from their rows, after which the rest is
	// examined again. Two ways to pick it, both following the given order (or its reverse):
	//   greedy = false: the longest prefix that qualifies -- elimination in the given order, batched;
	//   greedy = true: the pivots Kahn's algorithm can order, then a set of the others whose rows hold
	//     none of each other's columns (larger batches, but the order differs from the given one, and a
	//     pivot entry may cancel on the way).
	// The batch is removed from rest (which keeps its order); it is empty when the next pivot cannot be
	// applied (no entry).
	template <typename index_t, typename RowCols>
	void level_batch(std::vector<pivot_t<index_t>>& rest, [[maybe_unused]] size_t ncol, RowCols&& row_cols,
		std::vector<pivot_t<index_t>>& batch, const bool reverse = false, const bool greedy = false) {
		const size_t P = rest.size();
		auto at = [&](size_t i) { return reverse ? P - 1 - i : i; }; // i-th in priority order
		batch.clear();
		std::unordered_map<size_t, size_t> pos; // pivot column -> position in rest
		pos.reserve(2 * P);
		for (size_t k = 0; k < P; k++)
			pos[rest[k].c] = k;
		std::vector<char> has(P, 0);
		std::vector<std::vector<size_t>> out_edges(P); // k -> j: the row of k holds column j
		for (size_t k = 0; k < P; k++)
			row_cols(rest[k].r, [&](size_t c) {
				auto it = pos.find(c);
				if (it == pos.end())
					return;
				if (it->second == k)
					has[k] = 1;
				else
					out_edges[k].push_back(it->second);
				});
		std::vector<char> used(P, 0);
		if (!greedy) {
			for (size_t i = 0; i < P; i++) {
				const size_t k = at(i);
				if (!has[k])
					break;
				bool clash = false;
				for (auto j : out_edges[k])
					if (used[j]) {
						clash = true;
						break;
					}
				if (clash)
					break;
				used[k] = 1;
				batch.push_back(rest[k]);
			}
		}
		else {
			std::vector<size_t> indeg(P, 0);
			for (size_t k = 0; k < P; k++)
				if (has[k])
					for (auto j : out_edges[k])
						indeg[j]++;
			std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready; // priority ranks
			for (size_t k = 0; k < P; k++)
				if (has[k] && indeg[k] == 0)
					ready.push(at(k));
			while (!ready.empty()) {
				const size_t k = at(ready.top());
				ready.pop();
				used[k] = 1;
				batch.push_back(rest[k]);
				for (auto j : out_edges[k])
					if (--indeg[j] == 0 && has[j])
						ready.push(at(j));
			}
			std::vector<char> held(P, 0), in_set(P, 0); // held[j]: the row of a set member holds column j
			for (size_t i = 0; i < P; i++) {
				const size_t k = at(i);
				if (used[k] || !has[k] || held[k])
					continue;
				bool clash = false;
				for (auto j : out_edges[k])
					if (in_set[j]) {
						clash = true;
						break;
					}
				if (clash)
					continue;
				in_set[k] = 1;
				used[k] = 1;
				batch.push_back(rest[k]);
				for (auto j : out_edges[k])
					held[j] = 1;
			}
		}
		std::vector<pivot_t<index_t>> left;
		for (size_t k = 0; k < P; k++)
			if (!used[k])
				left.push_back(rest[k]);
		rest = std::move(left);
	}

	// the pivots of levels 1.. (level 0 holds rows with a single entry, whose columns the
	// elimination already removed everywhere) in back substitution order, on mat's rows
	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> backsub_pivots(const sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots, thread_pool* pool = nullptr) {
		std::vector<pivot_t<index_t>> n_pivots;
		for (size_t i = 1; i < pivots.size(); i++)
			n_pivots.insert(n_pivots.end(), pivots[i].begin(), pivots[i].end());
		backsub_order(n_pivots, mat.ncol, [&](size_t r, auto&& visit) {
			const auto& v = mat[r];
			for (size_t j = 0; j < v.nnz(); j++)
				visit((size_t)v(j));
			}, pool);
		return n_pivots;
	}

	template <typename T, typename index_t>
	void triangular_solver_2(sparse_mat<T, index_t>& mat, std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt) {

		auto& pool = opt->pool;
		// prepare the tmp array
		auto nthreads = pool.get_thread_count();
		schur_helper_buffer<T, index_t> g_helper(nthreads, mat.ncol);

		if (opt->abort)
			return;

		// built once for all the batches below
		SPR_T0(spr_tr);
		const auto tranmat = build_pivot_transpose(mat, pivots, pool);
		SPR_ADD(bs_transpose, spr_tr);

		size_t process = 0;
		// TODO: better split strategy
		size_t n_split = std::max(pivots.size() / 128ULL, 1ULL << 10);
		size_t rank = pivots.size();
		triangular_progress_t tprogress{ SparseRREF::clocknow(), 0.0, false };
		triangular_solver_2_rec(mat, tranmat, pivots, F, opt, g_helper, n_split, rank, process, tprogress);

		// A level can only report while it is still polling, so the tail of the phase
		// (everything done after the last report) would stay invisible without this.
		if (opt->verbose && pivots.size() >= n_split && rank > tprogress.last_count) {
			auto now = SparseRREF::clocknow();
			size_t now_nnz = mat.nnz();
			int bitlen_nnz = num_digits(now_nnz) + 1;
			progress(opt, "Row")
				.add("%*zu/%zu", num_digits(rank), rank, rank)
				.add("  nnz: %*zu", bitlen_nnz, now_nnz)
				.add("  density: %8.6g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
				.add("  speed: %6.6g row/s",
					rate_per_second(rank - tprogress.last_count, SparseRREF::usedtime(tprogress.last_time, now)))
				.add("    ");
			tprogress.reported = true;
		}

		if (opt->verbose && tprogress.reported)
			progress_end(opt);
	}

	template <typename T, typename index_t>
	void triangular_solver_2(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt) {

		// the first pivot is the row with only one nonzero value, so there is no need to do the elimination
		auto n_pivots = backsub_pivots(mat, pivots, &opt->pool);

		triangular_solver_2(mat, n_pivots, F, opt);
	}

	// Receives the rows of a replay from the GPU in place of the matrix (the reconstruction stores
	// residues straight from them): sink(rows, n, off, col, val) gets rows[0..n) with off[0..n]
	// relative to col / val; returning false aborts the delivery.
	using replay_sink_t = std::function<bool(const uint32_t* rows, uint32_t n, const uint64_t* off,
		const uint32_t* col, const uint64_t* val)>;

	// rows whose values live on the GPU keep their columns only (entries freed, alloc counts indices)
	template <typename index_t>
	void vec_set_structure(sparse_vec<ulong, index_t>& v, size_t n, const uint32_t* col) {
		if (v.entries != nullptr) {
			s_free(v.entries);
			v.entries = nullptr;
		}
		if (n == 0) {
			v.clear();
			return;
		}
		if (v._alloc < n) {
			const size_t cap = std::max(n, v._alloc + v._alloc / 2); // rows grow every round
			s_free(v.indices);
			v.indices = s_malloc<index_t>(cap);
			v._alloc = cap;
		}
		if constexpr (sizeof(index_t) == 4)
			std::memcpy(v.indices, col, n * 4);
		else
			for (size_t k = 0; k < n; k++)
				v.indices[k] = (index_t)col[k];
		v._nnz = n;
	}
	// a row of exactly n entries (columns and values)
	template <typename index_t>
	void vec_set_full(sparse_vec<ulong, index_t>& v, size_t n, const uint32_t* col, const uint64_t* val) {
		if (n == 0) {
			v.clear();
			return;
		}
		if (v._alloc != n || v.entries == nullptr) {
			v.clear();
			v.indices = s_malloc<index_t>(n);
			v.entries = s_malloc<ulong>(n);
			v._alloc = n;
		}
		if constexpr (sizeof(index_t) == 4)
			std::memcpy(v.indices, col, n * 4);
		else
			for (size_t k = 0; k < n; k++)
				v.indices[k] = (index_t)col[k];
		std::memcpy(v.entries, val, n * 8);
		v._nnz = n;
	}

#ifdef SPARSERREF_USE_CUDA

	// The elimination of sparse_mat_direct_rref with the matrix kept on the GPU: the rows of
	// pivots[1..] (rows, in the order of the CPU's leftrows: later levels first) go up once, each
	// level is rescaled and applied to the rows of the later levels there, then the back substitution,
	// and the rows come back once. Returns 0 when the GPU is not used or fails (mat unchanged), 1 when
	// mat holds the result (backsub_done tells whether the back substitution was done there; if it
	// failed, mat holds the eliminated rows and the CPU completes it), 2 when the complete result went
	// to the sink instead (mat's rows of pivots[1..] unchanged).
	template <typename T, typename index_t>
	int direct_rref_gpu(sparse_mat<T, index_t>& mat, const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const std::vector<size_t>& rows_in, const field_t& F, rref_option_t opt, bool& backsub_done,
		const replay_sink_t* sink) {
		backsub_done = false;
		if constexpr (!std::is_same_v<T, ulong>) {
			return 0;
		}
		else {
			const auto& gp = opt->policy.gpu;
			const size_t R = rows_in.size();
			if (!gp.replay || R < gp.call_min_rows || mat.nrow > UINT32_MAX || mat.ncol > UINT32_MAX || F.get_prime() >= (1ULL << 63))
				return 0;
			{
				// the input rows are sparse; the eliminated ones are what costs: judge by the last
				// level's rows, which the replay fills like the first prime did
				size_t nnz = 0;
				for (auto r : rows_in)
					nnz += mat[r].nnz();
				// once CUDA runs (the first prime's elimination used it) its start-up cost is paid
				if (!gpu_go(gp, nnz))
					return 0;
			}
			SPR_T0(spr_rp);
			auto& pool = opt->pool;
			std::vector<uint32_t> rows(R);
			std::vector<uint64_t> off(R + 1, 0);
			for (size_t i = 0; i < R; i++) {
				rows[i] = (uint32_t)rows_in[i];
				off[i + 1] = off[i] + mat[rows_in[i]].nnz();
			}
			bool ok = !gpu::fail_inject("replay") && gpu::res_begin(F.get_prime(), (uint32_t)mat.nrow, (uint32_t)mat.ncol, (uint32_t)R, rows.data(),
				off.data(), [&](uint32_t r0, uint32_t r1, uint32_t* col, uint64_t* val) {
					pool.detach_loop<size_t>(r0, r1, [&](size_t i) {
						auto& v = mat[rows_in[i]];
						const size_t a = off[i] - off[r0], n = v.nnz();
						if constexpr (sizeof(index_t) == 4)
							std::memcpy(col + a, v.indices, n * 4);
						else
							for (size_t k = 0; k < n; k++)
								col[a + k] = (uint32_t)v.indices[k];
						std::memcpy(val + a, v.entries, n * 8);
						});
					pool.wait();
				});
			size_t remaining = R; // rows[0, remaining) are still to be updated; a level's rows are the last of them
			std::vector<uint32_t> prow, pcol;
			std::vector<char> in_batch;
			for (size_t i = 1; ok && i < pivots.size(); i++) {
				if (pivots[i].empty())
					continue;
				// a level goes in one pass, in an order where no pivot row holds the column of an earlier
				// pivot (order_level): checked there, reordered on its rows' columns when needed, and if
				// it is not triangular in any order, applied in batches (level_batch)
				std::vector<pivot_t<index_t>> rest = pivots[i], batch;
				while (ok && !rest.empty()) {
					const size_t P = rest.size();
					prow.resize(P);
					pcol.resize(P);
					for (size_t j = 0; j < P; j++) {
						prow[j] = (uint32_t)rest[j].r;
						pcol[j] = (uint32_t)rest[j].c;
					}
					const int ord = gpu::res_level_order((uint32_t)P, prow.data(), pcol.data());
					if (ord < 0) {
						ok = false;
						break;
					}
					if (ord == 1) {
						batch = std::move(rest);
						rest.clear();
					}
					else {
						std::vector<uint64_t> loff(P + 1, 0);
						std::vector<uint32_t> lcol, lslot(mat.nrow, UINT32_MAX);
						for (size_t j = 0; j < P; j++)
							lslot[prow[j]] = (uint32_t)j;
						ok = gpu::res_download_pieces((uint32_t)P, prow.data(), false,
							[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t*) {
								for (uint32_t t = r0; t < r1; t++)
									loff[t + 1] = loff[t] + (o[t - r0 + 1] - o[t - r0]);
								lcol.insert(lcol.end(), col, col + o[r1 - r0]);
								return true;
							});
						if (!ok)
							break;
						auto row_cols = [&](size_t r, auto&& visit) {
							const uint32_t t = lslot[r];
							for (uint64_t x = loff[t]; x < loff[t + 1]; x++)
								visit((size_t)lcol[x]);
							};
						bool entries = true; // every pivot row holds its column
						for (size_t j = 0; j < P && entries; j++) {
							bool f = false;
							row_cols(prow[j], [&](size_t c) { f = f || c == pcol[j]; });
							entries = f;
						}
						batch = rest;
						if (entries && backsub_order(batch, mat.ncol, row_cols, &pool)) {
							rest.clear();
							SPR_DO(SparseRREF::prof::d().gpu_level_reorders++);
						}
						else {
							level_batch(rest, mat.ncol, row_cols, batch);
							SPR_DO(SparseRREF::prof::d().level_batches++);
							if (batch.empty()) { // not pivots of this matrix: the CPU reports it
								ok = false;
								break;
							}
						}
					}
					// the batch's rows go last among the level's rows still to be updated
					if (!rest.empty()) {
						const size_t seg = batch.size() + rest.size();
						in_batch.resize(mat.nrow, 0);
						for (auto [r, c] : batch)
							in_batch[r] = 1;
						std::stable_partition(rows.begin() + (remaining - seg), rows.begin() + remaining,
							[&](uint32_t r) { return !in_batch[r]; });
						for (auto [r, c] : batch)
							in_batch[r] = 0;
					}
					const size_t B = batch.size();
					prow.resize(B);
					pcol.resize(B);
					for (size_t j = 0; j < B; j++) {
						prow[j] = (uint32_t)batch[j].r;
						pcol[j] = (uint32_t)batch[j].c;
					}
					ok = gpu::res_rescale((uint32_t)B, prow.data(), pcol.data());
					remaining -= B;
					if (ok && remaining > 0)
						ok = gpu::res_schur((uint32_t)B, prow.data(), pcol.data(), (uint32_t)remaining, rows.data());
					if (opt->abort)
						ok = false;
				}
				if (remaining == 0)
					break;
			}
			// the back substitution too, while the rows are there
			bool bs_failed = false;
			if (ok && opt->is_back_sub && gp.backsub && !opt->abort) {
				// the order is checked on the eliminated rows, whose columns come up from the GPU
				std::vector<uint64_t> soff(R + 1, 0);
				std::vector<uint32_t> scol;
				std::vector<uint32_t> slot_of(mat.nrow, UINT32_MAX);
				for (size_t i = 0; i < R; i++)
					slot_of[rows[i]] = (uint32_t)i; // rows may have been reordered within levels
				ok = gpu::res_download_pieces((uint32_t)R, rows.data(), false,
					[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t*) {
						for (uint32_t i = r0; i < r1; i++)
							soff[i + 1] = soff[i] + (o[i - r0 + 1] - o[i - r0]);
						scol.insert(scol.end(), col, col + o[r1 - r0]);
						return true;
					});
				std::vector<pivot_t<index_t>> bp;
				for (size_t i = 1; i < pivots.size(); i++)
					bp.insert(bp.end(), pivots[i].begin(), pivots[i].end());
				if (ok)
					backsub_order(bp, mat.ncol, [&](size_t r, auto&& visit) {
						const uint32_t i = slot_of[r];
						if (i == UINT32_MAX)
							return;
						for (uint64_t t = soff[i]; t < soff[i + 1]; t++)
							visit((size_t)scol[t]);
						}, &pool);
				std::vector<uint32_t> bprow(bp.size()), bpcol(bp.size());
				for (size_t j = 0; j < bp.size(); j++) {
					bprow[j] = (uint32_t)bp[j].r;
					bpcol[j] = (uint32_t)bp[j].c;
				}
				const size_t n_split = gp.backsub_split ? gp.backsub_split : std::max<size_t>(bprow.size() / 64, 2048);
				backsub_done = ok && gpu::res_backsub((uint32_t)bprow.size(), bprow.data(), bpcol.data(), (uint32_t)n_split);
				// a failed back substitution leaves valid rows (each written row is the exact result of
				// the batches applied to it): the CPU completes it if the GPU can still be read
				bs_failed = !backsub_done && gpu::res_alive();
				ok = backsub_done || bs_failed;
			}
			int result = 0;
			if (ok && sink != nullptr && !bs_failed && (backsub_done || !opt->is_back_sub)) {
				SPR_T0(spr_wb);
				ok = gpu::res_download_pieces((uint32_t)R, rows.data(), true,
					[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t* val) {
						return (*sink)(rows.data() + r0, r1 - r0, o, col, val);
					});
				SPR_ADD(gpu_res_writeback, spr_wb);
				result = ok ? 2 : 0;
			}
			else if (ok) {
				// into new rows first: mat keeps the input until the whole result is here
				SPR_T0(spr_wb);
				std::vector<sparse_vec<T, index_t>> fresh(R);
				ok = gpu::res_download_pieces((uint32_t)R, rows.data(), true,
					[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t* val) {
						pool.detach_loop<size_t>(r0, r1, [&](size_t i) {
							const size_t b = o[i - r0];
							vec_set_full(fresh[i], o[i - r0 + 1] - b, col + b, val + b);
							});
						pool.wait();
						return true;
					});
				if (ok)
					for (size_t i = 0; i < R; i++)
						mat[rows[i]] = std::move(fresh[i]);
				SPR_ADD(gpu_res_writeback, spr_wb);
				result = ok ? 1 : 0;
			}
			if (result == 0)
				backsub_done = false;
			SPR_DO({
				auto& rs = gpu::res_profile();
				auto& pd = SparseRREF::prof::d();
				pd.gpu_res_rescale += rs.rescale; pd.gpu_res_prep += rs.prep; pd.gpu_res_kernel += rs.kernel;
				pd.gpu_res_compact += rs.compact; pd.gpu_res_xfer += rs.download; pd.gpu_res_compactions += rs.compactions;
				pd.gpu_res_backsub += rs.backsub;
				pd.gpu_res_calls++;
			});
			gpu::res_end();
			SPR_ADD(gpu_res, spr_rp);
			return result;
		}
	}
#endif

	// One level of a known pivot list is applied in one pass (schur_complete), which needs the same
	// order as the back substitution within the level (backsub_order), on the rows as they are when
	// the level is reached. Returns false if the level's pivot block is not triangular in any order
	// (it is then applied one pivot at a time).
	template <typename T, typename index_t>
	bool order_level(const sparse_mat<T, index_t>& mat, std::vector<pivot_t<index_t>>& lv, thread_pool& pool) {
		// every pivot row must hold its pivot column already
		for (auto [r, c] : lv)
			if (mat.find(r, c) == nullptr)
				return false;
		return backsub_order(lv, mat.ncol, [&](size_t r, auto&& visit) {
			const auto& v = mat[r];
			for (size_t j = 0; j < v.nnz(); j++)
				visit((size_t)v(j));
			}, &pool);
	}

	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> sparse_mat_direct_rref_part(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& sub_pivots,
		const field_t& F, rref_option_t opt, schur_helper_buffer<T, index_t>& g_helper) {

		if (sub_pivots.size() <= 1)
			return std::vector<pivot_t<index_t>>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();
		
		std::vector<pivot_t<index_t>> combined_pivots;
		size_t nnz = 0;
		size_t total_rank = 0;
		for (auto& p : sub_pivots) {
			combined_pivots.insert(combined_pivots.end(), p.begin(), p.end());
			for (auto [r, c] : p) {
				nnz += mat[r].nnz();
				total_rank++;
			}
		}

		double density = (double)(nnz) / (total_rank * mat.ncol);
		auto schur_complete_func = schur_strategy<T, index_t>(density);

		std::vector<size_t> leftrows;
		leftrows.reserve(total_rank);
		for (auto rp = sub_pivots.rbegin(); rp != sub_pivots.rend(); rp++) {
			for (auto p = rp->rbegin(); p != rp->rend(); p++) {
				leftrows.push_back(p->r);
			}
		}

		// combined_pivots is rebuilt in the order the pivots are applied (levels reordered where needed)
		combined_pivots.clear();
		auto apply_level = [&](const std::vector<pivot_t<index_t>>& pivs) {
			combined_pivots.insert(combined_pivots.end(), pivs.begin(), pivs.end());

			// rescale the pivots
			pool.detach_loop(0, pivs.size(), [&](size_t j) {
				auto [r, c] = pivs[j];
				T scalar = scalar_inv(*mat.find(r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			// remove the used rows
			leftrows.resize(leftrows.size() - pivs.size());
			pool.wait();

			// upper solver
			schur_index_guard<T, index_t> spr_index(g_helper, pivs);
			const bool spr_gpu = schur_try_gpu(mat, leftrows, pivs, F, pool, g_helper, opt->policy);
			if (!spr_gpu)
			pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
				auto id = SparseRREF::thread_id();
				schur_helper<T, index_t> helper(g_helper, id);
				for (size_t j = s; j < e; j++) {
					schur_complete_func(mat, leftrows[j], pivs, F, helper);
					if (opt->abort)
						return;
				}
				}, ((leftrows.size() < 20 * nthreads) ? 0 : leftrows.size() / 10));

			pool.wait();
			};

		std::vector<char> in_batch;
		for (const auto& level : sub_pivots) {
			if (level.size() == 0)
				continue;
			// one pass if it can be ordered; otherwise batches (level_batch): in the given order, else in
			// the reverse order, else greedy; each try starts from the rows as they were at the level
			std::vector<std::pair<size_t, sparse_vec<T, index_t>>> saved;
			std::vector<size_t> leftrows_saved;
			const size_t combined_at = combined_pivots.size();
			bool done = false;
			for (int attempt = 0; attempt < 3 && !done && !opt->abort; attempt++) {
				const bool reverse = attempt == 1, greedy = attempt == 2;
				if (attempt > 0) {
					if (saved.empty())
						break;
					for (auto& [r, v] : saved)
						mat[r] = v;
					leftrows = leftrows_saved;
					combined_pivots.resize(combined_at);
					SPR_DO(SparseRREF::prof::d().level_retries++);
				}
				auto rest = level;
				bool stuck = false;
				while (!rest.empty() && !opt->abort) {
					auto lv = rest;
					if (order_level(mat, lv, pool)) {
						apply_level(lv);
						rest.clear();
						break;
					}
					if (saved.empty()) {
						leftrows_saved = leftrows;
						for (auto r : leftrows)
							saved.emplace_back(r, mat[r]);
					}
					// the level's rows still to come are the last of leftrows; the batch's go last
					const size_t before = rest.size();
					std::vector<pivot_t<index_t>> batch;
					level_batch(rest, mat.ncol, [&](size_t r, auto&& visit) {
						for (size_t j = 0; j < mat[r].nnz(); j++)
							visit((size_t)mat[r](j));
						}, batch, reverse, greedy);
					if (batch.empty()) { // the next pivot has no entry in this order
						stuck = true;
						break;
					}
					in_batch.resize(mat.nrow, 0);
					for (auto [r, c] : batch)
						in_batch[r] = 1;
					std::stable_partition(leftrows.end() - before, leftrows.end(), [&](size_t r) { return !in_batch[r]; });
					for (auto [r, c] : batch)
						in_batch[r] = 0;
					SPR_DO(SparseRREF::prof::d().level_batches++);
					apply_level(batch);
				}
				done = !stuck;
			}
			if (!done && !opt->abort)
				throw std::invalid_argument("sparse_mat_direct_rref: the pivots of a level cannot be applied "
					"in the given order, its reverse or a greedy one (a pivot entry is missing or cancels)");
			if (opt->abort)
				break;
		}

		return combined_pivots;
	}

	// if already know the pivots, we can directly do the rref
	// if the pivots are of submatrix, it is dangerous to set clear_zero_row to true
	// sink (GPU builds only): the result of the GPU elimination may go there instead of mat (then
	// *delivered is set and mat's eliminated rows are not)
	template <typename T, typename index_t>
	void sparse_mat_direct_rref(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt, const bool clear_zero_row = true,
		[[maybe_unused]] const replay_sink_t* sink = nullptr, bool* delivered = nullptr) {
		if (delivered != nullptr)
			*delivered = false;
		auto& pool = opt->pool;
		constexpr index_t sv = index_sval<index_t>();

		// first set rows not in pivots to zero
		std::vector<index_t> rowset(mat.nrow, sv);
		size_t total_rank = 0;
		for (auto p : pivots) {
			total_rank += p.size();
			for (auto [r, c] : p)
				rowset[r] = c;
		}
		if (clear_zero_row) {
			// clear the rows not in pivots
			for (size_t i = 0; i < mat.nrow; i++)
				if (rowset[i] == sv)
					mat[i].clear();
		}
		else {
			std::fill(rowset.begin(), rowset.end(), 0);
		}

		for (auto [r, c] : pivots[0]) {
			mat[r].zero();
			mat[r].push_back(c, 1);
		}

		for (size_t i = 0; i < mat.nrow; i++) {
			mat[i].compress();
		}

		if (opt->eliminate_one_nnz) {
			std::vector<index_t> donelist(mat.nrow, sv);
			eliminate_row_with_one_nnz(mat, donelist, opt);
		}

		// then do the elimination parallelly
		auto nthreads = pool.get_thread_count();
		schur_helper_buffer<T, index_t> g_helper(nthreads, mat.ncol);

		size_t rank = pivots[0].size();

		double density = (double)mat.nnz() / (total_rank * mat.ncol);
		auto schur_complete_func = schur_strategy<T, index_t>(density);

		std::vector<size_t> leftrows;
		leftrows.reserve(total_rank);
		for (auto rp = pivots.rbegin(); rp != pivots.rend(); rp++) {
			for (auto p = rp->rbegin(); p != rp->rend(); p++) {
				leftrows.push_back(p->r);
			}
		}
		leftrows.resize(leftrows.size() - pivots[0].size());
		
		int bitlen_nrow = num_digits(total_rank);
		int bitlen_nnz = num_digits(mat.nnz()) + 1;

		bool spr_on_gpu = false, spr_backsub_done = false;
#ifdef SPARSERREF_USE_CUDA
		{
			const int r = direct_rref_gpu(mat, pivots, leftrows, F, opt, spr_backsub_done, sink);
			if (r == 2) {
				if (delivered != nullptr)
					*delivered = true;
				return;
			}
			spr_on_gpu = r == 1;
		}
#endif

		std::vector<pivot_t<index_t>> used_pivots;
		for (size_t i = 1; !spr_on_gpu && i < pivots.size(); i++) {
			if (pivots[i].size() == 0)
				continue;

			used_pivots.clear();
			if (pivots[i].size() < 512) {
				size_t trank = 0;
				std::vector<std::vector<pivot_t<index_t>>> sub_pivots;
				for (size_t ni = i; ni < pivots.size(); ni++) {
					trank += pivots[ni].size();
					sub_pivots.push_back(pivots[ni]);
					if (trank > 4096) {
						i = ni;
						break;
					}	
				}
				used_pivots = sparse_mat_direct_rref_part(mat, sub_pivots, F, opt, g_helper);
			}

			if (used_pivots.size() == 0) {
				used_pivots = pivots[i];
				bool rescaled = false;
				if (!order_level(mat, used_pivots, pool)) {
					// not triangular in any order: in batches within the level (sparse_mat_direct_rref_part),
					// then the combined order applies to the later levels in one pass
					std::vector<std::vector<pivot_t<index_t>>> grp{ pivots[i], {} };
					auto done = sparse_mat_direct_rref_part(mat, grp, F, opt, g_helper);
					if (!done.empty()) {
						used_pivots = std::move(done);
						rescaled = true;
					}
				}
				// rescale the pivots
				if (!rescaled) {
					pool.detach_loop(0, used_pivots.size(), [&](size_t j) {
						auto [r, c] = used_pivots[j];
						T scalar = scalar_inv(*mat.find(r, c), F);
						sparse_vec_rescale(mat[r], scalar, F);
						mat[r].reserve(mat[r].nnz());
						});
					pool.wait();
				}
			}

			leftrows.resize(leftrows.size() - used_pivots.size());
			if (leftrows.size() == 0) // done
				break;

			// upper solver
			// TODO: check mode
			std::atomic<size_t> cc = 0;
			size_t old_cc = cc.load(std::memory_order_relaxed);
			schur_index_guard<T, index_t> spr_index(g_helper, used_pivots);
			const bool spr_gpu = schur_try_gpu(mat, leftrows, used_pivots, F, pool, g_helper, opt->policy);
			if (!spr_gpu)
			pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
				auto id = SparseRREF::thread_id();
				schur_helper<T, index_t> helper(g_helper, id);
				for (size_t j = s; j < e; j++) {
					schur_complete_func(mat, leftrows[j], used_pivots, F, helper);
					cc++;
					if (opt->abort)
						return;
				}
				}, ((leftrows.size() < 20 * nthreads) ? 0 : leftrows.size() / 10));
			if (spr_gpu)
				cc = leftrows.size();

			if (opt->verbose) {
				auto cn = clocknow();
				const size_t printstep = opt->print_step > 0 ? (size_t)opt->print_step : 1;
				while (cc.load(std::memory_order_relaxed) < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						pool.wait(); // the running tasks capture this frame's locals
						return;
					}
					const size_t cur_cc = cc.load(std::memory_order_relaxed);
					if (cur_cc - old_cc > printstep) {
						auto now_nnz = mat.nnz();
						progress(opt, "Row")
							.add("%*zu/%zu", bitlen_nrow, (size_t)std::floor(rank + (cur_cc * 1.0 / leftrows.size()) * used_pivots.size()), total_rank)
							.add("  nnz: %*zu", bitlen_nnz, now_nnz)
							.add("  alloc: %zu", mat.alloc())
							.add("  speed: %g row/s", rate_per_second((((cur_cc - old_cc) * 1.0 / leftrows.size()) * used_pivots.size()), usedtime(cn, clocknow())))
							.add("          ");
						old_cc = cur_cc;
						cn = clocknow();
					}
					// block until the pool makes progress instead of spinning
					pool.wait_for(progress_poll_interval);
				}
			}
			pool.wait();
			rank += used_pivots.size();
		}
		if (opt->verbose) {
			progress_end(opt);
		}

		if (opt->is_back_sub && !spr_backsub_done)
			triangular_solver_2(mat, pivots, F, opt);
	}

	// a forward elimination whose values stay on the GPU (sparse_mat_rref_forward_gpu): the host
	// keeps the row structure up to date for the pivot search, the values come back at the end
	struct fwd_gpu_state {
		bool use = false;    // may start
		bool active = false; // the rows are on the GPU
		bool failed = false; // a GPU step failed and its state was lost: the host values are not usable
		bool tried = false;  // the upload was attempted
		bool recovered = false; // a GPU step failed, the rows came back and the CPU took over
		bool search = false;    // the pivot search runs on the GPU too: the host rows are not kept up to date
		std::vector<uint32_t> rows; // the uploaded rows
	};

#ifdef SPARSERREF_USE_CUDA
	// uploads the rows still to be eliminated and the pivot rows chosen so far (levels >= 1, which the
	// back substitution on the GPU needs) once the trailing matrix is large enough; the uploaded rows
	// then keep their columns only on the host
	template <typename T, typename index_t>
	void fwd_gpu_upload(sparse_mat<T, index_t>& mat, const std::vector<size_t>& leftrows,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots, const field_t& F,
		rref_option_t opt, fwd_gpu_state& gs) {
		if constexpr (std::is_same_v<T, ulong>) {
			const auto& gp = opt->policy.gpu;
			const size_t R = leftrows.size();
			if (R < gp.call_min_rows || mat.nrow > UINT32_MAX || mat.ncol > UINT32_MAX || F.get_prime() >= (1ULL << 63))
				return;
			size_t nnz = 0;
			for (auto r : leftrows)
				nnz += mat[r].nnz();
			if (nnz < gp.min_nnz)
				return; // not yet: tried again next round
			gs.tried = true;
			if (!gpu_go(gp, nnz))
				return;
			SPR_T0(spr_up);
			auto& pool = opt->pool;
			gs.rows.clear();
			for (size_t l = 1; l < pivots.size(); l++)
				for (auto [r, c] : pivots[l])
					gs.rows.push_back((uint32_t)r);
			for (auto r : leftrows)
				gs.rows.push_back((uint32_t)r);
			const size_t U = gs.rows.size();
			std::vector<uint64_t> off(U + 1, 0);
			for (size_t i = 0; i < U; i++)
				off[i + 1] = off[i] + mat[gs.rows[i]].nnz();
			gs.active = gpu::res_begin(F.get_prime(), (uint32_t)mat.nrow, (uint32_t)mat.ncol, (uint32_t)U,
				gs.rows.data(), off.data(), [&](uint32_t r0, uint32_t r1, uint32_t* col, uint64_t* val) {
					pool.detach_loop<size_t>(r0, r1, [&](size_t i) {
						auto& v = mat[gs.rows[i]];
						const size_t a = off[i] - off[r0], n = v.nnz();
						if constexpr (sizeof(index_t) == 4)
							std::memcpy(col + a, v.indices, n * 4);
						else
							for (size_t k = 0; k < n; k++)
								col[a + k] = (uint32_t)v.indices[k];
						std::memcpy(val + a, v.entries, n * 8);
						});
					pool.wait();
				});
			if (!gs.active)
				gpu::res_end();
			else {
				// the values are on the GPU now
				pool.detach_loop<size_t>(0, U, [&](size_t i) {
					auto& v = mat[gs.rows[i]];
					if (v.entries != nullptr) {
						s_free(v.entries);
						v.entries = nullptr;
					}
					});
				pool.wait();
			}
			SPR_ADD(gpu_fwd_xfer, spr_up);
		}
	}

	template <typename index_t>
	bool fwd_gpu_rescale(const std::vector<pivot_t<index_t>>& piv) {
		std::vector<uint32_t> prow(piv.size()), pcol(piv.size());
		for (size_t j = 0; j < piv.size(); j++) {
			prow[j] = (uint32_t)piv[j].r;
			pcol[j] = (uint32_t)piv[j].c;
		}
		return gpu::res_rescale((uint32_t)piv.size(), prow.data(), pcol.data());
	}

	// one round's Schur update on the GPU, then the new structure of the rows it wrote
	template <typename T, typename index_t>
	bool fwd_gpu_schur(sparse_mat<T, index_t>& mat, const std::vector<size_t>& leftrows,
		const std::vector<pivot_t<index_t>>& piv, thread_pool& pool, const bool with_structure = true) {
		if constexpr (std::is_same_v<T, ulong>) {
			if (leftrows.empty() || piv.empty())
				return true;
			SPR_T0(spr_k);
			std::vector<uint32_t> prow(piv.size()), pcol(piv.size()), rows(leftrows.size());
			for (size_t j = 0; j < piv.size(); j++) {
				prow[j] = (uint32_t)piv[j].r;
				pcol[j] = (uint32_t)piv[j].c;
			}
			for (size_t i = 0; i < leftrows.size(); i++)
				rows[i] = (uint32_t)leftrows[i];
			if (!gpu::res_schur((uint32_t)piv.size(), prow.data(), pcol.data(), (uint32_t)rows.size(), rows.data()))
				return false;
			SPR_ADD(gpu_fwd_schur, spr_k);
			if (!with_structure) { // the search runs on the GPU: nothing comes back
				SPR_DO(SparseRREF::prof::d().gpu_fwd_rounds++);
				return true;
			}
			SPR_T0(spr_d);
			const std::vector<uint32_t> changed = gpu::res_changed();
			const bool ok = gpu::res_download_pieces((uint32_t)changed.size(), changed.data(), false,
				[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t*) {
					pool.detach_loop<size_t>(r0, r1, [&](size_t i) {
						const size_t b = o[i - r0];
						vec_set_structure(mat[changed[i]], o[i - r0 + 1] - b, col + b);
						});
					pool.wait();
					return true;
				});
			SPR_ADD(gpu_fwd_xfer, spr_d);
			SPR_DO(SparseRREF::prof::d().gpu_fwd_rounds++);
			return ok;
		}
		else
			return false;
	}

	// after a failed GPU step: if the matrix on the GPU can still be read, its rows (with values) come
	// back and the elimination goes on on the CPU from the same round (the step is redone there:
	// rescaling a rescaled row and updating an updated row change nothing, since an updated row holds
	// no pivot column of the round). Returns false if the GPU state is lost.
	template <typename T, typename index_t>
	bool fwd_gpu_recover(sparse_mat<T, index_t>& mat, fwd_gpu_state& gs, thread_pool& pool) {
		if constexpr (std::is_same_v<T, ulong>) {
			bool ok = gpu::res_alive() && gpu::res_download_pieces((uint32_t)gs.rows.size(), gs.rows.data(), true,
				[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t* val) {
					pool.detach_loop<size_t>(r0, r1, [&](size_t i) {
						const size_t b = o[i - r0];
						vec_set_full(mat[gs.rows[i]], o[i - r0 + 1] - b, col + b, val + b);
						});
					pool.wait();
					return true;
				});
			gpu::res_end();
			gs.active = false;
			if (ok)
				gs.recovered = true;
			SPR_DO(if (ok) SparseRREF::prof::d().gpu_fwd_recoveries++);
			return ok;
		}
		else
			return false;
	}
	// one round's pivot search on the GPU (gpu::res_search); empty[i] tells whether leftrows[i] is empty
	template <typename index_t>
	bool fwd_gpu_search(const std::vector<size_t>& leftrows, const std::vector<index_t>& leftcols,
		std::vector<pivot_t<index_t>>& ps, std::vector<uint8_t>& empty, uint64_t& nnz) {
		std::vector<uint32_t> lr(leftrows.size()), lc(leftcols.size()), pr, pc;
		for (size_t i = 0; i < leftrows.size(); i++)
			lr[i] = (uint32_t)leftrows[i];
		for (size_t i = 0; i < leftcols.size(); i++)
			lc[i] = (uint32_t)leftcols[i];
		if (!gpu::res_search((uint32_t)lr.size(), lr.data(), (uint32_t)lc.size(), lc.data(), pr, pc, empty, nnz))
			return false;
		ps.clear();
		for (size_t j = 0; j < pr.size(); j++)
			ps.emplace_back((index_t)pr[j], (index_t)pc[j]);
		return true;
	}

	// the back substitution order of the pivots of levels 1.. checked on the GPU's rows (the host rows
	// are not up to date when the search ran there); reordered on their columns only if needed
	template <typename T, typename index_t>
	bool fwd_gpu_backsub_order(const sparse_mat<T, index_t>& mat, const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		thread_pool& pool, std::vector<pivot_t<index_t>>& bp) {
		bp.clear();
		for (size_t i = 1; i < pivots.size(); i++)
			bp.insert(bp.end(), pivots[i].begin(), pivots[i].end());
		std::vector<uint32_t> prow(bp.size()), pcol(bp.size());
		for (size_t j = 0; j < bp.size(); j++) {
			prow[j] = (uint32_t)bp[j].r;
			pcol[j] = (uint32_t)bp[j].c;
		}
		const int ord = gpu::res_level_order((uint32_t)bp.size(), prow.data(), pcol.data());
		if (ord < 0)
			return false;
		if (ord == 1)
			return true;
		std::vector<uint64_t> off(bp.size() + 1, 0);
		std::vector<uint32_t> cols, slot(mat.nrow, UINT32_MAX);
		for (size_t j = 0; j < bp.size(); j++)
			slot[prow[j]] = (uint32_t)j;
		if (!gpu::res_download_pieces((uint32_t)bp.size(), prow.data(), false,
			[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t*) {
				for (uint32_t t = r0; t < r1; t++)
					off[t + 1] = off[t] + (o[t - r0 + 1] - o[t - r0]);
				cols.insert(cols.end(), col, col + o[r1 - r0]);
				return true;
			}))
			return false;
		backsub_order(bp, mat.ncol, [&](size_t r, auto&& visit) {
			const uint32_t t = slot[r];
			for (uint64_t x = off[t]; x < off[t + 1]; x++)
				visit((size_t)cols[x]);
			}, &pool);
		return true;
	}
#else
	template <typename index_t>
	bool fwd_gpu_search(const std::vector<size_t>&, const std::vector<index_t>&, std::vector<pivot_t<index_t>>&,
		std::vector<uint8_t>&, uint64_t&) { return false; }
	template <typename T, typename index_t>
	void fwd_gpu_upload(sparse_mat<T, index_t>&, const std::vector<size_t>&, const std::vector<std::vector<pivot_t<index_t>>>&,
		const field_t&, rref_option_t, fwd_gpu_state&) {}
	template <typename index_t>
	bool fwd_gpu_rescale(const std::vector<pivot_t<index_t>>&) { return false; }
	template <typename T, typename index_t>
	bool fwd_gpu_schur(sparse_mat<T, index_t>&, const std::vector<size_t>&, const std::vector<pivot_t<index_t>>&, thread_pool&, bool = true) { return false; }
	template <typename T, typename index_t>
	bool fwd_gpu_recover(sparse_mat<T, index_t>&, fwd_gpu_state&, thread_pool&) { return false; }
#endif

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_forward_impl(sparse_mat_subview<T, index_t> submat, const field_t& F, rref_option_t opt,
			fwd_gpu_state* gs) {
		// first canonicalize, sort and compress the matrix

		sparse_mat<T, index_t>& mat = submat.get_mat();
		constexpr index_t sv = index_sval<index_t>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });

		// store the pivots that have been used
		// sv is not used
		std::vector<index_t> rowpivs(mat.nrow, sv);
		std::vector<std::vector<pivot_t<index_t>>> pivots;
		std::vector<pivot_t<index_t>> n_pivots;

		pool.wait();

		size_t now_nnz = mat.nnz();
		double density = (double)now_nnz / (mat.nrow * mat.ncol);
		auto schur_complete_func = schur_strategy<T, index_t>(density);

		if (opt->abort)
			return pivots;

		// eliminate rows with only one non-zero entry
		if (opt->eliminate_one_nnz) {
			size_t count = eliminate_row_with_one_nnz_rec(submat, rowpivs, opt);
			now_nnz = mat.nnz();

			for (size_t i = 0; i < mat.nrow; i++) {
				if (opt->col_weight(rowpivs[i]) < 0)
					rowpivs[i] = sv; // mark as unused
				if (rowpivs[i] != sv)
					n_pivots.emplace_back(i, rowpivs[i]);
			}
		}
		pivots.push_back(n_pivots);

		if (opt->abort)
			return pivots;

		// use a vector to label the left columns
		std::vector<index_t> leftcols = perm_init((index_t)(mat.ncol));
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != sv)
				leftcols[rowpivs[i]] = sv; // mark as used
		}
		std::erase_if(leftcols, [](index_t i) { return i == sv; });

		schur_helper_buffer<T, index_t> g_helper(nthreads, mat.ncol);

		std::vector<size_t> leftrows;
		leftrows.reserve(mat.nrow);
		submat.traverse([&rowpivs, &mat, &leftrows](size_t i) {
			if (rowpivs[i] == index_sval<index_t>() && mat[i].nnz() > 0)
				leftrows.push_back(i);
			}
		);

		int method = opt->method;

		// the per-thread transposes serve the left search only (methods 0 and 2); with the
		// right-only search they were allocated (ncol row headers per thread) and never used
		std::vector<sparse_mat<bool, index_t>> tranmat_vec;

		if (method != 1) {
			tranmat_vec.resize(nthreads);
			for (auto& tmat : tranmat_vec)
				tmat = sparse_mat<bool, index_t>(mat.ncol, mat.nrow);
			pool.detach_loop(0, leftrows.size(), [&](size_t i) {
				auto id = thread_id();
				auto r = leftrows[i];
				for (size_t j = 0; j < mat[r].nnz(); j++) {
					tranmat_vec[id][mat[r](j)].push_back(r);
				}
				});
			pool.wait();
		}

		if (method == 0 || method == 2) {
			// sort pivots by nnz, it will be faster
			std::ranges::stable_sort(leftcols, std::less{},
				[&tranmat_vec](size_t r) {
					size_t nnz = 0;
					for (auto& tmat : tranmat_vec) {
						nnz += tmat[r].nnz();
					}
					return nnz;
				});
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = num_digits(now_nnz) + 1;
		int bitlen_ncol = num_digits(mat.ncol);

		bit_array tmp_set(mat.ncol);

		bool spr_res = false;
		size_t trim_nnz = mat.nnz(); // the matrix size at the last heap trim, or the largest since
		// col_weight is a pure function of the column: once for all rounds
		std::vector<int64_t> col_w_all(method == 1 ? 0 : mat.ncol);
		for (size_t c = 0; c < col_w_all.size(); c++)
			col_w_all[c] = opt->col_weight((int64_t)c);

		auto rank = pivots[0].size();
		while (rank < mat.ncol) {
			// the values move to the GPU once the trailing matrix is large enough
			if (gs != nullptr && gs->use && !gs->tried) {
				fwd_gpu_upload(mat, leftrows, pivots, F, opt, *gs);
				spr_res = gs->active;
#ifdef SPARSERREF_USE_CUDA
				// the pivot search too (method 0): the column lists and the host structure are no longer kept
				if (spr_res && method == 0 && opt->policy.gpu.search && gpu::res_search_weights(col_w_all.data())) {
					gs->search = true;
					for (auto& tranmat : tranmat_vec)
						pool.detach_loop<size_t>(0, tranmat.nrow, [&](size_t c) { tranmat[c].clear(); });
					pool.wait();
				}
#endif
			}
			const bool spr_gsearch = spr_res && gs->search;
			auto start = clocknow();

			std::vector<pivot_t<index_t>> ps;

			SPR_T0(spr_search);
			// case 1 is to use the right search method
			std::vector<uint8_t> g_empty; // in a GPU search: whether each row of leftrows is empty
			uint64_t g_nnz = 0;
			bool searched = false;
			if (spr_gsearch) {
				searched = fwd_gpu_search(leftrows, leftcols, ps, g_empty, g_nnz);
				if (!searched) {
					// the CPU takes over: the rows come back and its column lists are rebuilt
					gs->search = false;
					spr_res = false;
					if (!fwd_gpu_recover(mat, *gs, pool)) {
						gs->failed = true;
						return pivots;
					}
					for (auto& tranmat : tranmat_vec)
						for (auto c : leftcols)
							tranmat[c].zero();
					pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t i) {
						auto& tranmat = tranmat_vec[thread_id()];
						const auto r = leftrows[i];
						for (size_t j = 0; j < mat[r].nnz(); j++)
							tranmat[mat[r](j)].push_back(r, true);
						});
					pool.wait();
				}
			}
			if (!searched) {
				switch (method) {
				case 1: ps = pivots_search_right(mat, leftrows, opt->col_weight); break;
				default: ps = pivots_search(mat, tranmat_vec, leftrows, leftcols, opt->col_weight, &pool, col_w_all.data()); break;
				}
			}
			const bool gsearch_round = spr_gsearch && searched;
			SPR_DO(const double spr_search_s = SparseRREF::prof::since(spr_search));
			SPR_DO(SparseRREF::prof::d().search += spr_search_s);

			if (ps.size() == 0)
				break;

			n_pivots.clear();
			for (auto& [r, c] : ps) {
				rowpivs[r] = c;
				n_pivots.emplace_back(r, c);
			}
			pivots.push_back(n_pivots);

			SPR_T0(spr_rescale);
			if (spr_res && !fwd_gpu_rescale(n_pivots)) {
				spr_res = false;
				if (!fwd_gpu_recover(mat, *gs, pool)) {
					gs->failed = true;
					return pivots;
				}
			}
			if (!spr_res)
			pool.detach_loop(0, n_pivots.size(), [&](size_t i) {
				auto [r, c] = n_pivots[i];
				T scalar = scalar_inv(*mat.find(r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			size_t n_leftrows = 0;
			for (size_t i = 0; i < leftrows.size(); i++) {
				auto row = leftrows[i];
				if (rowpivs[row] != sv)
					continue;
				// with the search on the GPU the host rows are not up to date: the search told which are empty
				if (gsearch_round && spr_res ? g_empty[i] != 0 : mat[row].nnz() == 0) {
					mat[row].clear();
					continue;
				}
				leftrows[n_leftrows] = row;
				n_leftrows++;
			}
			leftrows.resize(n_leftrows);
			pool.wait();
			SPR_ADD(rescale, spr_rescale);

			if (opt->abort)
				return pivots;

			SPR_T0(spr_schur);
			auto print_info = [&](size_t kk, size_t x, bool& print_once, double& oldpr) {
				double pr = kk + (1.0 * ps.size() * x) / leftrows.size();
				// require real progress, otherwise a round-entry report would show speed 0
				if (opt->verbose && pr > oldpr && (print_once || pr - oldpr > opt->print_step)) {
					auto end = clocknow();
					now_nnz = mat.nnz();
					auto now_alloc = mat.alloc();
					progress(opt, "Col")
						.add("%*zu/%zu", bitlen_ncol, (size_t)pr, mat.ncol)
						.add("  rank: %*zu", bitlen_ncol, kk + ps.size())
						.add("  nnz: %*zu", bitlen_nnz, now_nnz)
						.add("  alloc: %*zu", bitlen_nnz, now_alloc)
						.add("  density: %8.6g%%", density_percent(now_nnz, mat.nrow, mat.ncol))
						.add("  speed: %8.6g col/s", rate_per_second(pr - oldpr, usedtime(start, end)))
						.add("    ");
					oldpr = pr;
					start = end;
					print_once = false;
				}
				};

			if (method == 1) {
				std::atomic<size_t> done_count = 0;
				schur_index_guard<T, index_t> spr_index(g_helper, n_pivots);
				bool spr_gpu = true;
				if (spr_res && !fwd_gpu_schur(mat, leftrows, n_pivots, pool, !(gsearch_round && gs->search))) {
					spr_res = false;
					if (!fwd_gpu_recover(mat, *gs, pool)) {
						gs->failed = true;
						return pivots;
					}
				}
				if (!spr_res)
					spr_gpu = schur_try_gpu(mat, leftrows, n_pivots, F, pool, g_helper, opt->policy);
				if (!spr_gpu)
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = thread_id();
					schur_helper<T, index_t> helper(g_helper, id);
					for (size_t i = s; i < e; i++) {
						schur_complete_func(mat, leftrows[i], n_pivots, F, helper);
						done_count++;
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));
				if (spr_gpu)
					done_count = leftrows.size();

				// remove used cols
				size_t localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (done_count.load(std::memory_order_relaxed) < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						pool.wait(); // the running tasks capture this frame's locals
						return pivots;
					}

					print_info(rank, done_count.load(std::memory_order_relaxed), print_once, oldpr);
					// block until the pool makes progress instead of polling
					pool.wait_for(progress_poll_interval);
				}
				pool.wait();
			}
			else {
				schur_index_guard<T, index_t> spr_index(g_helper, n_pivots);
				bool spr_gpu = true;
				if (spr_res && !fwd_gpu_schur(mat, leftrows, n_pivots, pool, !(gsearch_round && gs->search))) {
					spr_res = false;
					if (!fwd_gpu_recover(mat, *gs, pool)) {
						gs->failed = true;
						return pivots;
					}
				}
				if (!spr_res)
					spr_gpu = schur_try_gpu(mat, leftrows, n_pivots, F, pool, g_helper, opt->policy);

				SPR_T0(spr_tp);
				// remove used cols; the lists of the remaining columns are emptied for this round's rows
				size_t localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				const bool host_lists = !(spr_res && gs != nullptr && gs->search); // the CPU keeps the column lists
				for (auto c : leftcols) {
					if (!host_lists) {
						if (!tmp_set.test(c))
							leftcols[localcount++] = c;
						continue;
					}
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
						// the lists are rebuilt each round; one that kept far more room than its last
						// count gives it back (the trailing matrix shrinks over the rounds)
						for (auto& tranmat : tranmat_vec) {
							auto& t = tranmat[c];
							if (t.alloc() > 2 * t.nnz() + 16)
								t.reserve(t.nnz() + t.nnz() / 4);
							t.zero();
						}
					}
					else {
						for (auto& tranmat : tranmat_vec)
							tranmat[c].clear();
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once
				if (!spr_gpu) {
					// each worker files the rows it updates in its own column lists, while they are in cache
					std::atomic<size_t> done_count = 0;
					pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
						auto id = thread_id();
						schur_helper<T, index_t> helper(g_helper, id);
						auto& tranmat = tranmat_vec[id];
						for (size_t i = s; i < e; i++) {
							const auto row = leftrows[i];
							schur_complete_func(mat, row, n_pivots, F, helper);
							if (host_lists)
								for (size_t j = 0; j < mat[row].nnz(); j++)
									tranmat[mat[row](j)].push_back(row, true);
							done_count.fetch_add(1, std::memory_order_relaxed);
							if (opt->abort)
								break;
						}
						}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));
					while (done_count.load(std::memory_order_relaxed) < leftrows.size()) {
						if (opt->abort) {
							pool.purge();
							pool.wait(); // the running tasks capture this frame's locals
							return pivots;
						}
						print_info(rank, done_count.load(std::memory_order_relaxed), print_once, oldpr);
						// block until the pool makes progress instead of polling
						pool.wait_for(progress_poll_interval);
					}
					pool.wait();
				}
				else if (host_lists) {
					// the rows were updated on the GPU: their lists are filed here
					pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t i) {
						auto& tranmat = tranmat_vec[thread_id()];
						const auto row = leftrows[i];
						for (size_t j = 0; j < mat[row].nnz(); j++)
							tranmat[mat[row](j)].push_back(row, true);
						}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));
					pool.wait();
				}
				print_info(rank, leftrows.size(), print_once, oldpr);
				SPR_ADD(tp_rebuild, spr_tp);
			}

			SPR_DO(const double spr_schur_s = SparseRREF::prof::since(spr_schur));
			SPR_DO(SparseRREF::prof::d().schur += spr_schur_s);
			SPR_DO(SparseRREF::prof::d().rounds++);
			SPR_DO(SparseRREF::prof::d().fwd.push_back({ ps.size(), leftrows.size(), gsearch_round ? (size_t)g_nnz : mat.nnz(), spr_search_s, spr_schur_s, SparseRREF::prof::rss_kb() }));

			// if the number of new pivots is less than 1% of the total columns, 
			// it is very expansive to compute the transpose of the matrix
			// so we only search the right columns
			if (method == 2 && ps.size() < mat.ncol / 100) {
				method = 1;
				for (auto& tranmat : tranmat_vec)
					tranmat.clear();
			}

			rank += n_pivots.size();
			// on the CPU, once the matrix has shrunk by a quarter since the last trim (or its largest size)
			if (!spr_res) {
				const size_t nz = mat.nnz();
				if (nz * 4 < trim_nnz * 3 && trim_nnz > ((size_t)1 << 22)) {
					SPR_T0(spr_tr);
					trim_heap(opt->policy);
					SPR_ADD(trim, spr_tr);
					trim_nnz = nz;
				}
				else if (nz > trim_nnz)
					trim_nnz = nz;
			}
		}

		if (opt->verbose)
			progress_message(opt, "\n** Rank: %zu nnz: %zu\n", (size_t)rank, mat.nnz());

		return pivots;
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_forward(sparse_mat_subview<T, index_t> submat, const field_t& F, rref_option_t opt) {
		return sparse_mat_rref_forward_impl<T, index_t>(submat, F, opt, nullptr);
	}

	template <typename T, typename index_t>
	auto sparse_mat_rref_forward(sparse_mat<T, index_t>& mat, const field_t& F, rref_option_t opt) {
		return sparse_mat_rref_forward<T, index_t>(sparse_mat_subview<T, index_t>(mat), F, opt);
	}

	// The forward elimination with the values on the GPU when it is enabled (Z/p), followed by the
	// back substitution there if want_backsub (backsub_done tells). A failed GPU step hands the rows
	// back to the CPU, which goes on from the same round; only if the GPU's state is lost does the
	// elimination start again on the CPU from the input: restore(mat) rebuilds it when given (the
	// reconstruction reduces the rational matrix again), otherwise from a copy taken here.
	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_forward_gpu(sparse_mat<T, index_t>& mat, const field_t& F, rref_option_t opt,
			[[maybe_unused]] const bool want_backsub, bool& backsub_done,
			[[maybe_unused]] const std::type_identity_t<std::function<void(sparse_mat<T, index_t>&)>>& restore = {}) {
		backsub_done = false;
#ifdef SPARSERREF_USE_CUDA
		if constexpr (std::is_same_v<T, ulong>) {
			if (opt->policy.gpu.enabled && opt->policy.gpu.forward) {
				auto& pool = opt->pool;
				sparse_mat<T, index_t> backup;
				if (!restore)
					backup = mat;
				fwd_gpu_state gs;
				gs.use = true;
				auto pivots = sparse_mat_rref_forward_impl<T, index_t>(sparse_mat_subview<T, index_t>(mat), F, opt, &gs);
				if (gs.active && opt->abort) {
					// leave rows with values behind (or empty ones)
					if (!fwd_gpu_recover(mat, gs, pool))
						for (auto r : gs.rows)
							mat[r].clear();
					return pivots;
				}
				if (gs.active && !gs.failed) {
					bool ok = true;
					if (want_backsub) {
						SPR_T0(spr_bs);
						std::vector<pivot_t<index_t>> bp;
						if (gs.search) // the host rows are not up to date: the order is checked on the GPU
							ok = fwd_gpu_backsub_order(mat, pivots, pool, bp);
						else
							bp = backsub_pivots(mat, pivots, &pool); // on the columns kept on the host
						std::vector<uint32_t> prow(bp.size()), pcol(bp.size());
						for (size_t j = 0; j < bp.size(); j++) {
							prow[j] = (uint32_t)bp[j].r;
							pcol[j] = (uint32_t)bp[j].c;
						}
						const size_t bs_split = opt->policy.gpu.backsub_split;
						const size_t n_split = bs_split ? bs_split : std::max<size_t>(prow.size() / 64, 2048);
						backsub_done = ok && gpu::res_backsub((uint32_t)prow.size(), prow.data(), pcol.data(), (uint32_t)n_split);
						// a failed back substitution leaves valid rows: the CPU completes it if they can be read
						if (!backsub_done)
							ok = gpu::res_alive();
						SPR_ADD(gpu_fwd_backsub, spr_bs);
					}
					SPR_T0(spr_dl);
					ok = ok && gpu::res_download_pieces((uint32_t)gs.rows.size(), gs.rows.data(), true,
						[&](uint32_t r0, uint32_t r1, const uint64_t* o, const uint32_t* col, const uint64_t* val) {
							pool.detach_loop<size_t>(r0, r1, [&](size_t i) {
								const size_t b = o[i - r0];
								vec_set_full(mat[gs.rows[i]], o[i - r0 + 1] - b, col + b, val + b);
								});
							pool.wait();
							return true;
						});
					SPR_ADD(gpu_fwd_xfer, spr_dl);
					SPR_DO({
						auto& rs = gpu::res_profile();
						auto& pd = SparseRREF::prof::d();
						pd.gpu_fwd_compact += rs.compact; pd.gpu_fwd_compactions += rs.compactions;
						pd.gpu_search_count += rs.s_count; pd.gpu_search_cand += rs.s_cand; pd.gpu_search_greedy += rs.s_greedy;
					});
					gpu::res_end();
					gs.active = false;
					if (ok)
						return pivots;
					backsub_done = false;
					gs.failed = true;
				}
				if (!gs.failed)
					return pivots; // it ran on the CPU, or the CPU took over from a failed step
				// the GPU's state was lost: again on the CPU from the input (rows may hold columns only)
				backsub_done = false;
				pool.detach_loop<size_t>(0, mat.nrow, [&](size_t i) { mat[i].clear(); });
				pool.wait();
				if (restore)
					restore(mat);
				else
					mat = std::move(backup);
				return sparse_mat_rref_forward<T, index_t>(mat, F, opt);
			}
		}
#endif
		return sparse_mat_rref_forward<T, index_t>(mat, F, opt);
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref(sparse_mat<T, index_t>& mat, const field_t& F, rref_option_t opt) {

		SPR_T0(spr_fw);
		bool spr_bs_done = false;
		auto pivots = sparse_mat_rref_forward_gpu(mat, F, opt, opt->is_back_sub, spr_bs_done);
		SPR_ADD(forward, spr_fw);
		SPR_MEM("forward");

		if (opt->shrink_memory || opt->policy.memory == memory_mode::lean) {
			opt->pool.detach_loop(0, mat.nrow, [&](auto i) {
				mat[i].reserve(mat[i].nnz());
				});
			opt->pool.wait();
		}

		if (opt->abort)
			return pivots;

		if (opt->is_back_sub && !spr_bs_done) {
			if (opt->verbose)
				progress_message(opt, "\n>> Reverse solving: \n");
			// triangular_solver(mat, pivots, F, opt, -1);
			SPR_T0(spr_b);
			triangular_solver_2(mat, pivots, F, opt);
			SPR_ADD(backsub, spr_b);
			SPR_MEM("backsub");
		}

		return pivots;
	}

	template <typename index_t>
	int_t sparse_mat_denominator_lcm(const sparse_mat<rat_t, index_t>& mat) {
		int_t d = 1;
		for (size_t i = 0; i < mat.nrow; i++) {
			d = Flint::LCM(d, sparse_vec_denominator_lcm(mat[i]));
		}
		return d;
	}

	template <typename index_t>
	int_t sparse_mat_height(const sparse_mat<rat_t, index_t>& mat) {
		if (mat.nrow == 0)
			return 1;

		int_t h = sparse_vec_height(mat[0]);
		for (size_t i = 1; i < mat.nrow; i++) {
			int_t hi = sparse_vec_height(mat[i]);
			if (hi > h)
				h = hi;
		}
		return h;
	}

#ifdef SPARSERREF_USE_CUDA
	// The full reconstruction attempt of the chunks that keep residues, on the GPU: the chunks stream
	// through the GPU's two slots and the CPU turns each result into packed values (a copy) while the
	// GPU works on the next; the entries the GPU leaves to the CPU (quotients beyond 62 bits) are done
	// with rec. Returns false if the GPU was not used or failed: the chunks still unpacked are then
	// the CPU's (those packed here stay packed).
	template <typename RecFn, typename HardFn>
	bool reconstruct_gpu_chunks(std::vector<qpack::chunk>& chunks, const std::vector<size_t>& eoff,
		const std::vector<ulong>& primes_used, const int_t& mod, size_t nent, rref_option_t opt,
		RecFn&& rec, const fmpz_comb_struct* comb, HardFn&& note_hard, const bool keep) {
		const auto& gp = opt->policy.gpu;
		const slong k = (slong)primes_used.size();
		const int L = (int)fmpz_size(mod.data());
		if (!gp.reconstruct || !gpu::rr_supported((int)k, L) || !gpu_go(gp, nent))
			return false;
		auto& pool = opt->pool;
		std::vector<size_t> todo;
		for (size_t c = 0; c < chunks.size(); c++)
			if (!chunks[c].packed())
				todo.push_back(c);
		if (todo.empty())
			return true;
		std::vector<uint64_t> m_limbs(L, 0), N_limbs(L, 0), inv_prefix(k, 1);
		fmpz_get_ui_array((ulong*)m_limbs.data(), L, mod.data());
		int_t Nb = mod - 1;
		call_flint(fmpz_fdiv_q_2exp, Nb, Nb, 1);
		call_flint(fmpz_sqrt, Nb, Nb);
		fmpz_get_ui_array((ulong*)N_limbs.data(), L, Nb.data());
		for (slong i = 1; i < k; i++) {
			ulong pi = primes_used[i], prod = 1;
			for (slong j = 0; j < i; j++)
				prod = n_mulmod2_preinv(prod, primes_used[j] % pi, pi, n_preinvert_limb(pi));
			inv_prefix[i] = n_invmod(prod, pi);
		}
		// the chunks in submissions of at most maxE entries (whole rows)
		const size_t maxE = gpu::rr_max_entries((int)k, L);
		struct sub_t { size_t c, li0, li1; };
		std::vector<sub_t> subs;
		std::vector<size_t> nsub(chunks.size(), 0);
		for (auto c : todo) {
			const auto& ch = chunks[c];
			for (size_t li = ch.li0; li < ch.li1;) {
				size_t l1 = li + 1;
				if (eoff[l1] - eoff[li] > maxE)
					return false; // a row larger than a submission
				while (l1 < ch.li1 && eoff[l1 + 1] - eoff[li] <= maxE)
					l1++;
				subs.push_back({ c, li, l1 });
				nsub[c]++;
				li = l1;
			}
		}
		gpu::rr_setup su;
		su.k = (int)k; su.primes = (const uint64_t*)primes_used.data(); su.inv_prefix = inv_prefix.data();
		su.L = L; su.m_limbs = m_limbs.data(); su.N_limbs = N_limbs.data();
		for (const auto& sb : subs) {
			su.max_entries = std::max(su.max_entries, eoff[sb.li1] - eoff[sb.li0]);
			su.max_rows = std::max(su.max_rows, sb.li1 - sb.li0);
		}
		SPR_T0(spr_gk);
		if (!gpu::rr_begin(su))
			return false;
		auto submit = [&](int slot, size_t si) -> bool {
			SPR_T0(spr_gs);
			const auto& sb = subs[si];
			const auto& ch = chunks[sb.c];
			const size_t e0 = eoff[sb.li0] - ch.e0, E = eoff[sb.li1] - eoff[sb.li0], R = sb.li1 - sb.li0;
			uint64_t* st = gpu::rr_stage(slot);
			uint64_t* rs = gpu::rr_stage_rows(slot);
			if (!st || !rs)
				return false;
			const size_t nblk = 32;
			pool.detach_loop<size_t>(0, (size_t)k * nblk, [&](size_t x) {
				const size_t t = x / nblk, bk = x % nblk;
				const size_t lo = E * bk / nblk, hi = E * (bk + 1) / nblk;
				std::memcpy(st + t * su.max_entries + lo, ch.resid[t].get() + e0 + lo, (hi - lo) * 8);
				});
			for (size_t r = 0; r <= R; r++)
				rs[r] = eoff[sb.li0 + r] - eoff[sb.li0];
			pool.wait();
			SPR_ADD(gpu_rr_stage, spr_gs);
			return gpu::rr_submit(slot, E, R);
			};
		// a chunk's values come in one segment per submission and are committed when all are there
		struct pend_t { std::unique_ptr<uint32_t[]> meta; std::vector<qpack::segment> segs; size_t done = 0; bool failed = false; };
		std::vector<pend_t> pend(chunks.size());
		// the packed values of submission sb from the GPU's view
		auto convert = [&](const sub_t& sb, const gpu::rr_view& vw) {
			auto& ch = chunks[sb.c];
			auto& pd = pend[sb.c];
			pd.done++;
			if (pd.failed)
				return;
			const size_t eb = eoff[sb.li0] - ch.e0; // the submission's first entry within the chunk
			const size_t E = eoff[sb.li1] - eoff[sb.li0], R = sb.li1 - sb.li0;
			std::atomic<size_t> n_fail = 0, n_cpu = 0, first_fail = SIZE_MAX;
			pool.detach_blocks<size_t>(0, E, [&](const size_t s, const size_t e) {
				size_t f = 0, c = 0;
				for (size_t x = s; x < e; x++) {
					const unsigned st = vw.meta[x] >> 30;
					if (st == 1) {
						if (f == 0) {
							size_t cur = first_fail.load();
							while (x < cur && !first_fail.compare_exchange_weak(cur, x)) {}
						}
						f++;
					}
					else if (st == 2)
						c++;
				}
				n_fail += f;
				n_cpu += c;
				});
			pool.wait();
			if (n_fail > 0) {
				note_hard(ch.e0 + eb + first_fail.load());
				pd.failed = true;
				pd.meta.reset();
				pd.segs.clear();
				return;
			}
			if (!pd.meta)
				pd.meta.reset(new uint32_t[ch.entries() > 0 ? ch.entries() : 1]);
			uint32_t* meta = pd.meta.get() + eb;
			qpack::segment sg;
			sg.li0 = sb.li0; sg.li1 = sb.li1;
			sg.row_limb.resize(R + 1);
			if (n_cpu == 0) {
				const size_t total = vw.row_limb[R];
				sg.limbs = qpack::limb_buf(total);
				const size_t nblk = 64;
				pool.detach_loop<size_t>(0, nblk, [&](size_t bk) {
					const size_t lo = total * bk / nblk, hi = total * (bk + 1) / nblk;
					std::memcpy(sg.limbs.p + lo, vw.limbs + lo, (hi - lo) * 8);
					const size_t elo = E * bk / nblk, ehi = E * (bk + 1) / nblk;
					std::memcpy(meta + elo, vw.meta + elo, (ehi - elo) * 4);
					});
				pool.wait();
				std::memcpy(sg.row_limb.data(), vw.row_limb, (R + 1) * 8);
			}
			else {
				// the entries left to the CPU, then rows sized with them
				SPR_DO(SparseRREF::prof::d().gpu_rr_cpu += n_cpu.load());
				std::vector<size_t> where;
				for (size_t x = 0; x < E; x++)
					if ((vw.meta[x] >> 30) == 2)
						where.push_back(x);
				std::vector<rat_t> vals(where.size());
				std::atomic<bool> failed = false;
				pool.detach_blocks<size_t>(0, where.size(), [&](const size_t s, const size_t e) {
					fmpz_comb_temp_t temp;
					fmpz_comb_temp_init(temp, comb);
					std::vector<ulong> rk(k);
					int_t z;
					for (size_t i = s; i < e && !failed.load(std::memory_order_relaxed); i++)
						if (!rec(ch, ch.e0 + eb + where[i], temp, rk.data(), z, vals[i])) {
							failed = true;
							note_hard(ch.e0 + eb + where[i]);
						}
					fmpz_comb_temp_clear(temp);
					});
				pool.wait();
				if (failed) {
					pd.failed = true;
					pd.meta.reset();
					pd.segs.clear();
					return;
				}
				std::vector<uint64_t> extra(R, 0);
				for (size_t i = 0; i < where.size(); i++) {
					const size_t ge = eoff[sb.li0] + where[i];
					const size_t li = (size_t)(std::upper_bound(eoff.begin() + sb.li0, eoff.begin() + sb.li1 + 1, ge) - eoff.begin()) - 1;
					extra[li - sb.li0] += qpack::fmpq_limbs(vals[i].data());
				}
				sg.row_limb[0] = 0;
				for (size_t r = 0; r < R; r++)
					sg.row_limb[r + 1] = sg.row_limb[r] + (vw.row_limb[r + 1] - vw.row_limb[r]) + extra[r];
				sg.limbs = qpack::limb_buf(sg.row_limb[R]);
				pool.detach_loop<size_t>(0, R, [&](size_t r) {
					const uint64_t* src = vw.limbs + vw.row_limb[r];
					uint64_t* dst = sg.limbs.p + sg.row_limb[r];
					const size_t li = sb.li0 + r;
					for (size_t x = eoff[li] - eoff[sb.li0]; x < eoff[li + 1] - eoff[sb.li0]; x++) {
						const uint32_t m = vw.meta[x];
						if ((m >> 30) == 2) {
							const size_t i = (size_t)(std::lower_bound(where.begin(), where.end(), x) - where.begin());
							meta[x] = qpack::write_fmpq(dst, vals[i].data());
							continue;
						}
						const size_t n = qpack::meta_limbs(m, src);
						std::memcpy(dst, src, n * 8);
						src += n;
						dst += n;
						meta[x] = m;
					}
					});
				pool.wait();
			}
			pd.segs.push_back(std::move(sg));
			if (pd.done == nsub[sb.c]) {
				ch.meta = std::move(pd.meta);
				ch.segs = std::move(pd.segs);
				if (!keep)
					ch.drop_resid();
				SPR_DO(SparseRREF::prof::d().n_packed++);
			}
			};
		bool ok = submit(0, 0) && (subs.size() < 2 || submit(1, 1));
		for (size_t ti = 0; ok && ti < subs.size() && !opt->abort; ti++) {
			const int slot = (int)(ti & 1);
			gpu::rr_view vw;
			SPR_T0(spr_gw);
			if (!gpu::rr_wait(slot, vw)) {
				ok = false;
				break;
			}
			SPR_ADD(gpu_rr_wait, spr_gw);
			SPR_DO(SparseRREF::prof::d().gpu_rr_h2d += vw.h2d);
			SPR_DO(SparseRREF::prof::d().gpu_rr_kernel += vw.kernel);
			SPR_DO(SparseRREF::prof::d().gpu_rr_d2h += vw.d2h);
			SPR_T0(spr_gc);
			convert(subs[ti], vw);
			SPR_ADD(gpu_rr_convert, spr_gc);
			if (ti + 2 < subs.size() && !opt->abort && !submit(slot, ti + 2))
				ok = false;
		}
		gpu::rr_end();
		SPR_ADD(gpu_rr, spr_gk);
		return ok && !opt->abort;
	}
#endif

	// In this function, we try to rref the rational matrix by first doing the rref 
	// of the modular matrix, and then reconstruct the rational matrix from the modular matrix.
	// 
	// The prime is first choosen as nextprime(2^60). If it's not enough, we will try the next prime
	// and combine it by using the Chinese Remainder Theorem. 
	// We will keep trying until the reconstruction is successful or the user aborts.
	// 
	// The condition to stop reconstruct: H(d*E)*H(mat)*ncol < product of primes
	// where H is the height of a matrix (the maximal height of each entry), 
	// E is the reconstracted rref matrix
	// d is an integer such that d*E is a integer matrix
	// 
	// IMPORTANT: In this function, we assume that the sharp of reduced matrix on the modular field
	// is the same as the one on the rational field. This is not always true in math, for example,
	// the matrix [[1, 1], [1, p + 1]] is full rank over Q, but rank 1 over F_p. 
	// And it reduces [[1, 1], [0, 0]] on F_p but [[1,0], [0, 1]] on Q. However, in practice, 
	// this is very very rare if the input matrix comes from a real application and
	// the prime is choosen as a large prime like nextprime(2^60). 
	// Therefore, we assume that here. If the user is dealing with a very special matrix,
	// it is better to do it manually because usually there would be other problems from math side.
	// 
	// checkrank is only used for sparse_mat_inverse
	//
	// The result is built in packed form (sparse_qpack.h), in chunks of whole rows: a chunk keeps the
	// residues of every prime used until all its entries reconstruct, then only the reconstructed
	// values, which a later prime checks (n = d r mod p) instead of storing its residues. A value
	// that agrees with every residue and lies within the bounds of the current modulus is what the
	// reconstruction at that modulus returns (the fraction within the bounds is unique), so the prime
	// at which this stops and every value are those of reconstructing all entries at every prime.
	// Returns true with the result in out; false on abort or when checkrank fails (mat is then the
	// input, possibly with the rows outside the pivots cleared, as before).
	template <typename index_t>
	bool sparse_mat_rref_reconstruct_packed(sparse_mat<rat_t, index_t>& mat, rref_option_t opt,
		std::vector<std::vector<pivot_t<index_t>>>& pivots, sparse_qmat_packed<index_t>& out,
		const bool checkrank = false) {

		constexpr index_t sv = index_sval<index_t>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();
		out.clear();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });
		pool.wait();

		ulong prime = n_nextprime(1ULL << 60, 0);
		field_t F(FIELD_Fp, prime);

		SPR_T0(spr_red);
		sparse_mat<ulong, index_t> matul(mat.nrow, mat.ncol);
		pool.detach_loop(0, mat.nrow, [&](auto i) {
			matul[i] = mat[i] % F.mod;
			});
		pool.wait();
		SPR_ADD(reduce, spr_red);
		SPR_DO(SparseRREF::prof::d().primes = 1);
		SPR_MEM("reduced");

		int_t m_height = sparse_mat_height(mat);

		auto cs = clocknow();
		SPR_T0(spr_fw);
		bool spr_bs_done = false;
		pivots = sparse_mat_rref_forward_gpu(matul, F, opt, opt->is_back_sub, spr_bs_done,
			[&](sparse_mat<ulong, index_t>& m) {
				pool.detach_loop(0, mat.nrow, [&](auto i) { m[i] = mat[i] % F.mod; });
				pool.wait();
			});
		SPR_ADD(forward, spr_fw);
		SPR_MEM("forward");

		if (opt->abort)
			return false;

		// the rows grew by doubling during the elimination and keep up to twice their final
		// size; trim them before the back substitution
		pool.detach_loop(0, matul.nrow, [&](auto i) { matul[i].reserve(matul[i].nnz()); });
		pool.wait();

		if (checkrank) {
			size_t rank = 0;
			for (auto& p : pivots)
				rank += p.size();
			if (rank != mat.nrow)
				return false;
		}

		SPR_T0(spr_b);
		if (opt->is_back_sub && !spr_bs_done)
			triangular_solver_2(matul, pivots, F, opt);
		SPR_ADD(backsub, spr_b);
		SPR_MEM("backsub");
		if (opt->abort)
			return false;

		// the structure of the result: the rows with entries, their offsets and columns
		out.nrow = mat.nrow;
		out.ncol = mat.ncol;
		auto& leftrows = out.rows;
		auto& eoff = out.eoff;
		for (size_t i = 0; i < mat.nrow; i++)
			if (matul[i].nnz() > 0)
				leftrows.push_back(i);
		eoff.assign(leftrows.size() + 1, 0);
		for (size_t li = 0; li < leftrows.size(); li++)
			eoff[li + 1] = eoff[li] + matul[leftrows[li]].nnz();
		const size_t nent = eoff.back();
		std::vector<size_t> li_of_row(mat.nrow, SIZE_MAX);
		for (size_t li = 0; li < leftrows.size(); li++)
			li_of_row[leftrows[li]] = li;
		out.col.reset(new index_t[nent > 0 ? nent : 1]);
		pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t li) {
			const auto& v = matul[leftrows[li]];
			std::copy(v.indices, v.indices + v.nnz(), out.col.get() + eoff[li]);
			});
		pool.wait();

		// chunks of whole rows: about 64 of them (a chunk whose values turn out wrong is the unit that
		// is redone), between 4096 and 4M entries
		const size_t chunk_target = opt->policy.rr_chunk ? opt->policy.rr_chunk : std::clamp<size_t>(nent / 64, 4096, (size_t)1 << 22);
		auto& chunks = out.chunks;
		for (size_t li = 0; li < leftrows.size();) {
			size_t l1 = li + 1;
			while (l1 < leftrows.size() && eoff[l1 + 1] - eoff[li] <= chunk_target)
				l1++;
			qpack::chunk ch;
			ch.li0 = li; ch.li1 = l1; ch.e0 = eoff[li]; ch.e1 = eoff[l1];
			chunks.push_back(std::move(ch));
			li = l1;
		}
		std::vector<size_t> chunk_e0(chunks.size());
		size_t max_chunk = 0;
		for (size_t c = 0; c < chunks.size(); c++) {
			chunk_e0[c] = chunks[c].e0;
			max_chunk = std::max(max_chunk, chunks[c].entries());
		}
		auto chunk_of = [&](size_t e) -> size_t {
			return (size_t)(std::upper_bound(chunk_e0.begin(), chunk_e0.end(), e) - chunk_e0.begin()) - 1;
			};

		int_t mod = prime;
		std::vector<ulong> primes_used{ prime };
		std::vector<size_t> hard; // entries whose last reconstruction attempt failed
		std::mutex hard_mutex;
		auto note_hard = [&](size_t e) {
			std::lock_guard<std::mutex> lock(hard_mutex);
			if (std::find(hard.begin(), hard.end(), e) == hard.end()) {
				if (hard.size() >= 64)
					hard.erase(hard.begin());
				hard.push_back(e);
			}
			};
		auto any_unpacked = [&]() {
			for (auto& ch : chunks)
				if (!ch.packed())
					return true;
			return false;
			};
		auto all_packed = [&]() { return !any_unpacked(); };

		// Packed chunks also keep their residues while all residues fit this budget: a value found at a
		// prime whose height check fails can still be wrong, and then the residues are needed again
		// (rebuilding them from the values costs about a reconstruction when there are many primes).
		const size_t keep_bytes = opt->policy.rr_keep_mb << 20;
		auto keep_resid = [&](size_t nprimes) { return nprimes * nent * sizeof(ulong) <= keep_bytes; };

		// the current prime's residues of the entries of row li: from matul, or by entry from a replay
		// delivered by the GPU
		std::unique_ptr<ulong[]> cur;
		bool cur_valid = false;
		auto res_row = [&](size_t li) -> const ulong* {
			return cur_valid ? cur.get() + eoff[li] : matul[leftrows[li]].entries;
			};

		// the current prime's residues appended to every chunk that keeps residues
		auto store_residues = [&]() {
			const bool keep = keep_resid(primes_used.size() + 1);
			for (auto& ch : chunks) {
				if (ch.packed() && !keep)
					ch.drop_resid();
				if (!ch.packed() || !ch.resid.empty())
					ch.resid.emplace_back(new ulong[ch.entries() > 0 ? ch.entries() : 1]);
			}
			pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t li) {
				auto& ch = chunks[chunk_of(eoff[li])];
				if (ch.packed() && ch.resid.empty())
					return;
				const ulong* r = res_row(li);
				std::copy(r, r + (eoff[li + 1] - eoff[li]), ch.resid.back().get() + (eoff[li] - ch.e0));
				});
			pool.wait();
			};

		// once the current prime's values are in the residue store, matul is not read again before the
		// next prime rebuilds it from mat: its rows (the whole result modulo p) go back to the system
		auto release_matul = [&]() {
			if (opt->policy.memory == memory_mode::fast) // its memory serves the next prime
				return;
			pool.detach_loop<size_t>(0, matul.nrow, [&](size_t i) { matul[i].clear(); });
			pool.wait();
			trim_heap(opt->policy);
			SPR_MEM("matul released");
			};

		// row blocks of about equal entries within [li0, li1)
		auto row_blocks = [&](size_t li0, size_t li1, size_t nb) {
			std::vector<size_t> b{ li0 };
			const size_t total = eoff[li1] - eoff[li0];
			nb = std::max<size_t>(1, std::min(nb, li1 - li0));
			for (size_t li = li0; li < li1; li++)
				if ((eoff[li + 1] - eoff[li0]) * nb >= total * b.size() && b.size() < nb && li + 1 < li1)
					b.push_back(li + 1);
			b.push_back(li1);
			return b;
			};

		// the fixed-size reconstruction of sparse_rr_core.h for the current primes, where it applies
#ifdef SPARSERREF_HAVE_RR_CORE
		qpack::fast_rr frr;
#endif
		// reconstruct entry e of chunk ch from its residues into q (k primes; comb for k > 1)
		auto rec_entry = [&](const qpack::chunk& ch, size_t e, slong k, const fmpz_comb_t comb, fmpz_comb_temp_t temp,
			ulong* rk, int_t& z, rat_t& q) -> bool {
			if (k == 1) {
				z = ch.resid[0][e - ch.e0];
				return Flint::rational_reconstruct(q, z, mod);
			}
			for (slong t = 0; t < k; t++)
				rk[t] = ch.resid[t][e - ch.e0];
			fmpz_multi_CRT_ui(z.data(), rk, comb, temp, 0);
			return Flint::rational_reconstruct(q, z, mod);
			};
		// whether entry e of chunk ch has a fraction within the bounds (the candidate checks)
		auto rec_ok = [&](const qpack::chunk& ch, size_t e, slong k, const fmpz_comb_t comb, fmpz_comb_temp_t temp,
			ulong* rk, int_t& z, rat_t& q) -> bool {
#ifdef SPARSERREF_HAVE_RR_CORE
			if (frr.on()) {
				for (slong t = 0; t < k; t++)
					rk[t] = ch.resid[t][e - ch.e0];
				uint64_t buf[72], *p = buf; // at most 2 x 34 limbs
				uint32_t m;
				const int st = frr.entry((const uint64_t*)rk, p, m);
				if (st != 2)
					return st == 0;
			}
#endif
			return rec_entry(ch, e, k, comb, temp, rk, z, q);
			};

		// all entries of an unpacked chunk on the CPU; on success the chunk keeps the values only
		auto attempt_chunk_cpu = [&](qpack::chunk& ch, slong k, const fmpz_comb_t comb) -> bool {
			const size_t half = fmpz_size(mod.data()) / 2 + 1; // limbs of N = floor(sqrt((mod - 1) / 2)), at most
			const size_t per = qpack::entry_limbs(half, half);
			auto bounds = row_blocks(ch.li0, ch.li1, 4 * nthreads);
			const size_t nb = bounds.size() - 1;
			std::vector<qpack::segment> segs(nb);
			std::unique_ptr<uint32_t[]> meta(new uint32_t[ch.entries() > 0 ? ch.entries() : 1]);
			std::atomic<bool> failed = false;
			pool.detach_loop<size_t>(0, nb, [&](size_t b) {
				auto& sg = segs[b];
				sg.li0 = bounds[b]; sg.li1 = bounds[b + 1];
				sg.limbs = qpack::limb_buf((eoff[sg.li1] - eoff[sg.li0]) * per);
				sg.row_limb.resize(sg.li1 - sg.li0 + 1);
				uint64_t* p = sg.limbs.p;
				fmpz_comb_temp_t temp;
				if (k > 1)
					fmpz_comb_temp_init(temp, comb);
				std::vector<ulong> rk(k);
				int_t z;
				rat_t q;
#ifdef SPARSERREF_HAVE_RR_CORE
				qpack::common_den cden; // carried over the rows of the block: they usually share it
#endif
				for (size_t li = sg.li0; li < sg.li1 && !failed.load(std::memory_order_relaxed) && !opt->abort; li++) {
					sg.row_limb[li - sg.li0] = (uint64_t)(p - sg.limbs.p);
					for (size_t e = eoff[li]; e < eoff[li + 1]; e++) {
#ifdef SPARSERREF_HAVE_RR_CORE
						if (frr.on()) {
							for (slong t = 0; t < k; t++)
								rk[t] = ch.resid[t][e - ch.e0];
							const int st = frr.entry_cd((const uint64_t*)rk.data(), cden, p, meta[e - ch.e0]);
							if (st == 0 || st == 3)
								continue;
							if (st == 1) {
								failed = true;
								note_hard(e);
								break;
							}
						}
#endif
						if (!rec_entry(ch, e, k, comb, temp, rk.data(), z, q)) {
							failed = true;
							note_hard(e);
							break;
						}
						meta[e - ch.e0] = qpack::write_fmpq(p, q.data());
					}
				}
				sg.row_limb.back() = (uint64_t)(p - sg.limbs.p);
				sg.limbs.shrink((size_t)(p - sg.limbs.p));
				if (k > 1)
					fmpz_comb_temp_clear(temp);
				});
			pool.wait();
			if (failed || opt->abort)
				return false;
			ch.meta = std::move(meta);
			ch.segs = std::move(segs);
			if (!keep_resid(k))
				ch.drop_resid();
			SPR_DO(SparseRREF::prof::d().n_packed++);
			return true;
			};

		// for each packed chunk, whether all its values agree with the new prime p, whose residues are
		// the values of matul (one parallel pass over all packed chunks)
		auto verify_packed = [&](ulong p) -> std::vector<char> {
			nmod_t md;
			nmod_init(&md, p);
			std::unique_ptr<std::atomic<char>[]> flag(new std::atomic<char>[chunks.size() + 1]);
			for (size_t c = 0; c <= chunks.size(); c++)
				flag[c].store(0);
			struct task { size_t c, s, li0, li1; };
			std::vector<task> tasks;
			for (size_t c = 0; c < chunks.size(); c++) {
				if (!chunks[c].packed())
					continue;
				for (size_t si = 0; si < chunks[c].segs.size(); si++) {
					const auto& sg = chunks[c].segs[si];
					auto bb = row_blocks(sg.li0, sg.li1, std::max<size_t>(1, (eoff[sg.li1] - eoff[sg.li0]) >> 15));
					for (size_t i = 0; i + 1 < bb.size(); i++)
						tasks.push_back({ c, si, bb[i], bb[i + 1] });
				}
			}
			pool.detach_loop<size_t>(0, tasks.size(), [&](size_t ti) {
				const auto& T = tasks[ti];
				const auto& ch = chunks[T.c];
				const auto& sg = ch.segs[T.s];
				for (size_t li = T.li0; li < T.li1; li++) {
					if (flag[T.c].load(std::memory_order_relaxed))
						return;
					const ulong* r = res_row(li);
					const uint64_t* lp = sg.limbs.p + sg.row_limb[li - sg.li0];
					for (size_t x = eoff[li]; x < eoff[li + 1]; x++) {
						qpack::entry_view v;
						lp = qpack::read_entry(lp, ch.meta[x - ch.e0], v);
						ulong nm = mpn_mod_1((mp_srcptr)v.num, (mp_size_t)v.nl, p);
						if (v.neg && nm)
							nm = p - nm;
						const ulong dm = mpn_mod_1((mp_srcptr)v.den, (mp_size_t)v.dl, p);
						if (nmod_mul(dm, r[x - eoff[li]], md) != nm) {
							flag[T.c].store(1, std::memory_order_relaxed);
							return;
						}
					}
				}
				});
			pool.wait();
			std::vector<char> bad(chunks.size());
			for (size_t c = 0; c < chunks.size(); c++)
				bad[c] = flag[c].load();
			return bad;
			};

		// a packed chunk back to residues for the primes used so far (n d^-1 mod p_t); with many primes
		// the residues of n and d come from a remainder tree
		auto unpack_chunk = [&](qpack::chunk& ch) {
			const size_t k = primes_used.size(), E = ch.entries();
			std::vector<nmod_t> md(k);
			for (size_t t = 0; t < k; t++) {
				nmod_init(&md[t], primes_used[t]);
				ch.resid.emplace_back(new ulong[E > 0 ? E : 1]);
			}
			const bool tree = k >= 16;
			fmpz_comb_t ucomb;
			if (tree)
				fmpz_comb_init(ucomb, primes_used.data(), (slong)k);
			for (auto& sg : ch.segs) {
				pool.detach_blocks<size_t>(sg.li0, sg.li1, [&](const size_t s, const size_t e) {
					fmpz_comb_temp_t temp;
					if (tree)
						fmpz_comb_temp_init(temp, ucomb);
					int_t nz, dz;
					// up to MB entries at a time, so that each prime needs one inversion per batch
					constexpr size_t MB = 64;
					std::vector<ulong> nr(MB * k), dr(MB * k), pre(MB);
					std::vector<size_t> xs(MB);
					size_t nb = 0;
					auto flush = [&]() {
						for (size_t t = 0; t < k; t++) {
							const ulong p = primes_used[t];
							ulong acc = 1;
							for (size_t i = 0; i < nb; i++) {
								pre[i] = acc;
								acc = nmod_mul(acc, dr[i * k + t], md[t]);
							}
							ulong inv = n_invmod(acc, p); // 1 / (d_0 ... d_{nb-1})
							for (size_t i = nb; i-- > 0;) {
								const ulong di = nmod_mul(inv, pre[i], md[t]); // 1 / d_i
								inv = nmod_mul(inv, dr[i * k + t], md[t]);
								ch.resid[t][xs[i] - ch.e0] = nmod_mul(nr[i * k + t], di, md[t]);
							}
						}
						nb = 0;
						};
					for (size_t li = s; li < e; li++) {
						const uint64_t* lp = sg.limbs.p + sg.row_limb[li - sg.li0];
						for (size_t x = eoff[li]; x < eoff[li + 1]; x++) {
							qpack::entry_view v;
							lp = qpack::read_entry(lp, ch.meta[x - ch.e0], v);
							ulong* nrow = nr.data() + nb * k;
							ulong* drow = dr.data() + nb * k;
							if (tree) {
								fmpz_set_ui_array(nz.data(), (const ulong*)v.num, (slong)v.nl);
								fmpz_set_ui_array(dz.data(), (const ulong*)v.den, (slong)v.dl);
								fmpz_multi_mod_ui(nrow, nz.data(), ucomb, temp);
								fmpz_multi_mod_ui(drow, dz.data(), ucomb, temp);
							}
							else
								for (size_t t = 0; t < k; t++) {
									nrow[t] = mpn_mod_1((mp_srcptr)v.num, (mp_size_t)v.nl, primes_used[t]);
									drow[t] = mpn_mod_1((mp_srcptr)v.den, (mp_size_t)v.dl, primes_used[t]);
								}
							if (v.neg)
								for (size_t t = 0; t < k; t++)
									if (nrow[t])
										nrow[t] = primes_used[t] - nrow[t];
							xs[nb++] = x;
							if (nb == MB)
								flush();
						}
					}
					if (nb > 0)
						flush();
					if (tree)
						fmpz_comb_temp_clear(temp);
					});
				pool.wait();
			}
			if (tree)
				fmpz_comb_clear(ucomb);
			ch.drop_packed();
			};

		// H(d E) * H(mat) * ncol < mod, with d the lcm of the denominators, on the packed values; an
		// entry is multiplied out only if its size bound can reach the largest product so far
		auto check_height_condition = [&]() -> bool {
			struct blk { size_t c, s, li0, li1; };
			std::vector<blk> blocks;
			for (size_t c = 0; c < chunks.size(); c++)
				for (size_t si = 0; si < chunks[c].segs.size(); si++) {
					const auto& sg = chunks[c].segs[si];
					auto b = row_blocks(sg.li0, sg.li1, std::max<size_t>(1, (eoff[sg.li1] - eoff[sg.li0]) >> 16));
					for (size_t i = 0; i + 1 < b.size(); i++)
						blocks.push_back({ c, si, b[i], b[i + 1] });
				}
			auto bitlen = [](const uint64_t* x, size_t n) -> size_t {
				while (n > 0 && x[n - 1] == 0) n--;
				return n == 0 ? 0 : (n - 1) * 64 + (64 - (size_t)std::countl_zero(x[n - 1]));
				};
			auto same = [](const qpack::entry_view& a, const qpack::entry_view& b) {
				return a.dl == b.dl && std::memcmp(a.den, b.den, a.dl * 8) == 0;
				};
			std::unique_ptr<__mpz_struct[]> part(new __mpz_struct[nthreads]);
			for (size_t i = 0; i < nthreads; i++)
				mpz_init_set_ui(&part[i], 1);
			pool.detach_loop<size_t>(0, blocks.size(), [&](size_t bi) {
				const auto& B = blocks[bi];
				const auto& ch = chunks[B.c];
				const auto& sg = ch.segs[B.s];
				mpz_ptr lcm = &part[SparseRREF::thread_id()];
				qpack::entry_view prev;
				bool have_prev = false;
				for (size_t li = B.li0; li < B.li1; li++) {
					const uint64_t* lp = sg.limbs.p + sg.row_limb[li - sg.li0];
					for (size_t x = eoff[li]; x < eoff[li + 1]; x++) {
						qpack::entry_view v;
						lp = qpack::read_entry(lp, ch.meta[x - ch.e0], v);
						if ((v.dl == 1 && v.den[0] == 1) || (have_prev && same(v, prev)))
							continue;
						prev = v;
						have_prev = true;
						mpz_t dv;
						mpz_roinit_n(dv, (mp_srcptr)v.den, (mp_size_t)v.dl);
						if (!mpz_divisible_p(lcm, dv))
							mpz_lcm(lcm, lcm, dv);
					}
				}
				});
			pool.wait();
			mpz_t d;
			mpz_init_set_ui(d, 1);
			for (size_t i = 0; i < nthreads; i++) {
				mpz_lcm(d, d, &part[i]);
				mpz_set_ui(&part[i], 1); // reused for the heights
			}
			const size_t bd = mpz_sizeinbase(d, 2);
			pool.detach_loop<size_t>(0, blocks.size(), [&](size_t bi) {
				const auto& B = blocks[bi];
				const auto& ch = chunks[B.c];
				const auto& sg = ch.segs[B.s];
				mpz_ptr h = &part[SparseRREF::thread_id()];
				size_t hbits = mpz_sizeinbase(h, 2);
				mpz_t q, val;
				mpz_init(q);
				mpz_init(val);
				qpack::entry_view qden; // the denominator q = d / den was computed for
				bool have_q = false;
				for (size_t li = B.li0; li < B.li1; li++) {
					const uint64_t* lp = sg.limbs.p + sg.row_limb[li - sg.li0];
					for (size_t x = eoff[li]; x < eoff[li + 1]; x++) {
						qpack::entry_view v;
						lp = qpack::read_entry(lp, ch.meta[x - ch.e0], v);
						const size_t bn = bitlen(v.num, v.nl), bden = bitlen(v.den, v.dl);
						if (bn + bd + 1 < hbits + bden) // |n| d / den < 2^(bn + bd - bden + 1) <= 2^(hbits - 1) <= h
							continue;
						if (!have_q || !same(v, qden)) {
							mpz_t dv;
							mpz_roinit_n(dv, (mp_srcptr)v.den, (mp_size_t)v.dl);
							mpz_divexact(q, d, dv);
							qden = v;
							have_q = true;
						}
						mpz_t nv;
						mpz_roinit_n(nv, (mp_srcptr)v.num, (mp_size_t)v.nl);
						mpz_mul(val, q, nv);
						if (mpz_cmp(val, h) > 0) {
							mpz_set(h, val);
							hbits = mpz_sizeinbase(h, 2);
						}
					}
				}
				mpz_clear(q);
				mpz_clear(val);
				});
			pool.wait();
			int_t hh = 1;
			for (size_t i = 0; i < nthreads; i++) {
				int_t t;
				fmpz_set_ui_array(t.data(), (const ulong*)mpz_limbs_read(&part[i]), (slong)mpz_size(&part[i]));
				if (t > hh)
					hh = t;
				mpz_clear(&part[i]);
			}
			mpz_clear(d);
			return m_height * out.ncol * hh < mod;
			};

		// the first prime
		SPR_T0(spr_rr);
		store_residues();
		release_matul();
#ifdef SPARSERREF_HAVE_RR_CORE
		frr.init(primes_used.data(), 1, mod);
#endif
		for (auto& ch : chunks) {
			if (opt->abort)
				break;
			attempt_chunk_cpu(ch, 1, nullptr);
		}
		bool isok = all_packed();
		SPR_ADD(rr_first, spr_rr);
		SPR_MEM("rr first");

		SPR_T0(spr_h0);
		if (isok)
			isok = check_height_condition();
		SPR_ADD(height, spr_h0);

		auto verbose = opt->verbose;

		if (verbose) {
			progress_end(opt);
		}

		if (opt->abort)
			return false;

		// set rows not in pivots to zero
		if (!isok) {
			std::vector<index_t> rowset(mat.nrow, sv);
			for (auto p : pivots)
				for (auto [r, c] : p)
					rowset[r] = c;
			for (size_t i = 0; i < mat.nrow; i++)
				if (rowset[i] == sv) {
					mat[i].clear();
				}
		}

		while (!isok && !opt->abort) {
			prime = n_nextprime(prime, 0);
			auto ce = clocknow();

			if (verbose) {
				progress_message(opt, ">> Reconstruct failed, now mod ~ 2^%llu, used time: %g s\n",
					(unsigned long long)mod.bits(), usedtime(cs, ce));
			}
			cs = ce;
			F = field_t(FIELD_Fp, prime);
			SPR_DO(SparseRREF::prof::d().primes++);
			SPR_T0(spr_red2);
			pool.detach_loop(0, mat.nrow, [&](auto i) {
				matul[i] = mat[i] % F.mod;
				});
			pool.wait();
			SPR_ADD(reduce, spr_red2);
			if (opt->abort)
				break;
			SPR_T0(spr_rp);
			// a replay on the GPU hands its rows over here: their residues go by entry into cur and their
			// columns are compared with the result's (the rows of level 0 stay in matul)
			cur_valid = false;
			std::atomic<bool> same_support = true;
			std::vector<char> got;
			bool delivered = false;
#ifdef SPARSERREF_USE_CUDA
			replay_sink_t sink = [&](const uint32_t* rows, uint32_t n, const uint64_t* o, const uint32_t* col,
				const uint64_t* val) -> bool {
				pool.detach_loop<size_t>(0, n, [&](size_t i) {
					const size_t li = li_of_row[rows[i]];
					const size_t cnt = o[i + 1] - o[i];
					if (li == SIZE_MAX) {
						if (cnt > 0)
							same_support = false;
						return;
					}
					got[li] = 1;
					bool ok = cnt == eoff[li + 1] - eoff[li];
					for (size_t j = 0; ok && j < cnt; j++)
						ok = col[o[i] + j] == (uint32_t)out.col[eoff[li] + j];
					if (!ok) {
						same_support = false;
						return;
					}
					std::memcpy(cur.get() + eoff[li], val + o[i], cnt * 8);
					});
				pool.wait();
				return true;
				};
			if (opt->policy.gpu.enabled) {
				if (!cur)
					cur.reset(new ulong[nent > 0 ? nent : 1]);
				got.assign(leftrows.size(), 0);
				sparse_mat_direct_rref(matul, pivots, F, opt, true, &sink, &delivered);
			}
			else
#endif
				sparse_mat_direct_rref(matul, pivots, F, opt);
			SPR_ADD(replay, spr_rp);
			// the direct rref bails out early when aborting, which can leave matul
			// structurally different from the first result, so stop before using it
			if (opt->abort)
				break;
			SPR_T0(spr_crt);
			// a prime that changes the support of the result is not usable: skip it
			if (delivered) {
				// the rows not handed over are in matul
				pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t li) {
					if (got[li])
						return;
					const auto& a = matul[leftrows[li]];
					bool ok = a.nnz() == eoff[li + 1] - eoff[li];
					for (size_t j = 0; ok && j < a.nnz(); j++)
						ok = a(j) == out.col[eoff[li] + j];
					if (!ok)
						same_support = false;
					else
						std::copy(a.entries, a.entries + a.nnz(), cur.get() + eoff[li]);
					});
				pool.wait();
				cur_valid = true;
			}
			else {
				pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t li) {
					const auto& a = matul[leftrows[li]];
					bool ok = a.nnz() == eoff[li + 1] - eoff[li];
					for (size_t j = 0; ok && j < a.nnz(); j++)
						ok = a(j) == out.col[eoff[li] + j];
					if (!ok)
						same_support = false;
					});
				pool.wait();
			}
			if (!same_support) {
				if (verbose)
					progress_message(opt, ">> prime %llu changes the support of the result, skipped\n",
						(unsigned long long)prime);
				SPR_ADD(crt, spr_crt);
				continue;
			}
			SPR_DO(SparseRREF::prof::d().t_check += SparseRREF::prof::since(spr_crt));
			SPR_T0(spr_st);
			// packed chunks check their values against the new residues; one that disagrees goes back
			// to residues; then every chunk with residues stores the new prime's
			{
				SPR_T0(spr_v);
				const auto bad = verify_packed(prime);
				SPR_ADD(t_verify, spr_v);
				for (size_t c = 0; c < chunks.size(); c++) {
					auto& ch = chunks[c];
					if (!ch.packed())
						continue;
					SPR_DO(SparseRREF::prof::d().n_verify++);
					if (!bad[c])
						continue;
					if (!ch.resid.empty())
						ch.drop_packed(); // the residues were kept
					else {
						SPR_T0(spr_u);
						unpack_chunk(ch);
						SPR_ADD(t_unpack, spr_u);
						SPR_DO(SparseRREF::prof::d().n_unpack++);
					}
				}
			}
			store_residues();
			release_matul();
			primes_used.push_back(prime);
			SPR_ADD(t_store, spr_st);
			SPR_MEM("prime stored");
			mod = mod * prime;
			const slong k = (slong)primes_used.size();

			if (any_unpacked()) {
				SPR_T0(spr_cb);
				fmpz_comb_t comb;
				fmpz_comb_init(comb, primes_used.data(), k);
#ifdef SPARSERREF_HAVE_RR_CORE
				frr.init(primes_used.data(), (size_t)k, mod);
#endif
				SPR_ADD(t_comb, spr_cb);
				SPR_T0(spr_sm);
				// candidate checks, cheapest first: the entries that failed last time (they usually fail
				// again until the very end), then a sample of about 4096 entries, then all entries
				bool sample_ok = true;
				{
					fmpz_comb_temp_t temp;
					fmpz_comb_temp_init(temp, comb);
					std::vector<ulong> rk(k);
					int_t z;
					rat_t qtmp;
					for (size_t h = 0; h < hard.size() && sample_ok; h++) {
						const auto& ch = chunks[chunk_of(hard[h])];
						if (!ch.packed())
							sample_ok = rec_ok(ch, hard[h], k, comb, temp, rk.data(), z, qtmp);
					}
					fmpz_comb_temp_clear(temp);
				}
				if (sample_ok) {
					// a stride over the entries of the chunks that keep residues
					std::vector<size_t> rc, rcum{ 0 };
					for (size_t c = 0; c < chunks.size(); c++)
						if (!chunks[c].packed()) {
							rc.push_back(c);
							rcum.push_back(rcum.back() + chunks[c].entries());
						}
					const size_t nres = rcum.back();
					const size_t stride = std::max<size_t>(1, nres / 4096);
					const size_t nsample = (nres + stride - 1) / stride;
					std::atomic<bool> failed = false;
					pool.detach_blocks<size_t>(0, nsample, [&](const size_t s, const size_t e) {
						fmpz_comb_temp_t temp;
						fmpz_comb_temp_init(temp, comb);
						std::vector<ulong> rk(k);
						int_t z;
						rat_t qtmp;
						for (size_t i = s; i < e && !failed.load(std::memory_order_relaxed); i++) {
							const size_t at = i * stride;
							const size_t r = (size_t)(std::upper_bound(rcum.begin(), rcum.end(), at) - rcum.begin()) - 1;
							const auto& ch = chunks[rc[r]];
							const size_t x = ch.e0 + (at - rcum[r]);
							if (!rec_ok(ch, x, k, comb, temp, rk.data(), z, qtmp)) {
								failed = true;
								note_hard(x);
							}
						}
						fmpz_comb_temp_clear(temp);
						});
					pool.wait();
					sample_ok = !failed;
				}
				SPR_ADD(t_sample, spr_sm);
				SPR_T0(spr_fu);
				if (sample_ok) {
					SPR_DO(SparseRREF::prof::d().n_full++);
					bool gpu_done = false;
#ifdef SPARSERREF_USE_CUDA
					gpu_done = reconstruct_gpu_chunks(chunks, eoff, primes_used, mod, nent, opt,
						[&](const qpack::chunk& ch, size_t x, fmpz_comb_temp_t temp, ulong* rk, int_t& z, rat_t& q) {
							return rec_entry(ch, x, k, comb, temp, rk, z, q);
						}, comb, note_hard, keep_resid(k));
#endif
					if (!gpu_done)
						for (auto& ch : chunks) {
							if (opt->abort)
								break;
							if (!ch.packed())
								attempt_chunk_cpu(ch, k, comb);
						}
				}
				SPR_ADD(t_full, spr_fu);
				SPR_MEM("crt+rr");
				fmpz_comb_clear(comb);
			}
			SPR_ADD(crt, spr_crt);
			if (opt->abort)
				break;

			// if ok, check height condition
			isok = all_packed();
			SPR_T0(spr_h);
			if (isok)
				isok = check_height_condition();
			SPR_ADD(height, spr_h);
		}
		opt->verbose = verbose;

		if (opt->abort)
			return false;

		if (opt->verbose) {
			progress_message(opt, "** Reconstruct success! Using mod ~ 2^%llu.                \n",
				(unsigned long long)mod.bits());
		}

#ifdef SPARSERREF_PROFILE
		{
			auto& pd = SparseRREF::prof::d();
			for (auto& ch : chunks)
				for (auto& sg : ch.segs)
					ch.for_entries(sg, eoff, [&](size_t, size_t j, size_t, const qpack::entry_view& v) {
						pd.res_ent++;
						pd.res_limbs += v.nl + v.dl;
						pd.res_hist[std::min<size_t>(v.nl, 3)][std::min<size_t>(v.dl, 3)]++;
						});
		}
#endif
		SPR_MEM("reconstructed");
		return true;
	}

	template <typename index_t>
	std::vector<std::vector<pivot_t<index_t>>> sparse_mat_rref_reconstruct(
		sparse_mat<rat_t, index_t>& mat, rref_option_t opt, const bool checkrank = false) {
		std::vector<std::vector<pivot_t<index_t>>> pivots;
		sparse_qmat_packed<index_t> packed;
		if (sparse_mat_rref_reconstruct_packed(mat, opt, pivots, packed, checkrank))
			packed.materialize(mat, opt->pool);
		return pivots;
	}

	// The kernel is read off the RREF, not off the original matrix: the pivot
	// submatrix must be fully diagonal
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_rref_kernel(const sparse_mat<T, index_t>& M,
		const std::vector<pivot_t<index_t>>& pivots, const field_t& F, rref_option_t opt) {

		auto rank = pivots.size();

		if (rank == M.ncol)
			return sparse_mat<T, index_t>();

		// just a warning, the kernel is not correct if the backward substitution was skipped
		if (!opt->is_back_sub) {
			std::cerr << "Warning: sparse_mat_rref_kernel: the backward substitution was skipped, so "
				<< "the pivot submatrix is only upper triangular and the result is not a kernel of "
				<< "the input" << std::endl;
		}

		constexpr index_t sv = index_sval<index_t>();
		T m1 = scalar_neg((T)1, F);

		sparse_mat<T, index_t> K(M.ncol, M.ncol - rank);

		auto nonpivs = perm_init((index_t)M.ncol);

		for (auto [r, c] : pivots) {
			K[c] = M[r];
			*K[c].find(c) = (T)0;
			K[c].canonicalize();
			nonpivs[c] = sv;
		}
		std::erase_if(nonpivs, [](index_t i) { return i == sv; });
		std::vector<index_t> nonpivs_ord(M.ncol, sv);
		for (size_t i = 0; i < nonpivs.size(); i++)
			nonpivs_ord[nonpivs[i]] = (index_t)i;

		for (size_t i = 0; i < nonpivs.size(); i++) {
			K[nonpivs[i]].push_back((index_t)i, m1);
		}

		for (auto [r, c] : pivots) {
			for (size_t j = 0; j < K[c].nnz(); j++)
				K[c](j) = nonpivs_ord[K[c](j)];
		}

		return K;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_rref_kernel(const sparse_mat<T, index_t>& M,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt) {
		std::vector<pivot_t<index_t>> n_pivots;
		for (auto& p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		return sparse_mat_rref_kernel(M, n_pivots, F, opt);
	}

	// error code:
	// 0 - sucess
	// 1 - matrix is not square
	// 2 - matrix is not invertible
	// 3 - Field is not supported
	// 4 - aborted (opt->abort was set; the matrix is only partially reduced)
	template <typename T, typename index_t>
	int sparse_mat_inverse(sparse_mat<T, index_t>& M1, const sparse_mat<T, index_t>& M,
		const field_t& F, rref_option_t opt) noexcept {
		if (M.nrow != M.ncol) {
			std::cerr << "Error: sparse_mat_inverse: matrix is not square" << std::endl;
			return 1;
		}

		if (M.nnz() == 0) {
			std::cerr << "Error: sparse_mat_inverse: zero matrix" << std::endl;
			return 2;
		}

		auto& pool = opt->pool;

		// define the Augmented matrix
		M1 = M;
		M1.compress();
		for (size_t i = 0; i < M1.nrow; i++) {
			if (M1[i].nnz() == 0) {
				std::cerr << "Error: sparse_mat_inverse: matrix is not invertible" << std::endl;
				return 2;
			}
			M1[i].push_back(i + M.ncol, (T)1);
		}
		M1.ncol *= 2;

		// backup the option
		auto old_col_weight = opt->col_weight;
		std::function<int64_t(int64_t)> new_col_weight = [&](int64_t i) {
			return ((i < M.ncol) ? old_col_weight(i) : -1);
			};
		opt->col_weight = new_col_weight;
		bool is_back_sub = opt->is_back_sub;
		opt->is_back_sub = true;

		std::vector<std::vector<pivot_t<index_t>>> pivots;
		if (F.ring == RING::FIELD_Fp)
			pivots = sparse_mat_rref(M1, F, opt);
		else if (F.ring == RING::FIELD_QQ)
			if constexpr (std::is_same_v<T, rat_t>) {
				pivots = sparse_mat_rref_reconstruct(M1, opt, true);
			}
			else {
				// type is not rat_t when field is QQ
				std::cerr << "Warning: sparse_mat_inverse: type is not rat_t when field is QQ" << std::endl;
				pivots = sparse_mat_rref(M1, F, opt);
			}
		else {
			std::cerr << "Error: sparse_mat_inverse: field not supported" << std::endl;
			// restore the option
			opt->col_weight = old_col_weight;
			opt->is_back_sub = is_back_sub;
			return 3;
		}

		if (opt->abort) {
			// the elimination or the back substitution stopped early: the pivot submatrix is
			// not diagonal, so the extraction below would read garbage
			opt->col_weight = old_col_weight;
			opt->is_back_sub = is_back_sub;
			return 4;
		}

		std::vector<pivot_t<index_t>> flatten_pivots;
		size_t rank = 0;
		for (auto& p : pivots) {
			rank += p.size();
			flatten_pivots.insert(flatten_pivots.end(), p.begin(), p.end());
		}

		if (rank != M.nrow) {
			std::cerr << "Error: sparse_mat_inverse: matrix is not invertible" << std::endl;
			// restore the option
			opt->col_weight = old_col_weight;
			opt->is_back_sub = is_back_sub;
			return 2;
		}

		auto perm = perm_init(M.nrow);
		std::sort(perm.begin(), perm.end(),
			[&](size_t a, size_t b) {
				return flatten_pivots[a].c < flatten_pivots[b].c;
			});
		for (size_t i = 0; i < M.nrow; i++) {
			perm[i] = flatten_pivots[perm[i]].r;
		}

		permute(perm, M1.rows);

		for (size_t i = 0; i < M1.nrow; i++) {
			// the first ncol columns is the identity matrix,
			// we need to remove it
			for (size_t j = 0; j < M1[i].nnz() - 1; j++) {
				M1[i][j] = M1[i][j + 1];
				M1[i](j) = M1[i](j + 1) - M.ncol;
			}
			M1[i].resize(M1[i].nnz() - 1);
		}
		M1.ncol = M.ncol;

		// restore the option
		opt->col_weight = old_col_weight;
		opt->is_back_sub = is_back_sub;

		return 0;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_inverse(const sparse_mat<T, index_t>& M,
		const field_t& F, rref_option_t opt) {
		sparse_mat<T, index_t> res;
		int error = sparse_mat_inverse(res, M, F, opt);
		if (error != 0) 
			return sparse_mat<T, index_t>();
		return res;
	}

	// IO
	template <typename T, typename index_t, typename S>
	sparse_mat<T, index_t> sparse_mat_read(S& st, const field_t& F) {
		if (!st.is_open()) {
			std::cerr << "Error: sparse_mat_read: file not open." << std::endl;
			return sparse_mat<T, index_t>();
		}

		std::string line;
		std::vector<size_t> dims;
		sparse_mat<T, index_t> mat;
		size_t declared_nnz = 0, entries_read = 0;
		bool has_declared_nnz = false;

		while (std::getline(st, line)) {
			if (line.empty() || line[0] == '%')
				continue;

			size_t start = 0;
			size_t end = line.find(' ');
			while (end != std::string::npos) {
				if (start != end) {
					uint64_t dim = 0;
					if (!string_to_ull(line.substr(start, end - start), dim)) {
						std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
						return sparse_mat<T, index_t>();
					}
					dims.push_back(dim);
				}
				start = end + 1;
				end = line.find(' ', start);
			}
			if (start < line.size()) {
				// the third number of a coordinate header is the entry count; an SMS header carries a
				// type letter there instead, and that one is not a number and is not checked
				uint64_t declared = 0;
				has_declared_nnz = string_to_ull(line.substr(start), declared);
				declared_nnz = static_cast<size_t>(declared);
				if (dims.size() != 2) {
					std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
					return sparse_mat<T, index_t>();
				}
				mat = sparse_mat<T, index_t>(dims[0], dims[1]);
			}
			break;
		}

		while (std::getline(st, line)) {
			if (line.empty() || line[0] == '%')
				continue;

			bool is_end = false;
			size_t rowcol[2];
			size_t* rowcolptr = rowcol;
			size_t start = 0;
			size_t end = line.find(' ');
			size_t count = 0;

			while (end != std::string::npos && count < 2) {
				if (start != end) {
					uint64_t val = 0;
					if (!string_to_ull(line.substr(start, end - start), val)) {
						std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
						return sparse_mat<T, index_t>();
					}
					if (val == 0) {
						is_end = true;
						break;
					}
					*rowcolptr = val - 1;
					rowcolptr++;
					count++;
				}
				start = end + 1;
				end = line.find(' ', start);
			}

			if (is_end)
				break;

			if (count != 2) {
				std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
				return sparse_mat<T, index_t>();
			}

			// set_str returns a non-zero value for a literal that is not a number in this field,
			// which must be reported instead of being stored as an unvalidated value; the value
			// is normalized, since the arithmetic below assumes the canonical form
			T val{};
			int val_ok = 0;
			if constexpr (std::is_same_v<T, ulong>) {
				rat_t raw_val;
				val_ok = raw_val.set_str(line.substr(start), 10, true);
				val = raw_val % F.mod;
			}
			else if constexpr (std::is_same_v<T, rat_t>) {
				val_ok = val.set_str(line.substr(start), 10, true);
			}
			else if constexpr (std::is_same_v<T, int_t>) {
				val_ok = val.set_str(line.substr(start));
			}

			if (val_ok != 0) {
				std::cerr << "Error: sparse_mat_read: the value is not a number: "
					<< line.substr(start) << std::endl;
				return sparse_mat<T, index_t>();
			}

			// the indices come from the file: a malformed one must not index the rows of the matrix
			if (rowcol[0] >= mat.nrow || rowcol[1] >= mat.ncol) {
				report_error("sparse_mat_read: an index is out of range");
				return sparse_mat<T, index_t>();
			}

			entries_read++;
			mat[rowcol[0]].push_back(rowcol[1], val);
		}

		if (has_declared_nnz && entries_read != declared_nnz) {
			report_error("sparse_mat_read: the number of entries does not match the header");
			return sparse_mat<T, index_t>();
		}

		// a position listed more than once means one entry holding the sum of its values
		for (size_t i = 0; i < mat.nrow; i++)
			mat[i].merge_duplicates(F);

		return mat;
	}

	// appends "row col value\n" (1-based positions) in the format fmpz_get_str / fmpq_get_str use;
	// values that fit a word are printed without going through FLINT's string functions
	template <typename T>
	inline void sparse_write_entry(std::string& out, size_t row, size_t col, const T& val) {
		char num_buf[32];
		auto [ptr1, ec1] = std::to_chars(num_buf, num_buf + sizeof(num_buf), row + 1);
		out.append(num_buf, ptr1);
		out.push_back(' ');
		auto [ptr2, ec2] = std::to_chars(num_buf, num_buf + sizeof(num_buf), col + 1);
		out.append(num_buf, ptr2);
		out.push_back(' ');
		if constexpr (std::is_integral_v<T>) {
			auto [ptr3, ec3] = std::to_chars(num_buf, num_buf + sizeof(num_buf), val);
			out.append(num_buf, ptr3);
		}
		else if constexpr (std::is_same_v<T, rat_t>) {
			const fmpz n = *fmpq_numref(val.data()), d = *fmpq_denref(val.data());
			if (!COEFF_IS_MPZ(n) && !COEFF_IS_MPZ(d)) {
				auto [ptr3, ec3] = std::to_chars(num_buf, num_buf + sizeof(num_buf), (slong)n);
				out.append(num_buf, ptr3);
				if (d != 1) {
					out.push_back('/');
					auto [ptr4, ec4] = std::to_chars(num_buf, num_buf + sizeof(num_buf), (slong)d);
					out.append(num_buf, ptr4);
				}
			}
			else
				out += val.get_str();
		}
		else if constexpr (std::is_same_v<T, int_t>) {
			const fmpz n = *val.data();
			if (!COEFF_IS_MPZ(n)) {
				auto [ptr3, ec3] = std::to_chars(num_buf, num_buf + sizeof(num_buf), (slong)n);
				out.append(num_buf, ptr3);
			}
			else
				out += val.get_str();
		}
		out.push_back('\n');
	}

	// pool: optional; with more than one thread the rows are formatted in parallel blocks and
	// written in order (the same bytes as the serial path)
	template <typename T, typename S, typename index_t>
	void sparse_mat_write(sparse_mat<T, index_t>& mat, S& st, enum SPARSE_FILE_TYPE type, thread_pool* pool = nullptr) {
		const bool par = pool != nullptr && pool->get_thread_count() > 1 && mat.nrow > 1;
		switch (type) {
		case SPARSE_FILE_TYPE_PLAIN:
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		case SPARSE_FILE_TYPE_MTX: {
			// MatrixMarket has no field for rationals: such a matrix keeps the p/q values the library
			// reads back, and its header announces them as a "rational" field instead of claiming
			// integers; an integral matrix keeps the standard header
			if constexpr (std::is_same_v<T, rat_t>) {
				std::atomic<bool> integral = true;
				auto scan = [&](size_t r0, size_t r1) {
					for (size_t i = r0; i < r1 && integral.load(std::memory_order_relaxed); i++)
						for (size_t j = 0; j < mat[i].nnz(); j++)
							if (!mat[i].entries[j].is_integer()) {
								integral = false;
								break;
							}
					};
				if (par) {
					pool->detach_blocks<size_t>(0, mat.nrow, scan);
					pool->wait();
				}
				else
					scan(0, mat.nrow);
				st << (integral ? "%%MatrixMarket matrix coordinate integer general\n"
					: "%%MatrixMarket matrix coordinate rational general\n");
			}
			else if constexpr (std::is_same_v<T, ulong> || std::is_same_v<T, int_t>) {
				st << "%%MatrixMarket matrix coordinate integer general\n";
			}
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		}
		case SPARSE_FILE_TYPE_SMS: {
			char type_char =
				std::is_same_v<T, rat_t> ? 'Q' :
				(std::is_same_v<T, ulong> || std::is_same_v<T, int_t>) ? 'M' : '\0';
			if (type_char == '\0') {
				return;
			}
			st << mat.nrow << ' ' << mat.ncol << ' ' << type_char << '\n';
			break;
		}
		default:
			return;
		}

		auto format_rows = [&mat](size_t r0, size_t r1, std::string& out) {
			for (size_t i = r0; i < r1; ++i) {
				for (auto [ind, val] : mat[i]) {
					if (val == 0) continue;
					sparse_write_entry(out, i, ind, val);
				}
			}
			};

		if (!par) {
			// in blocks, so the text never needs more than one block of memory
			std::string line_buf;
			for (size_t r0 = 0; r0 < mat.nrow;) {
				size_t r1 = r0, n = 0;
				while (r1 < mat.nrow && (r1 == r0 || n < write_block_entries))
					n += mat[r1++].nnz();
				line_buf.clear();
				format_rows(r0, r1, line_buf);
				st.write(line_buf.data(), line_buf.size());
				r0 = r1;
			}
		}
		else {
			std::vector<size_t> bound{ 0 };
			size_t acc = 0;
			for (size_t i = 0; i < mat.nrow; i++) {
				acc += mat[i].nnz();
				if (acc >= write_block_entries && i + 1 < mat.nrow) {
					bound.push_back(i + 1);
					acc = 0;
				}
			}
			bound.push_back(mat.nrow);
			write_blocks_ordered(st, *pool, bound.size() - 1, [&](size_t b, std::string& out) {
				size_t n = 0;
				for (size_t i = bound[b]; i < bound[b + 1]; i++)
					n += mat[i].nnz();
				out.reserve(n * 32);
				format_rows(bound[b], bound[b + 1], out);
				});
		}

		if (type == SPARSE_FILE_TYPE_SMS) {
			st << "0 0 0\n";
		}
	}

} // namespace SparseRREF

#endif