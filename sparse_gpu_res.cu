/*
	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// A Z/p matrix kept on the GPU during an elimination whose pivots are known; see sparse_gpu.h.
//
// Rows live in an arena (global columns, values) addressed by per-row offset and length. An updated
// row is written to fresh space taken from the top of the arena (its old space becomes garbage);
// when the arena is full the live rows are compacted into a second arena and the rows that did not
// fit are updated again. The Schur kernel is the one of sparse_gpu.cu (one block per row, a dense
// accumulator over U = the columns of the pivot rows, pivots visited through a bit set in order),
// reading its rows from the arena; the per-level preparation (U, compact pivot rows) runs on the GPU.
#include "sparse_gpu.h"
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace SparseRREF::gpu {

	namespace {
		constexpr int BS = 256;
		using clk = std::chrono::steady_clock;
		double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

		// transfers go through two page-locked buffers of this many entries (whole rows; a longer row
		// grows its buffer), so the host never holds a page-locked copy of the matrix
		uint64_t piece_entries() {
			return std::max<uint64_t>(1, get_options().transfer_piece);
		}
		// staging buffer b (0, 1) for n entries: columns at the start, values after them
		bool staging(int b, uint64_t n, uint32_t*& col, uint64_t*& val) {
			const size_t cbytes = (size_t)(n * 4 + 7) / 8 * 8;
			char* h = (char*)host_buffer(2 + b, cbytes + n * 8 + 16);
			if (!h) return false;
			col = (uint32_t*)h;
			val = (uint64_t*)(h + cbytes);
			return true;
		}

		__device__ __forceinline__ uint64_t mulmod_shoup(uint64_t a, uint64_t b, uint64_t a_pr, uint64_t p) {
			uint64_t q = __umul64hi(a_pr, b);
			uint64_t r = a * b - q * p;
			return r >= p ? r - p : r;
		}
		__device__ uint64_t modinv(uint64_t a, uint64_t p) {
			int64_t t0 = 0, t1 = 1;
			uint64_t r0 = p, r1 = a % p;
			while (r1) {
				uint64_t q = r0 / r1, r2 = r0 - q * r1;
				r0 = r1; r1 = r2;
				int64_t t2 = t0 - (int64_t)q * t1;
				t0 = t1; t1 = t2;
			}
			return t0 < 0 ? (uint64_t)(t0 + (int64_t)p) : (uint64_t)t0;
		}

		size_t smem_bytes(uint32_t nU, uint32_t P) {
			size_t wU = (nU + 31) / 32, wP = (P + 31) / 32;
			return (size_t)nU * 8 + wU * 4 + wP * 4 + (wU + 1) * 4 + 16;
		}
		size_t smem_bytes_global(uint32_t nU, uint32_t P) {
			size_t wU = (nU + 31) / 32, wP = (P + 31) / 32;
			return wP * 4 + (wU + 1) * 4 + 16;
		}
		size_t slot_words(uint32_t nU) {
			size_t w = (size_t)nU + ((nU + 31) / 32 + 1) / 2;
			return (w + 31) / 32 * 32;
		}
		constexpr size_t scratch_budget = (size_t)3 << 30;

		// divide each pivot row by its pivot entry (one warp per row)
		__global__ void rescale_kernel(uint32_t P, const uint32_t* __restrict__ prow, const uint32_t* __restrict__ pcol,
			const uint64_t* __restrict__ row_off, const uint32_t* __restrict__ row_len,
			const uint32_t* __restrict__ col, uint64_t* __restrict__ val, uint64_t p) {
			const size_t gt = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
			const uint32_t w = (uint32_t)(gt >> 5), lane = threadIdx.x & 31;
			if (w >= P) return;
			const uint32_t r = prow[w], c = pcol[w];
			const uint64_t off = row_off[r];
			const uint32_t len = row_len[r];
			uint64_t inv = 0, inv_pr = 0;
			if (lane == 0) {
				uint32_t lo = 0, hi = len;
				while (lo < hi) {
					uint32_t mid = (lo + hi) >> 1;
					if (col[off + mid] < c) lo = mid + 1; else hi = mid;
				}
				const uint64_t v = (lo < len && col[off + lo] == c) ? val[off + lo] : 0;
				inv = v ? modinv(v, p) : 0;
				inv_pr = (uint64_t)(((unsigned __int128)inv << 64) / p);
			}
			inv = __shfl_sync(0xffffffffu, inv, 0);
			inv_pr = __shfl_sync(0xffffffffu, inv_pr, 0);
			for (uint32_t k = lane; k < len; k += 32)
				val[off + k] = mulmod_shoup(inv, val[off + k], inv_pr, p);
		}

		// U = the columns of the pivot rows: mark, then (after a scan) number them
		__global__ void mark_kernel(uint32_t P, const uint32_t* __restrict__ prow, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, uint32_t* __restrict__ inU) {
			for (uint32_t j = blockIdx.x; j < P; j += gridDim.x) {
				const uint32_t r = prow[j];
				const uint64_t off = row_off[r];
				for (uint32_t k = threadIdx.x; k < row_len[r]; k += blockDim.x)
					inU[col[off + k]] = 1;
			}
		}
		__global__ void number_kernel(uint32_t ncol, const uint32_t* __restrict__ inU, const uint32_t* __restrict__ pos,
			int32_t* __restrict__ cid_of_col, uint32_t* __restrict__ col_of_cid) {
			const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
			if (c >= ncol) return;
			if (inU[c]) {
				cid_of_col[c] = (int32_t)pos[c];
				col_of_cid[pos[c]] = c;
			}
			else
				cid_of_col[c] = -1;
		}
		__global__ void pivinfo_kernel(uint32_t P, const uint32_t* __restrict__ prow, const uint32_t* __restrict__ pcol,
			const uint32_t* __restrict__ row_len, const int32_t* __restrict__ cid_of_col,
			uint32_t* __restrict__ piv_cid, int32_t* __restrict__ posU, uint64_t* __restrict__ plen,
			unsigned int* __restrict__ err) {
			const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
			if (j >= P) return;
			const int32_t cid = cid_of_col[pcol[j]];
			plen[j] = row_len[prow[j]];
			if (cid < 0) { // the pivot row does not hold its pivot column (not uploaded?): refuse
				atomicOr(err, 1u);
				piv_cid[j] = 0;
				return;
			}
			piv_cid[j] = (uint32_t)cid;
			posU[cid] = (int32_t)j;
		}
		__global__ void pvfill_kernel(uint32_t P, const uint32_t* __restrict__ prow, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, const uint64_t* __restrict__ val,
			const int32_t* __restrict__ cid_of_col, const uint64_t* __restrict__ pv_off,
			uint32_t* __restrict__ pv_cid, uint64_t* __restrict__ pv_val) {
			for (uint32_t j = blockIdx.x; j < P; j += gridDim.x) {
				const uint32_t r = prow[j];
				const uint64_t off = row_off[r], o = pv_off[j];
				for (uint32_t k = threadIdx.x; k < row_len[r]; k += blockDim.x) {
					pv_cid[o + k] = (uint32_t)cid_of_col[col[off + k]];
					pv_val[o + k] = val[off + k];
				}
			}
		}

		// The Schur update of the rows rows[0..R) of the arena (see schur_kernel in sparse_gpu.cu). A
		// row that holds no pivot column is left as it is; an updated row is written at the top of
		// the arena; a row that does not fit is left as it is with status 1.
		template <bool GLOBAL>
		__global__ void __launch_bounds__(BS) res_kernel(
			uint64_t* __restrict__ scratch, size_t slot_words,
			uint64_t p, uint32_t nU, uint32_t P, uint32_t R,
			const uint64_t* __restrict__ pv_off, const uint32_t* __restrict__ pv_cid, const uint64_t* __restrict__ pv_val,
			const uint32_t* __restrict__ piv_cid, const int32_t* __restrict__ posU,
			const int32_t* __restrict__ cid_of_col, const uint32_t* __restrict__ col_of_cid,
			const uint32_t* __restrict__ rows, uint64_t* __restrict__ row_off, uint32_t* __restrict__ row_len,
			uint32_t* __restrict__ acol, uint64_t* __restrict__ aval, uint32_t* __restrict__ tp_idx,
			const int32_t* __restrict__ own_of_row, // nullptr, or the pivot position of a row that is itself a pivot row
			uint8_t* __restrict__ status, unsigned long long* __restrict__ top, unsigned long long cap) {

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
			__shared__ int s_j, s_over, s_any;
			__shared__ uint64_t s_e, s_epr;
			__shared__ unsigned long long s_base;
			__shared__ uint32_t s_wcnt[BS / 32];
			const int tid = threadIdx.x;
			const int lane = tid & 31, warp = tid >> 5;

			if constexpr (!GLOBAL) {
				for (uint32_t i = tid; i < nU; i += BS) acc[i] = 0;
				for (uint32_t i = tid; i < wU; i += BS) pres[i] = 0;
			}
			for (uint32_t i = tid; i < wP; i += BS) pbits[i] = 0;
			if (tid == 0) s_any = 0;
			__syncthreads();

			for (uint32_t ri = blockIdx.x; ri < R; ri += gridDim.x) {
				const uint32_t row = rows[ri];
				const uint64_t rb = row_off[row];
				const uint32_t rlen = row_len[row];
				const uint32_t* rcol = acol + rb;
				const uint64_t* rval = aval + rb;
				uint32_t* tpi = tp_idx + rb;
				uint32_t npass = 0;
				for (uint32_t b0 = 0; b0 < rlen; b0 += BS) {
					const uint32_t k = b0 + tid;
					bool pass = false;
					if (k < rlen) {
						const int32_t c = cid_of_col[rcol[k]];
						if (c >= 0) {
							acc[c] = rval[k];
							atomicOr(&pres[c >> 5], 1u << (c & 31));
							int pos = posU[c];
							if (pos >= 0) {
								atomicOr(&pbits[pos >> 5], 1u << (pos & 31));
								s_any = 1;
							}
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
						tpi[off + __popc(m & ((1u << lane) - 1u))] = k;
					npass += tot;
					__syncthreads();
				}
				if (own_of_row != nullptr) { // a pivot row is not reduced by its own pivot
					if (tid == 0) {
						const int o = own_of_row[row];
						if (o >= 0)
							pbits[o >> 5] &= ~(1u << (o & 31));
					}
					__syncthreads();
				}
				const bool any = s_any != 0;

				if (any) {
					uint32_t w = 0;
					while (true) {
						if (tid < 32) {
							int found = -1;
							for (uint32_t base = w; base < wP; base += 32) {
								uint32_t idx = base + tid;
								uint32_t word = idx < wP ? pbits[idx] : 0u;
								unsigned mm = __ballot_sync(0xffffffffu, word != 0);
								if (mm) {
									int ln = __ffs(mm) - 1;
									uint32_t wv = __shfl_sync(0xffffffffu, word, ln);
									found = (int)((base + ln) * 32 + (__ffs(wv) - 1));
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
						if (ev != 0) {
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
					const uint32_t cnt = wpref[wU] + npass;
					if (tid == 0) {
						unsigned long long b = atomicAdd(top, (unsigned long long)cnt);
						s_base = b;
						s_over = (b + cnt > cap) ? 1 : 0;
						status[ri] = s_over ? 1 : 2; // 2: written
					}
					__syncthreads();
					if (!s_over) {
						const unsigned long long base = s_base;
						for (uint32_t wi = tid; wi < wU; wi += BS) {
							uint32_t mm = pres[wi];
							uint32_t r = wpref[wi];
							while (mm) {
								int bit = __ffs(mm) - 1;
								mm &= mm - 1;
								uint32_t c = wi * 32 + bit;
								uint32_t gc = col_of_cid[c];
								uint32_t lo = 0, hi = npass;
								while (lo < hi) {
									uint32_t mid = (lo + hi) >> 1;
									if (rcol[tpi[mid]] < gc) lo = mid + 1; else hi = mid;
								}
								acol[base + r + lo] = gc;
								aval[base + r + lo] = acc[c];
								r++;
							}
						}
						for (uint32_t q = tid; q < npass; q += BS) {
							const uint32_t kk = tpi[q];
							const uint32_t gc = rcol[kk];
							uint32_t lo = 0, hi = nU;
							while (lo < hi) {
								uint32_t mid = (lo + hi) >> 1;
								if (col_of_cid[mid] < gc) lo = mid + 1; else hi = mid;
							}
							uint32_t below = wpref[lo >> 5];
							if (lo & 31)
								below += __popc(pres[lo >> 5] & ((1u << (lo & 31)) - 1u));
							acol[base + q + below] = gc;
							aval[base + q + below] = rval[kk];
						}
						__syncthreads();
						if (tid == 0) {
							row_off[row] = base;
							row_len[row] = cnt;
						}
					}
				}
				else if (tid == 0)
					status[ri] = 0;
				__syncthreads();
				// clear the accumulator (also for a row that holds no pivot column)
				for (uint32_t wi = tid; wi < wU; wi += BS) {
					uint32_t mm = pres[wi];
					while (mm) {
						int bit = __ffs(mm) - 1;
						mm &= mm - 1;
						acc[wi * 32 + bit] = 0;
					}
					pres[wi] = 0;
				}
				if (!any)
					for (uint32_t i = tid; i < wP; i += BS) pbits[i] = 0; // nothing was set, kept for safety
				if (tid == 0) s_any = 0;
				__syncthreads();
			}
		}

		// copy rows[0..R) to dst at the offsets new_off (a scan of their lengths); one block per row
		__global__ void gather_kernel(uint32_t R, const uint32_t* __restrict__ rows, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ scol, const uint64_t* __restrict__ sval,
			const uint64_t* __restrict__ new_off, uint32_t* __restrict__ dcol, uint64_t* __restrict__ dval) {
			for (uint32_t i = blockIdx.x; i < R; i += gridDim.x) {
				const uint32_t r = rows[i];
				const uint64_t so = row_off[r], d = new_off[i];
				const uint32_t n = row_len[r];
				for (uint32_t k = threadIdx.x; k < n; k += blockDim.x) {
					dcol[d + k] = scol[so + k];
					if (dval) dval[d + k] = sval[so + k];
				}
			}
		}
		__global__ void lens_kernel(uint32_t R, const uint32_t* __restrict__ rows, const uint32_t* __restrict__ row_len,
			uint64_t* __restrict__ lens) {
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i < R) lens[i] = row_len[rows[i]];
			if (i == R) lens[R] = 0;
		}
		__global__ void setoff_kernel(uint32_t R, const uint32_t* __restrict__ rows, const uint64_t* __restrict__ new_off,
			uint64_t* __restrict__ row_off) {
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i < R) row_off[rows[i]] = new_off[i];
		}

		__global__ void setpos_kernel(uint32_t P, const uint32_t* __restrict__ keys, int32_t* __restrict__ map, int clear) {
			const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
			if (j < P) map[keys[j]] = clear ? -1 : (int32_t)j;
		}
		// count / list, for each pivot column, the live rows other than its pivot row that hold it
		__global__ void tcount_kernel(uint32_t R, const uint32_t* __restrict__ rows, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, const int32_t* __restrict__ colpiv,
			const uint32_t* __restrict__ prow, unsigned long long* __restrict__ cnt) {
			for (uint32_t i = blockIdx.x; i < R; i += gridDim.x) {
				const uint32_t r = rows[i];
				const uint64_t off = row_off[r];
				for (uint32_t k = threadIdx.x; k < row_len[r]; k += blockDim.x) {
					const int32_t j = colpiv[col[off + k]];
					if (j >= 0 && prow[j] != r)
						atomicAdd(&cnt[j], 1ULL);
				}
			}
		}
		__global__ void tfill_kernel(uint32_t R, const uint32_t* __restrict__ rows, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, const int32_t* __restrict__ colpiv,
			const uint32_t* __restrict__ prow, unsigned long long* __restrict__ cursor, uint32_t* __restrict__ trows) {
			for (uint32_t i = blockIdx.x; i < R; i += gridDim.x) {
				const uint32_t r = rows[i];
				const uint64_t off = row_off[r];
				for (uint32_t k = threadIdx.x; k < row_len[r]; k += blockDim.x) {
					const int32_t j = colpiv[col[off + k]];
					if (j >= 0 && prow[j] != r)
						trows[atomicAdd(&cursor[j], 1ULL)] = r;
				}
			}
		}

		// ---- the pivot search of a round on the GPU (pivots_search, method 0) ----
		constexpr uint32_t NONE = 0xffffffffu;

		// per row of leftrows: column counts, each column's best row (fewest entries, then smallest index,
		// packed so that atomicMin finds it), whether the row is empty, and the entries in all
		__global__ void scount_kernel(uint32_t R, const uint32_t* __restrict__ lrows, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, unsigned long long* __restrict__ cnt,
			unsigned long long* __restrict__ key, uint8_t* __restrict__ empty, unsigned long long* __restrict__ nnz) {
			for (uint32_t i = blockIdx.x; i < R; i += gridDim.x) {
				const uint32_t r = lrows[i];
				const uint64_t off = row_off[r];
				const uint32_t len = row_len[r];
				if (threadIdx.x == 0) {
					empty[i] = len == 0;
					if (len) atomicAdd(nnz, (unsigned long long)len);
				}
				const unsigned long long kv = ((unsigned long long)len << 32) | r;
				for (uint32_t k = threadIdx.x; k < len; k += blockDim.x) {
					const uint32_t c = col[off + k];
					atomicAdd(&cnt[c], 1ULL);
					atomicMin(&key[c], kv);
				}
			}
		}
		// the rows of each column (any order), from the offsets in cur
		__global__ void sfill_kernel(uint32_t R, const uint32_t* __restrict__ lrows, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, unsigned long long* __restrict__ cur,
			uint32_t* __restrict__ crow) {
			for (uint32_t i = blockIdx.x; i < R; i += gridDim.x) {
				const uint32_t r = lrows[i];
				const uint64_t off = row_off[r];
				for (uint32_t k = threadIdx.x; k < row_len[r]; k += blockDim.x)
					crow[atomicAdd(&cur[col[off + k]], 1ULL)] = r;
			}
		}
		// the row each column of leftcols would take
		__global__ void slrow_kernel(uint32_t L, const uint32_t* __restrict__ lcols, const unsigned long long* __restrict__ cnt,
			const unsigned long long* __restrict__ key, const int64_t* __restrict__ w, uint32_t* __restrict__ lrow) {
			const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
			if (i >= L) return;
			const uint32_t c = lcols[i];
			lrow[i] = (cnt[c] > 0 && w[c] >= 0) ? (uint32_t)(key[c] & 0xffffffffULL) : NONE;
		}
		// the column each row of leftrows would take: fewest rows, then least weight, then smallest index,
		// among its columns of weight >= 0 (one warp per row)
		__global__ void srcol_kernel(uint32_t R, const uint32_t* __restrict__ lrows, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, const unsigned long long* __restrict__ cnt,
			const int64_t* __restrict__ w, uint32_t* __restrict__ rcol) {
			const uint32_t lane = threadIdx.x & 31;
			const uint64_t warps = (uint64_t)gridDim.x * (blockDim.x >> 5);
			for (uint64_t i = (uint64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); i < R; i += warps) {
				const uint32_t r = lrows[i];
				const uint64_t off = row_off[r];
				const uint32_t len = row_len[r];
				unsigned long long bc = ~0ULL;
				int64_t bw = 0;
				uint32_t bcol = NONE;
				for (uint32_t k = lane; k < len; k += 32) {
					const uint32_t c = col[off + k];
					const int64_t wc = w[c];
					if (wc < 0) continue;
					const unsigned long long n = cnt[c];
					if (n < bc || (n == bc && (wc < bw || (wc == bw && c < bcol)))) { bc = n; bw = wc; bcol = c; }
				}
				for (int o = 16; o > 0; o >>= 1) {
					const unsigned long long n = __shfl_down_sync(0xffffffffu, bc, o);
					const int64_t wc = __shfl_down_sync(0xffffffffu, bw, o);
					const uint32_t c = __shfl_down_sync(0xffffffffu, bcol, o);
					if (n < bc || (n == bc && (wc < bw || (wc == bw && c < bcol)))) { bc = n; bw = wc; bcol = c; }
				}
				if (lane == 0) rcol[i] = bc == ~0ULL ? NONE : bcol;
			}
		}

		// The sequential part of a look, in one block of 1024 threads, over items in order, each with a
		// candidate (NONE: none). The sequential rule: items in order, each taken unless an earlier taken
		// item blocks it; taking item i blocks the keys in its candidate's list: the columns of its
		// candidate row (mode 0, left look: item = column, key = the column) or the rows of its candidate
		// column (mode 1, right look: item = row, key = the row; the rows of the columns already taken,
		// pre[0..npre), are blocked from the start). An item's key is always in its own list.
		// Here a batch of 1024 items is decided in steps. mark[x] holds the smallest position (plus one;
		// 0 for pre) of a taken item blocking key x, so a mark blocks only the items after it. In a step,
		// the items not yet decided and not blocked stamp the keys of their lists with their position (the
		// earliest wins, owner); an item whose key carries its own stamp cannot be blocked by any earlier
		// undecided item, so it is taken, exactly as in the sequential order (the earliest undecided item
		// always is). The taken items mark, the rest are tested again. Taken items are written in order.
		template <int MODE>
		__device__ __forceinline__ void slist(uint32_t cd, const uint64_t* __restrict__ row_off, const uint32_t* __restrict__ row_len,
			const unsigned long long* __restrict__ coff, uint64_t& b, uint32_t& n) {
			if (MODE == 0) { b = row_off[cd]; n = row_len[cd]; }
			else { b = coff[cd]; n = (uint32_t)(coff[cd + 1] - coff[cd]); }
		}
		template <int MODE>
		__global__ void __launch_bounds__(1024) sgreedy_kernel(uint32_t n, const uint32_t* __restrict__ keys,
			const uint32_t* __restrict__ cand, const uint64_t* __restrict__ row_off, const uint32_t* __restrict__ row_len,
			const uint32_t* __restrict__ col, const unsigned long long* __restrict__ coff, const uint32_t* __restrict__ crow,
			uint32_t npre, const uint32_t* __restrict__ pre, uint32_t* __restrict__ mark, unsigned long long* __restrict__ owner,
			uint32_t* __restrict__ out_r, uint32_t* __restrict__ out_c, uint32_t* __restrict__ out_n) {
			typedef cub::BlockScan<uint32_t, 1024> scan_t;
			__shared__ typename scan_t::TempStorage scan_tmp;
			__shared__ uint32_t s_pref[1025]; // inclusive prefix of the lengths of the lists to walk
			__shared__ uint32_t s_cd[1024];
			__shared__ int s_any;
			__shared__ uint32_t s_n, s_tot;
			const int tid = threadIdx.x;
			const uint32_t* lst = MODE == 0 ? col : crow;
			if (tid == 0) s_n = 0;
			if (MODE == 1) {
				for (uint32_t j = 0; j < npre; j++) {
					const uint32_t c = pre[j];
					for (unsigned long long t = coff[c] + tid; t < coff[c + 1]; t += blockDim.x)
						atomicMin(&mark[crow[t]], 0u);
				}
			}
			__syncthreads();
			// the lists of the items with flag set: their lengths' prefix in s_pref, the total in s_tot
			auto prefix = [&](bool flag, uint32_t cd) {
				uint64_t b = 0;
				uint32_t len = 0;
				if (flag) slist<MODE>(cd, row_off, row_len, coff, b, len);
				uint32_t incl;
				scan_t(scan_tmp).InclusiveSum(len, incl);
				s_pref[tid + 1] = incl;
				if (tid == 0) s_pref[0] = 0;
				if (tid == (int)blockDim.x - 1) s_tot = incl;
				__syncthreads();
				};
			// walk the entries of the lists prefix() described, all threads sharing them: f(item, k) for
			// entry k of the list of item (a thread index of this batch)
			auto walk_entries = [&](auto&& f) {
				const uint32_t tot = s_tot;
				for (uint32_t t = tid; t < tot; t += blockDim.x) {
					uint32_t lo = 0, hi = blockDim.x;
					while (lo + 1 < hi) {
						const uint32_t mid = (lo + hi) >> 1;
						if (s_pref[mid] <= t) lo = mid; else hi = mid;
					}
					f(lo, t - s_pref[lo]);
				}
				__syncthreads();
				};
			auto walk = [&](bool flag, uint32_t cd, auto&& f) {
				prefix(flag, cd);
				walk_entries(f);
				};
			uint32_t epoch = 0;
			for (uint32_t base = 0; base < n; base += blockDim.x) {
				const uint32_t i = base + tid, tag = i + 1;
				uint32_t key = 0, cd = NONE;
				bool valid = false, taken = false;
				if (i < n) {
					cd = cand[i];
					key = keys[i];
					valid = cd != NONE && __ldcg(&mark[key]) >= tag;
				}
				s_cd[tid] = cd;
				while (true) {
					if (tid == 0) s_any = 0;
					__syncthreads();
					if (valid) s_any = 1;
					__syncthreads();
					if (!s_any) break;
					epoch++;
					// stamping all undecided items pays when their lists are short in all; otherwise the
					// earliest undecided item alone is taken (its marks usually block much of the rest)
					prefix(valid, cd);
					bool take;
					if (s_tot <= 16384u) {
						walk_entries([&](uint32_t item, uint32_t k) {
							uint64_t b; uint32_t len;
							slist<MODE>(s_cd[item], row_off, row_len, coff, b, len);
							atomicMax(&owner[lst[b + k]], ((unsigned long long)epoch << 32) | (0xffffffffu - (base + item)));
							});
						take = valid && __ldcg(&owner[key]) == (((unsigned long long)epoch << 32) | (0xffffffffu - i));
					}
					else {
						uint32_t before;
						scan_t(scan_tmp).ExclusiveSum(valid ? 1u : 0u, before);
						take = valid && before == 0;
						__syncthreads();
					}
					if (take) { taken = true; valid = false; }
					walk(take, cd, [&](uint32_t item, uint32_t k) {
						uint64_t b; uint32_t len;
						slist<MODE>(s_cd[item], row_off, row_len, coff, b, len);
						atomicMin(&mark[lst[b + k]], base + item + 1);
						});
					if (valid && __ldcg(&mark[key]) < tag) valid = false;
				}
				uint32_t pos;
				scan_t(scan_tmp).ExclusiveSum(taken ? 1u : 0u, pos);
				if (taken) {
					if (MODE == 0) { out_r[s_n + pos] = cd; out_c[s_n + pos] = key; }
					else { out_r[s_n + pos] = key; out_c[s_n + pos] = cd; }
				}
				__syncthreads();
				if (tid == (int)blockDim.x - 1) s_n += pos + (taken ? 1u : 0u);
				__syncthreads();
			}
			if (tid == 0) *out_n = s_n;
		}

		// whether a pivot row holds the column of an earlier pivot of the list (bit 1) or lacks its own
		// column (bit 2) (lvlpos: list position by column)
		__global__ void lvlcheck_kernel(uint32_t P, const uint32_t* __restrict__ prow, const uint64_t* __restrict__ row_off,
			const uint32_t* __restrict__ row_len, const uint32_t* __restrict__ col, const int32_t* __restrict__ lvlpos,
			unsigned int* __restrict__ err) {
			__shared__ int found;
			for (uint32_t j = blockIdx.x; j < P; j += gridDim.x) {
				const uint32_t r = prow[j];
				const uint64_t off = row_off[r];
				if (threadIdx.x == 0) found = 0;
				__syncthreads();
				for (uint32_t k = threadIdx.x; k < row_len[r]; k += blockDim.x) {
					const int32_t q = lvlpos[col[off + k]];
					if (q >= 0 && (uint32_t)q < j)
						atomicOr(err, 1u);
					if ((uint32_t)q == j)
						found = 1;
				}
				__syncthreads();
				if (threadIdx.x == 0 && !found) // the pivot row does not hold its column (yet)
					atomicOr(err, 2u);
				__syncthreads();
			}
		}
		__global__ void stamp_kernel(uint32_t B, const uint32_t* __restrict__ rows, uint32_t* __restrict__ stamp, uint32_t id) {
			const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
			if (j < B) stamp[rows[j]] = id;
		}
		// the rows that hold a pivot column of [lo, hi), each once, other than the stamped ones
		__global__ void collect_kernel(uint32_t lo, uint32_t hi, const unsigned long long* __restrict__ toff,
			const uint32_t* __restrict__ trows, uint32_t* __restrict__ stamp, uint32_t id,
			uint32_t* __restrict__ list, unsigned int* __restrict__ count) {
			for (uint32_t j = lo + blockIdx.x; j < hi; j += gridDim.x)
				for (unsigned long long t = toff[j] + threadIdx.x; t < toff[j + 1]; t += blockDim.x) {
					const uint32_t r = trows[t];
					if (atomicExch(&stamp[r], id) != id)
						list[atomicAdd(count, 1u)] = r;
				}
		}

		template <typename X>
		struct dvec {
			X* p = nullptr;
			size_t n = 0;
			bool alloc(size_t m) {
				if (m <= n && p) return true;
				if (p) cudaFree(p);
				p = nullptr; n = 0;
				if (cudaMalloc(&p, std::max<size_t>(m, 1) * sizeof(X)) != cudaSuccess) { p = nullptr; cudaGetLastError(); return false; }
				n = std::max<size_t>(m, 1);
				return true;
			}
			void free() { if (p) cudaFree(p); p = nullptr; n = 0; }
		};

		struct res_t {
			bool on = false;
			uint64_t p = 0;
			uint32_t nrow = 0, ncol = 0, nlive = 0;
			int sms = 0;
			size_t smem_optin = 0;
			cudaStream_t s = nullptr;
			uint64_t cap = 0, top = 0;
			dvec<uint32_t> col[2], tp;
			dvec<uint64_t> val[2];
			int cur = 0;
			dvec<uint64_t> row_off;
			dvec<uint32_t> row_len, live;          // live: the uploaded rows
			dvec<uint32_t> inU, pos, piv_cid, col_of_cid, prow, pcol, rows;
			dvec<int32_t> cid_of_col, posU;
			dvec<uint64_t> plen, pv_off, pv_val, lens, noff, scratch;
			dvec<uint32_t> pv_cid;
			dvec<uint8_t> status;
			dvec<unsigned long long> dtop;
			dvec<char> cubtmp;
			dvec<uint32_t> retry, bprow, bpcol, trows, stamp, list;
			dvec<int32_t> colpiv, rowpiv, lvlpos;
			// the pivot search
			dvec<int64_t> sw;                         // column weights
			dvec<uint32_t> slrows, slcols, slrow, srcol, sbits, sout;
			dvec<unsigned long long> scnt, skey, soff, scur, snnz, sown;
			dvec<uint8_t> sempty;
			dvec<uint32_t> soutn;
			dvec<unsigned long long> tcnt, toff, tcur;
			dvec<unsigned int> dcount, perr;
			cudaEvent_t ev[2] = {};
			bool scratch_zero = false;
			std::vector<uint32_t> changed; // rows written by the last schur_dev
			res_stats st;
		};
		res_t g_res;

		bool scan(dvec<uint64_t>& in, dvec<uint64_t>& out, size_t n, cudaStream_t s) {
			auto& g = g_res;
			size_t tb = 0;
			if (cub::DeviceScan::ExclusiveSum(nullptr, tb, in.p, out.p, n, s) != cudaSuccess) return false;
			if (!g.cubtmp.alloc(tb + 16)) return false;
			return cub::DeviceScan::ExclusiveSum(g.cubtmp.p, tb, in.p, out.p, n, s) == cudaSuccess;
		}
		bool scan32(dvec<uint32_t>& in, dvec<uint32_t>& out, size_t n, cudaStream_t s) {
			auto& g = g_res;
			size_t tb = 0;
			if (cub::DeviceScan::ExclusiveSum(nullptr, tb, in.p, out.p, n, s) != cudaSuccess) return false;
			if (!g.cubtmp.alloc(tb + 16)) return false;
			return cub::DeviceScan::ExclusiveSum(g.cubtmp.p, tb, in.p, out.p, n, s) == cudaSuccess;
		}

		// gather rows[0..R) contiguously into the other arena (and its offsets into noff[0..R]); returns the total
		bool gather_rows(uint32_t R, const uint32_t* d_rows, uint64_t& total, bool with_val = true) {
			auto& g = g_res;
			cudaStream_t s = g.s;
			if (!g.lens.alloc((size_t)R + 1) || !g.noff.alloc((size_t)R + 1)) return false;
			lens_kernel<<<(R + 1 + 255) / 256, 256, 0, s>>>(R, d_rows, g.row_len.p, g.lens.p);
			if (!scan(g.lens, g.noff, (size_t)R + 1, s)) return false;
			if (cudaMemcpyAsync(&total, g.noff.p + R, 8, cudaMemcpyDeviceToHost, s) != cudaSuccess
				|| cudaStreamSynchronize(s) != cudaSuccess)
				return false;
			if (total > g.cap) return false;
			const int o = 1 - g.cur;
			if (R > 0)
				gather_kernel<<<std::min<uint32_t>(R, 65535u * 8), 128, 0, s>>>(R, d_rows, g.row_off.p, g.row_len.p,
					g.col[g.cur].p, g.val[g.cur].p, g.noff.p, g.col[o].p, with_val ? g.val[o].p : nullptr);
			return cudaGetLastError() == cudaSuccess;
		}

		// move the live rows into the other arena, contiguously
		bool compact() {
			auto& g = g_res;
			auto t0 = clk::now();
			uint64_t total = 0;
			if (!gather_rows(g.nlive, g.live.p, total)) return false;
			if (g.nlive > 0)
				setoff_kernel<<<(g.nlive + 255) / 256, 256, 0, g.s>>>(g.nlive, g.live.p, g.noff.p, g.row_off.p);
			if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(g.s) != cudaSuccess) return false;
			g.cur = 1 - g.cur;
			g.top = total;
			g.st.compactions++;
			g.st.compact += since(t0);
			return true;
		}
	}

	res_stats& res_profile() { return g_res.st; }

	void res_end() {
		auto& g = g_res;
		if (g.s) cudaStreamSynchronize(g.s);
		for (auto* d : { &g.col[0], &g.col[1], &g.tp, &g.row_len, &g.live, &g.inU, &g.pos, &g.piv_cid, &g.col_of_cid,
			&g.prow, &g.pcol, &g.rows, &g.pv_cid })
			d->free();
		for (auto* d : { &g.val[0], &g.val[1], &g.row_off, &g.plen, &g.pv_off, &g.pv_val, &g.lens, &g.noff, &g.scratch })
			d->free();
		g.cid_of_col.free(); g.posU.free(); g.status.free(); g.dtop.free(); g.cubtmp.free();
		for (auto* d : { &g.retry, &g.bprow, &g.bpcol, &g.trows, &g.stamp, &g.list }) d->free();
		for (auto* d : { &g.slrows, &g.slcols, &g.slrow, &g.srcol, &g.sbits, &g.sout, &g.soutn }) d->free();
		for (auto* d : { &g.scnt, &g.skey, &g.soff, &g.scur, &g.snnz, &g.sown }) d->free();
		g.sw.free(); g.sempty.free();
		g.colpiv.free(); g.rowpiv.free(); g.lvlpos.free(); g.tcnt.free(); g.toff.free(); g.tcur.free(); g.dcount.free(); g.perr.free();
		for (auto& e : g.ev) {
			if (e) cudaEventDestroy(e);
			e = nullptr;
		}
		if (g.s) cudaStreamDestroy(g.s);
		g.s = nullptr;
		g.on = false;
		g.scratch_zero = false;
	}

	bool res_begin(uint64_t p, uint32_t nrow, uint32_t ncol, uint32_t R, const uint32_t* rows,
		const uint64_t* off, const res_fill_fn& fill) {
		res_end();
		auto& g = g_res;
		g.st = res_stats();
		if (!available()) return false;
		int dev = 0, v = 0;
		if (cudaGetDevice(&dev) != cudaSuccess) return false;
		cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev); g.sms = v;
		cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev); g.smem_optin = (size_t)v;
		if (cudaStreamCreateWithFlags(&g.s, cudaStreamNonBlocking) != cudaSuccess) { g.s = nullptr; return false; }
		for (auto& e : g.ev)
			if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) { e = nullptr; res_end(); return false; }
		g.p = p; g.nrow = nrow; g.ncol = ncol; g.nlive = R;
		const uint64_t nnz = off[R];
		// two arenas and the pass-through scratch (28 bytes per entry) take most of the free memory,
		// leaving room for the per-level buffers and the large-U accumulators
		size_t fr = 0, tot = 0;
		if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { res_end(); return false; }
		const size_t reserve = scratch_budget + ((size_t)1 << 30) + (size_t)ncol * 24 + (size_t)nrow * 12;
		if (fr <= reserve) { res_end(); return false; }
		g.cap = (uint64_t)((fr - reserve) * 0.85 / 28);
#ifdef SPARSERREF_TEST_HOOKS
		if (const char* e = std::getenv("SPARSERREF_GPU_RES_CAP")) // testing: a smaller arena compacts more often
			g.cap = std::min<uint64_t>(g.cap, std::strtoull(e, nullptr, 10));
#endif
		if (g.cap < nnz + nnz / 2 + 1024) { res_end(); return false; }
		bool ok = g.col[0].alloc(g.cap) && g.col[1].alloc(g.cap) && g.val[0].alloc(g.cap) && g.val[1].alloc(g.cap)
			&& g.tp.alloc(g.cap) && g.row_off.alloc(nrow) && g.row_len.alloc(nrow) && g.live.alloc(R)
			&& g.inU.alloc(ncol) && g.pos.alloc(ncol) && g.cid_of_col.alloc(ncol) && g.col_of_cid.alloc(ncol)
			&& g.posU.alloc(ncol) && g.dtop.alloc(1) && g.lvlpos.alloc(ncol)
			&& cudaMemsetAsync(g.lvlpos.p, 0xff, (size_t)ncol * 4, g.s) == cudaSuccess;
		if (!ok) { res_end(); return false; }
		g.cur = 0;
		auto t0 = clk::now();
		// rows go to the start of arena 0 in the given order
		std::vector<uint64_t> roff(nrow, 0);
		std::vector<uint32_t> rlen(nrow, 0);
		for (uint32_t i = 0; i < R; i++) {
			roff[rows[i]] = off[i];
			rlen[rows[i]] = (uint32_t)(off[i + 1] - off[i]);
		}
		// the entries in pieces through the two staging buffers: the host fills one while the other is copied
		bool used[2] = { false, false };
		for (uint32_t r0 = 0, b = 0; ok && r0 < R; b ^= 1) {
			uint32_t r1 = r0 + 1;
			while (r1 < R && off[r1 + 1] - off[r0] <= piece_entries())
				r1++;
			const uint64_t n = off[r1] - off[r0];
			uint32_t* hc;
			uint64_t* hv;
			ok = (!used[b] || cudaEventSynchronize(g.ev[b]) == cudaSuccess) && staging((int)b, n, hc, hv);
			if (!ok) break;
			fill(r0, r1, hc, hv);
			ok = (n == 0 || (cudaMemcpyAsync(g.col[0].p + off[r0], hc, n * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess
				&& cudaMemcpyAsync(g.val[0].p + off[r0], hv, n * 8, cudaMemcpyHostToDevice, g.s) == cudaSuccess))
				&& cudaEventRecord(g.ev[b], g.s) == cudaSuccess;
			used[b] = true;
			r0 = r1;
		}
		ok = ok && cudaMemcpyAsync(g.row_off.p, roff.data(), (size_t)nrow * 8, cudaMemcpyHostToDevice, g.s) == cudaSuccess
			&& cudaMemcpyAsync(g.row_len.p, rlen.data(), (size_t)nrow * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess
			&& cudaMemcpyAsync(g.live.p, rows, (size_t)R * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess
			&& cudaStreamSynchronize(g.s) == cudaSuccess;
		if (!ok) { res_end(); return false; }
		g.top = nnz;
		g.st.download += since(t0); // upload time counted with the transfers
		g.on = true;
		return true;
	}

	bool res_rescale(uint32_t P, const uint32_t* prow, const uint32_t* pcol) {
		auto& g = g_res;
		if (!g.on) return false;
		if (P == 0) return true;
		auto t0 = clk::now();
		bool ok = g.prow.alloc(P) && g.pcol.alloc(P)
			&& cudaMemcpyAsync(g.prow.p, prow, (size_t)P * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess
			&& cudaMemcpyAsync(g.pcol.p, pcol, (size_t)P * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess;
		if (!ok) return false;
		const uint64_t threads = (uint64_t)P * 32;
		rescale_kernel<<<(unsigned)((threads + 255) / 256), 256, 0, g.s>>>(P, g.prow.p, g.pcol.p, g.row_off.p, g.row_len.p,
			g.col[g.cur].p, g.val[g.cur].p, g.p);
		ok = cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(g.s) == cudaSuccess;
		g.st.rescale += since(t0);
		return ok && !fail_inject("rescale"); // testing: after the rows were rescaled
	}

	namespace {
	// update the rows d_rows[0..R) by the ordered pivots (d_prow, d_pcol)[0..P), all device arrays
	bool schur_dev(uint32_t P, const uint32_t* d_prow, const uint32_t* d_pcol, uint32_t R, const uint32_t* d_rows,
		const int32_t* d_own) {
		auto& g = g_res;
		if (P == 0 || R == 0) return true;
		cudaStream_t s = g.s;
		auto t0 = clk::now();
		// U, the compact pivot rows, posU
		bool ok = g.piv_cid.alloc(P) && g.plen.alloc((size_t)P + 1) && g.pv_off.alloc((size_t)P + 1) && g.status.alloc(R)
			&& cudaMemsetAsync(g.inU.p, 0, (size_t)g.ncol * 4, s) == cudaSuccess;
		if (!ok) return false;
		const uint32_t* acol = g.col[g.cur].p;
		mark_kernel<<<std::min<uint32_t>(P, 65535u), 128, 0, s>>>(P, d_prow, g.row_off.p, g.row_len.p, acol, g.inU.p);
		if (cudaGetLastError() != cudaSuccess || !scan32(g.inU, g.pos, g.ncol, s)) return false;
		number_kernel<<<(g.ncol + 255) / 256, 256, 0, s>>>(g.ncol, g.inU.p, g.pos.p, g.cid_of_col.p, g.col_of_cid.p);
		uint32_t last_pos = 0, last_in = 0;
		ok = cudaGetLastError() == cudaSuccess
			&& cudaMemcpyAsync(&last_pos, g.pos.p + g.ncol - 1, 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaMemcpyAsync(&last_in, g.inU.p + g.ncol - 1, 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaMemsetAsync(g.posU.p, 0xff, (size_t)g.ncol * 4, s) == cudaSuccess;
		if (!ok) return false;
		unsigned int perr = 0;
		ok = g.perr.alloc(1) && cudaMemsetAsync(g.perr.p, 0, 4, s) == cudaSuccess;
		if (!ok) return false;
		pivinfo_kernel<<<(P + 255) / 256, 256, 0, s>>>(P, d_prow, d_pcol, g.row_len.p, g.cid_of_col.p, g.piv_cid.p, g.posU.p, g.plen.p, g.perr.p);
		ok = cudaGetLastError() == cudaSuccess && cudaMemsetAsync(g.plen.p + P, 0, 8, s) == cudaSuccess
			&& scan(g.plen, g.pv_off, (size_t)P + 1, s);
		uint64_t nPv = 0;
		ok = ok && cudaMemcpyAsync(&nPv, g.pv_off.p + P, 8, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaMemcpyAsync(&perr, g.perr.p, 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaStreamSynchronize(s) == cudaSuccess;
		if (!ok || perr) return false;
		const uint32_t nU = last_pos + last_in;
		if (!g.pv_cid.alloc(nPv + 1) || !g.pv_val.alloc(nPv + 1)) return false;
		pvfill_kernel<<<std::min<uint32_t>(P, 65535u), 128, 0, s>>>(P, d_prow, g.row_off.p, g.row_len.p, acol, g.val[g.cur].p,
			g.cid_of_col.p, g.pv_off.p, g.pv_cid.p, g.pv_val.p);
		if (cudaGetLastError() != cudaSuccess) return false;
		// kernel shape
		const bool global = smem_bytes(nU, P) > g.smem_optin;
		if (global && smem_bytes_global(nU, P) > g.smem_optin) return false;
		const size_t sm = global ? smem_bytes_global(nU, P) : smem_bytes(nU, P);
		auto kern = global ? res_kernel<true> : res_kernel<false>;
		if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) != cudaSuccess) return false;
		int occ = 0;
		if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, BS, sm) != cudaSuccess || occ < 1) return false;
		uint64_t blocks = (uint64_t)occ * g.sms;
		size_t sw = 0;
		if (global) {
			sw = slot_words(nU);
			blocks = std::min<uint64_t>(blocks, scratch_budget / (sw * 8));
			if (blocks < 1) return false;
			// the kernel leaves its accumulators zero: clear only fresh memory
			uint64_t* before = g.scratch.p;
			if (!g.scratch.alloc(blocks * sw)) return false;
			if ((g.scratch.p != before || !g.scratch_zero) && cudaMemsetAsync(g.scratch.p, 0, g.scratch.n * 8, s) != cudaSuccess) return false;
			g.scratch_zero = true;
		}
		g.st.prep += since(t0);

		// update; rows that do not fit wait for a compaction and go again
		auto t1 = clk::now();
		g.changed.clear();
		std::vector<uint32_t> todo;
		const uint32_t* cur_rows = d_rows;
		uint32_t n = R;
		for (int attempt = 0; attempt < 3 && n > 0; attempt++) {
			if (attempt > 0) {
				g.st.kernel += since(t1);
				if (!compact()) return false;
				t1 = clk::now();
				if (!g.retry.alloc(n) || cudaMemcpyAsync(g.retry.p, todo.data(), (size_t)n * 4, cudaMemcpyHostToDevice, s) != cudaSuccess)
					return false;
				cur_rows = g.retry.p;
			}
			unsigned long long top = g.top;
			if (cudaMemcpyAsync(g.dtop.p, &top, 8, cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
			const uint32_t grid = (uint32_t)std::min<uint64_t>(n, blocks);
			kern<<<grid, BS, sm, s>>>(g.scratch.p, sw, g.p, nU, P, n,
				g.pv_off.p, g.pv_cid.p, g.pv_val.p, g.piv_cid.p, g.posU.p, g.cid_of_col.p, g.col_of_cid.p,
				cur_rows, g.row_off.p, g.row_len.p, g.col[g.cur].p, g.val[g.cur].p, g.tp.p, d_own,
				g.status.p, g.dtop.p, (unsigned long long)g.cap);
			std::vector<uint8_t> st(n);
			std::vector<uint32_t> ids(attempt == 0 ? n : 0);
			if (attempt == 0 && cudaMemcpyAsync(ids.data(), d_rows, (size_t)n * 4, cudaMemcpyDeviceToHost, s) != cudaSuccess)
				return false;
			ok = cudaGetLastError() == cudaSuccess
				&& cudaMemcpyAsync(st.data(), g.status.p, n, cudaMemcpyDeviceToHost, s) == cudaSuccess
				&& cudaMemcpyAsync(&top, g.dtop.p, 8, cudaMemcpyDeviceToHost, s) == cudaSuccess
				&& cudaStreamSynchronize(s) == cudaSuccess;
			if (!ok) return false;
			g.top = std::min<uint64_t>(top, g.cap);
			std::vector<uint32_t> again;
			for (uint32_t i = 0; i < n; i++) {
				const uint32_t r = attempt == 0 ? ids[i] : todo[i];
				if (st[i] == 1) again.push_back(r);
				else if (st[i] == 2) g.changed.push_back(r);
			}
			todo.swap(again);
			n = (uint32_t)todo.size();
			// after an overflow the top is past the capacity: the next compaction resets it
		}
		g.st.kernel += since(t1);
		return n == 0;
	}
	}

	bool res_schur(uint32_t P, const uint32_t* prow, const uint32_t* pcol, uint32_t R, const uint32_t* rows) {
		auto& g = g_res;
		if (!g.on) return false;
		if (P == 0 || R == 0) return true;
		bool ok = g.prow.alloc(P) && g.pcol.alloc(P) && g.rows.alloc(R)
			&& cudaMemcpyAsync(g.prow.p, prow, (size_t)P * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess
			&& cudaMemcpyAsync(g.pcol.p, pcol, (size_t)P * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess
			&& cudaMemcpyAsync(g.rows.p, rows, (size_t)R * 4, cudaMemcpyHostToDevice, g.s) == cudaSuccess;
		return ok && schur_dev(P, g.prow.p, g.pcol.p, R, g.rows.p, nullptr) && !fail_inject("schur"); // testing: after the update
	}

	bool res_backsub(uint32_t P, const uint32_t* prow, const uint32_t* pcol, uint32_t n_split) {
		auto& g = g_res;
		if (!g.on) return false;
		if (P == 0) return true;
		if (n_split == 0) n_split = 1;
		auto t0 = clk::now();
		cudaStream_t s = g.s;
		// the pivot column transpose of the live rows, built once: back substitution never creates
		// entries in the pivot columns of the batches still to come (pivot k's row holds only pivot
		// columns after k), so it stays exact
		bool ok = g.bprow.alloc(P) && g.bpcol.alloc(P) && g.colpiv.alloc(g.ncol) && g.rowpiv.alloc(g.nrow)
			&& g.stamp.alloc(g.nrow) && g.tcnt.alloc((size_t)P + 1) && g.toff.alloc((size_t)P + 1) && g.tcur.alloc((size_t)P + 1)
			&& g.dcount.alloc(1) && g.list.alloc(g.nlive)
			&& cudaMemcpyAsync(g.bprow.p, prow, (size_t)P * 4, cudaMemcpyHostToDevice, s) == cudaSuccess
			&& cudaMemcpyAsync(g.bpcol.p, pcol, (size_t)P * 4, cudaMemcpyHostToDevice, s) == cudaSuccess
			&& cudaMemsetAsync(g.colpiv.p, 0xff, (size_t)g.ncol * 4, s) == cudaSuccess
			&& cudaMemsetAsync(g.rowpiv.p, 0xff, (size_t)g.nrow * 4, s) == cudaSuccess
			&& cudaMemsetAsync(g.stamp.p, 0, (size_t)g.nrow * 4, s) == cudaSuccess
			&& cudaMemsetAsync(g.tcnt.p, 0, ((size_t)P + 1) * 8, s) == cudaSuccess;
		if (!ok) return false;
		setpos_kernel<<<(P + 255) / 256, 256, 0, s>>>(P, g.bpcol.p, g.colpiv.p, 0);
		tcount_kernel<<<std::min<uint32_t>(std::max<uint32_t>(g.nlive, 1), 65535u), 128, 0, s>>>(g.nlive, g.live.p, g.row_off.p,
			g.row_len.p, g.col[g.cur].p, g.colpiv.p, g.bprow.p, g.tcnt.p);
		{
			size_t tb = 0;
			ok = cudaGetLastError() == cudaSuccess
				&& cub::DeviceScan::ExclusiveSum(nullptr, tb, g.tcnt.p, g.toff.p, (size_t)P + 1, s) == cudaSuccess
				&& g.cubtmp.alloc(tb + 16)
				&& cub::DeviceScan::ExclusiveSum(g.cubtmp.p, tb, g.tcnt.p, g.toff.p, (size_t)P + 1, s) == cudaSuccess;
		}
		unsigned long long T = 0;
		ok = ok && cudaMemcpyAsync(&T, g.toff.p + P, 8, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaMemcpyAsync(g.tcur.p, g.toff.p, ((size_t)P + 1) * 8, cudaMemcpyDeviceToDevice, s) == cudaSuccess
			&& cudaStreamSynchronize(s) == cudaSuccess && g.trows.alloc(T + 1);
		if (!ok) return false;
		tfill_kernel<<<std::min<uint32_t>(std::max<uint32_t>(g.nlive, 1), 65535u), 128, 0, s>>>(g.nlive, g.live.p, g.row_off.p,
			g.row_len.p, g.col[g.cur].p, g.colpiv.p, g.bprow.p, g.tcur.p, g.trows.p);
		if (cudaGetLastError() != cudaSuccess) return false;
		// batches from the end: first the batch reduces itself (each row by the later pivots of the
		// batch), then the rows that hold its pivot columns are reduced by its (now reduced) rows
		uint32_t hi = P, id = 1;
		while (hi > 0) {
			const uint32_t lo = hi > n_split ? hi - n_split : 0, B = hi - lo;
			if (B > 1) {
				setpos_kernel<<<(B + 255) / 256, 256, 0, s>>>(B, g.bprow.p + lo, g.rowpiv.p, 0);
				if (cudaGetLastError() != cudaSuccess
					|| !schur_dev(B, g.bprow.p + lo, g.bpcol.p + lo, B, g.bprow.p + lo, g.rowpiv.p))
					return false;
				setpos_kernel<<<(B + 255) / 256, 256, 0, s>>>(B, g.bprow.p + lo, g.rowpiv.p, 1);
			}
			stamp_kernel<<<(B + 255) / 256, 256, 0, s>>>(B, g.bprow.p + lo, g.stamp.p, id);
			unsigned int cnt = 0;
			ok = cudaMemsetAsync(g.dcount.p, 0, 4, s) == cudaSuccess;
			if (!ok) return false;
			collect_kernel<<<std::min<uint32_t>(B, 65535u), 128, 0, s>>>(lo, hi, g.toff.p, g.trows.p, g.stamp.p, id,
				g.list.p, g.dcount.p);
			ok = cudaGetLastError() == cudaSuccess
				&& cudaMemcpyAsync(&cnt, g.dcount.p, 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
				&& cudaStreamSynchronize(s) == cudaSuccess;
			if (!ok) return false;
			if (cnt > 0 && !schur_dev(B, g.bprow.p + lo, g.bpcol.p + lo, cnt, g.list.p, nullptr))
				return false;
			hi = lo;
			id++;
			if (hi > 0 && hi <= P / 2 && fail_inject("backsub")) // testing: half way
				return false;
		}
		g.st.backsub += since(t0);
		return true;
	}

	bool res_download_pieces(uint32_t R, const uint32_t* rows, bool with_val, const res_take_fn& take) {
		auto& g = g_res;
		if (!g.on) return false;
		if (R == 0) return true;
		auto t0 = clk::now();
		if (!g.rows.alloc(R) || cudaMemcpyAsync(g.rows.p, rows, (size_t)R * 4, cudaMemcpyHostToDevice, g.s) != cudaSuccess)
			return false;
		uint64_t total = 0;
		if (!gather_rows(R, g.rows.p, total, with_val)) return false;
		std::vector<uint64_t> hoff((size_t)R + 1), rel;
		if (cudaMemcpyAsync(hoff.data(), g.noff.p, ((size_t)R + 1) * 8, cudaMemcpyDeviceToHost, g.s) != cudaSuccess
			|| cudaStreamSynchronize(g.s) != cudaSuccess)
			return false;
		const int o = 1 - g.cur;
		// the pieces; piece i + 1 is copied while the host takes piece i
		std::vector<uint32_t> cut{ 0 };
		while (cut.back() < R) {
			const uint32_t r0 = cut.back();
			uint32_t r1 = r0 + 1;
			while (r1 < R && hoff[r1 + 1] - hoff[r0] <= piece_entries())
				r1++;
			cut.push_back(r1);
		}
		const size_t np = cut.size() - 1;
		std::vector<uint32_t*> hc(np);
		std::vector<uint64_t*> hv(np);
		auto issue = [&](size_t i) -> bool {
			const uint64_t a = hoff[cut[i]], n = hoff[cut[i + 1]] - a;
			if (!staging((int)(i & 1), n, hc[i], hv[i])) return false;
			return n == 0 || (cudaMemcpyAsync(hc[i], g.col[o].p + a, n * 4, cudaMemcpyDeviceToHost, g.s) == cudaSuccess
				&& (!with_val || cudaMemcpyAsync(hv[i], g.val[o].p + a, n * 8, cudaMemcpyDeviceToHost, g.s) == cudaSuccess));
		};
		if (!issue(0) || cudaEventRecord(g.ev[0], g.s) != cudaSuccess) return false;
		for (size_t i = 0; i < np; i++) {
			if (cudaEventSynchronize(g.ev[i & 1]) != cudaSuccess) return false;
			if (i + 1 < np && (!issue(i + 1) || cudaEventRecord(g.ev[(i + 1) & 1], g.s) != cudaSuccess)) return false;
			const uint32_t r0 = cut[i], r1 = cut[i + 1];
			rel.resize((size_t)(r1 - r0) + 1);
			for (uint32_t r = r0; r <= r1; r++)
				rel[r - r0] = hoff[r] - hoff[r0];
			if (!take(r0, r1, rel.data(), hc[i], with_val ? hv[i] : nullptr)) {
				cudaStreamSynchronize(g.s);
				return false;
			}
		}
		if (with_val) g.st.download += since(t0);
		else g.st.struct_dl += since(t0);
		return true;
	}

	bool res_search_weights(const int64_t* w) {
		auto& g = g_res;
		if (!g.on) return false;
		return g.sw.alloc(g.ncol) && cudaMemcpyAsync(g.sw.p, w, (size_t)g.ncol * 8, cudaMemcpyHostToDevice, g.s) == cudaSuccess
			&& cudaStreamSynchronize(g.s) == cudaSuccess;
	}

	bool res_search(uint32_t R, const uint32_t* leftrows, uint32_t L, const uint32_t* leftcols,
		std::vector<uint32_t>& prow, std::vector<uint32_t>& pcol, std::vector<uint8_t>& empty, uint64_t& nnz) {
		auto& g = g_res;
		if (!g.on || !g.sw.p) return false;
		auto t0 = clk::now();
		cudaStream_t s = g.s;
		const uint32_t ncol = g.ncol, nrow = g.nrow;
		bool ok = g.slrows.alloc(R) && g.slcols.alloc(L) && g.slrow.alloc(L) && g.srcol.alloc(R) && g.sempty.alloc(R)
			&& g.scnt.alloc(ncol) && g.skey.alloc(ncol) && g.soff.alloc((size_t)ncol + 1) && g.scur.alloc((size_t)ncol + 1)
			&& g.snnz.alloc(1) && g.sout.alloc(2 * ((size_t)L + R) + 2) && g.soutn.alloc(2)
			&& (R == 0 || cudaMemcpyAsync(g.slrows.p, leftrows, (size_t)R * 4, cudaMemcpyHostToDevice, s) == cudaSuccess)
			&& (L == 0 || cudaMemcpyAsync(g.slcols.p, leftcols, (size_t)L * 4, cudaMemcpyHostToDevice, s) == cudaSuccess)
			&& cudaMemsetAsync(g.scnt.p, 0, (size_t)ncol * 8, s) == cudaSuccess
			&& cudaMemsetAsync(g.skey.p, 0xff, (size_t)ncol * 8, s) == cudaSuccess
			&& cudaMemsetAsync(g.snnz.p, 0, 8, s) == cudaSuccess;
		if (!ok) return false;
		const uint32_t* acol = g.col[g.cur].p;
		const uint32_t grid = std::min<uint32_t>(std::max<uint32_t>(R, 1), 65535u * 4);
		if (R > 0)
			scount_kernel<<<grid, 128, 0, s>>>(R, g.slrows.p, g.row_off.p, g.row_len.p, acol, g.scnt.p, g.skey.p, g.sempty.p, g.snnz.p);
		// offsets of the columns' row lists (in the pass-through scratch: no Schur update runs meanwhile)
		{
			size_t tb = 0;
			ok = cudaGetLastError() == cudaSuccess
				&& cub::DeviceScan::ExclusiveSum(nullptr, tb, g.scnt.p, g.soff.p, (size_t)ncol, s) == cudaSuccess
				&& g.cubtmp.alloc(tb + 16)
				&& cub::DeviceScan::ExclusiveSum(g.cubtmp.p, tb, g.scnt.p, g.soff.p, (size_t)ncol, s) == cudaSuccess;
		}
		unsigned long long last_off = 0, last_cnt = 0, hn = 0;
		ok = ok && cudaMemcpyAsync(&last_off, g.soff.p + ncol - 1, 8, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaMemcpyAsync(&last_cnt, g.scnt.p + ncol - 1, 8, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaMemcpyAsync(&hn, g.snnz.p, 8, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaStreamSynchronize(s) == cudaSuccess;
		if (!ok) return false;
		const unsigned long long total = last_off + last_cnt;
		if (total > g.cap) return false; // cannot happen: the rows of leftrows are in the arena
		ok = cudaMemcpyAsync(g.soff.p + ncol, &total, 8, cudaMemcpyHostToDevice, s) == cudaSuccess
			&& cudaMemcpyAsync(g.scur.p, g.soff.p, (size_t)ncol * 8, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
		if (!ok) return false;
		auto t1 = clk::now();
		g.st.s_count += since(t0);
		uint32_t* crow = g.tp.p;
		if (R > 0)
			sfill_kernel<<<grid, 128, 0, s>>>(R, g.slrows.p, g.row_off.p, g.row_len.p, acol, g.scur.p, crow);
		if (L > 0)
			slrow_kernel<<<(L + 255) / 256, 256, 0, s>>>(L, g.slcols.p, g.scnt.p, g.skey.p, g.sw.p, g.slrow.p);
		if (R > 0)
			srcol_kernel<<<(unsigned)std::min<uint64_t>(((uint64_t)R + 7) / 8, (uint64_t)g.sms * 64), 256, 0, s>>>(R, g.slrows.p,
				g.row_off.p, g.row_len.p, acol, g.scnt.p, g.sw.p, g.srcol.p);
		if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) return false;
		auto t2 = clk::now();
		g.st.s_cand += std::chrono::duration<double>(t2 - t1).count();
		// the two sequential passes; their marks in shared memory when they fit
		uint32_t* out_r = g.sout.p;
		uint32_t* out_c = g.sout.p + ((size_t)L + R + 1);
		// per key: the stamps (epoch and position) and the marks (smallest blocking position), cleared per pass
		if (!g.sown.alloc((size_t)std::max(ncol, nrow) + 1) || !g.sbits.alloc((size_t)std::max(ncol, nrow) + 1)) return false;
		unsigned long long* owner = g.sown.p;
		uint32_t* mark = g.sbits.p;
		auto launch = [&](auto kern, uint32_t n, uint32_t nkeys, const uint32_t* keys, const uint32_t* cand,
			uint32_t npre, const uint32_t* pre, uint32_t* o_r, uint32_t* o_c, uint32_t* o_n) -> bool {
			if (cudaMemsetAsync(owner, 0, ((size_t)nkeys + 1) * 8, s) != cudaSuccess
				|| cudaMemsetAsync(mark, 0xff, ((size_t)nkeys + 1) * 4, s) != cudaSuccess)
				return false;
			kern<<<1, 1024, 0, s>>>(n, keys, cand, g.row_off.p, g.row_len.p, acol, g.soff.p, crow, npre, pre, mark, owner, o_r, o_c, o_n);
			return cudaGetLastError() == cudaSuccess;
		};
		uint32_t nl = 0, nr = 0;
		ok = launch(sgreedy_kernel<0>, L, ncol, g.slcols.p, g.slrow.p, 0, nullptr, out_r, out_c, g.soutn.p)
			&& cudaMemcpyAsync(&nl, g.soutn.p, 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaStreamSynchronize(s) == cudaSuccess;
		if (!ok) return false;
		ok = launch(sgreedy_kernel<1>, R, nrow, g.slrows.p, g.srcol.p, nl, out_c, out_r + nl, out_c + nl, g.soutn.p + 1)
			&& cudaMemcpyAsync(&nr, g.soutn.p + 1, 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaStreamSynchronize(s) == cudaSuccess;
		if (!ok) return false;
		g.st.s_greedy += since(t2);
		std::vector<uint32_t> hr(nl + nr), hc(nl + nr);
		empty.resize(R);
		ok = (nl + nr == 0 || (cudaMemcpyAsync(hr.data(), out_r, (size_t)(nl + nr) * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaMemcpyAsync(hc.data(), out_c, (size_t)(nl + nr) * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess))
			&& (R == 0 || cudaMemcpyAsync(empty.data(), g.sempty.p, R, cudaMemcpyDeviceToHost, s) == cudaSuccess)
			&& cudaStreamSynchronize(s) == cudaSuccess;
		if (!ok) return false;
		// the left look's pivots in reverse order of acceptance (pivots_search puts them at the front)
		prow.clear();
		pcol.clear();
		for (uint32_t j = nl; j-- > 0;) { prow.push_back(hr[j]); pcol.push_back(hc[j]); }
		for (uint32_t j = nl; j < nl + nr; j++) { prow.push_back(hr[j]); pcol.push_back(hc[j]); }
		nnz = hn;
		g.st.search += since(t0);
		return !fail_inject("search");
	}

	int res_level_order(uint32_t P, const uint32_t* prow, const uint32_t* pcol) {
		auto& g = g_res;
		if (!g.on) return -1;
		if (P < 2) return 1;
		cudaStream_t s = g.s;
		unsigned int err = 0;
		bool ok = g.prow.alloc(P) && g.pcol.alloc(P) && g.perr.alloc(1)
			&& cudaMemcpyAsync(g.prow.p, prow, (size_t)P * 4, cudaMemcpyHostToDevice, s) == cudaSuccess
			&& cudaMemcpyAsync(g.pcol.p, pcol, (size_t)P * 4, cudaMemcpyHostToDevice, s) == cudaSuccess
			&& cudaMemsetAsync(g.perr.p, 0, 4, s) == cudaSuccess;
		if (!ok) return -1;
		setpos_kernel<<<(P + 255) / 256, 256, 0, s>>>(P, g.pcol.p, g.lvlpos.p, 0);
		lvlcheck_kernel<<<std::min<uint32_t>(P, 65535u), 128, 0, s>>>(P, g.prow.p, g.row_off.p, g.row_len.p,
			g.col[g.cur].p, g.lvlpos.p, g.perr.p);
		setpos_kernel<<<(P + 255) / 256, 256, 0, s>>>(P, g.pcol.p, g.lvlpos.p, 1);
		ok = cudaGetLastError() == cudaSuccess
			&& cudaMemcpyAsync(&err, g.perr.p, 4, cudaMemcpyDeviceToHost, s) == cudaSuccess
			&& cudaStreamSynchronize(s) == cudaSuccess;
		if (!ok) return -1;
		return err ? 0 : 1;
	}

	bool res_alive() {
		auto& g = g_res;
		if (fail_inject("lost")) return false; // testing: as if the device state were gone
		return g.on && cudaStreamSynchronize(g.s) == cudaSuccess && cudaGetLastError() == cudaSuccess;
	}

	const std::vector<uint32_t>& res_changed() { return g_res.changed; }

}
