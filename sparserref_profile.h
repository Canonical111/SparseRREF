/*
	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// Optional phase timers (compile with -DSPARSERREF_PROFILE); without the macro every hook is a no-op.
#pragma once
#ifdef SPARSERREF_PROFILE
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#endif

namespace SparseRREF::prof {
	struct round_t { size_t pivots, leftrows, nnz; double search, schur; size_t rss_kb = 0; };
	struct data_t {
		double search = 0, rescale = 0, schur = 0, forward = 0, trim = 0, tp_rebuild = 0;
		double backsub = 0, bs_schur = 0, bs_tri = 0, bs_transpose = 0;
		double reduce = 0, rr_first = 0, height = 0, replay = 0, crt = 0;
		size_t rounds = 0, primes = 0, bs_batches = 0, bs_schur_rows = 0;
		double gpu_pack = 0, gpu_h2d = 0, gpu_kernel = 0, gpu_d2h = 0, gpu_unpack = 0, gpu_wait = 0;
		size_t gpu_calls = 0, gpu_redo = 0, gpu_rows = 0, gpu_skip = 0, gpu_in = 0;
		double t_check = 0, t_store = 0, t_comb = 0, t_sample = 0, t_full = 0;
		size_t n_full = 0, gpu_rr_cpu = 0, n_verify = 0, n_unpack = 0, n_packed = 0;
		double t_verify = 0, t_unpack = 0;
		double gpu_rr = 0, gpu_rr_convert = 0, gpu_rr_stage = 0, gpu_rr_wait = 0, gpu_rr_h2d = 0, gpu_rr_kernel = 0, gpu_rr_d2h = 0;
		double gpu_res = 0, gpu_res_backsub = 0, gpu_res_rescale = 0, gpu_res_prep = 0, gpu_res_kernel = 0, gpu_res_compact = 0, gpu_res_xfer = 0, gpu_res_writeback = 0;
		size_t gpu_res_compactions = 0, gpu_res_calls = 0, gpu_level_reorders = 0, level_batches = 0, level_retries = 0;
		double gpu_fwd_schur = 0, gpu_fwd_xfer = 0, gpu_fwd_scatter = 0, gpu_fwd_backsub = 0, gpu_fwd_compact = 0;
		size_t gpu_fwd_rounds = 0, gpu_fwd_compactions = 0, gpu_fwd_recoveries = 0;
		double gpu_search_count = 0, gpu_search_cand = 0, gpu_search_greedy = 0;
		std::vector<round_t> fwd;
		struct mem_t { const char* label; double t; size_t rss_kb, hwm_kb; };
		std::vector<mem_t> mem;
		size_t res_ent = 0, res_hist[4][4] = {}, res_same_den = 0, res_limbs = 0;
	};
	inline data_t& d() { static data_t x; return x; }
	using clk = std::chrono::steady_clock;
	inline double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
	inline clk::time_point t_origin() { static const clk::time_point t = clk::now(); return t; }
	// resident set and its peak so far (kB; Linux and macOS)
	inline void rss_hwm(size_t& rss, size_t& hwm) {
		rss = 0; hwm = 0;
#ifdef __linux__
		if (FILE* f = std::fopen("/proc/self/status", "r")) {
			char line[256];
			while (std::fgets(line, sizeof(line), f)) {
				if (!std::strncmp(line, "VmRSS:", 6)) rss = std::strtoull(line + 6, nullptr, 10);
				else if (!std::strncmp(line, "VmHWM:", 6)) hwm = std::strtoull(line + 6, nullptr, 10);
			}
			std::fclose(f);
		}
#elif defined(__APPLE__)
		mach_task_basic_info_data_t info;
		mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
		if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &n) == KERN_SUCCESS)
			rss = (size_t)(info.resident_size >> 10);
		struct rusage ru;
		if (getrusage(RUSAGE_SELF, &ru) == 0)
			hwm = (size_t)ru.ru_maxrss >> 10; // bytes on macOS
#endif
	}
	inline size_t rss_kb() { size_t r, h; rss_hwm(r, h); return r; }
	// records a labelled point of the memory timeline
	inline void mem_mark(const char* label) {
		size_t rss, hwm;
		rss_hwm(rss, hwm);
		d().mem.push_back({ label, since(t_origin()), rss, hwm });
	}

	inline void report(FILE* f) {
		auto& p = d();
		std::fprintf(f, "PROFILE forward %.3f s (pivot search %.3f, rescale %.3f, schur %.3f of which the search's column lists %.3f, heap trim %.3f; %zu rounds)\n",
			p.forward, p.search, p.rescale, p.schur, p.tp_rebuild, p.trim, p.rounds);
		std::fprintf(f, "PROFILE backsub %.3f s (transpose %.3f, batched schur %.3f over %zu batches / %zu rows, per-column %.3f)\n",
			p.backsub, p.bs_transpose, p.bs_schur, p.bs_batches, p.bs_schur_rows, p.bs_tri);
		std::fprintf(f, "PROFILE reconstruct: primes %zu, reduce %.3f, first rational reconstruction %.3f, height checks %.3f, replay %.3f, crt+rr %.3f\n",
			p.primes, p.reduce, p.rr_first, p.height, p.replay, p.crt);
		std::fprintf(f, "PROFILE crt detail: support check %.3f, store %.3f (verify %.3f in %zu, unpack %.3f in %zu), comb %.3f, sample %.3f, full %.3f (%zu full attempts, %zu chunks packed)\n",
			p.t_check, p.t_store, p.t_verify, p.n_verify, p.t_unpack, p.n_unpack, p.t_comb, p.t_sample, p.t_full, p.n_full, p.n_packed);
		std::fprintf(f, "PROFILE gpu rr: %.3f s in all (staging %.3f, waiting %.3f, conversion %.3f; on the gpu h2d %.3f, kernels %.3f, d2h %.3f), %zu entries left to the cpu\n",
			p.gpu_rr, p.gpu_rr_stage, p.gpu_rr_wait, p.gpu_rr_convert, p.gpu_rr_h2d, p.gpu_rr_kernel, p.gpu_rr_d2h, p.gpu_rr_cpu);
		std::fprintf(f, "PROFILE gpu: %zu parts, host pack %.3f, wait %.3f, unpack %.3f; on the gpu h2d %.3f, kernel %.3f, d2h %.3f; rows sent %zu, skipped %zu, rerun %zu, entries in %zu\n",
			p.gpu_calls, p.gpu_pack, p.gpu_wait, p.gpu_unpack, p.gpu_h2d, p.gpu_kernel, p.gpu_d2h, p.gpu_rows, p.gpu_skip, p.gpu_redo, p.gpu_in);
		std::fprintf(f, "PROFILE gpu replay: %zu replays, %.3f s in all (rescale %.3f, level prep %.3f, kernels %.3f, compaction %.3f in %zu, transfers %.3f, write back %.3f; back substitution %.3f of it), %zu levels reordered\n",
			p.gpu_res_calls, p.gpu_res, p.gpu_res_rescale, p.gpu_res_prep, p.gpu_res_kernel, p.gpu_res_compact, p.gpu_res_compactions, p.gpu_res_xfer, p.gpu_res_writeback, p.gpu_res_backsub, p.gpu_level_reorders);
		if (p.level_batches)
			std::fprintf(f, "PROFILE levels not triangular in any order: %zu batches, %zu retries reversed\n", p.level_batches, p.level_retries);
		std::fprintf(f, "PROFILE gpu forward: %zu rounds, schur %.3f, transfers %.3f, back substitution %.3f, compaction %.3f in %zu, %zu recoveries\n",
			p.gpu_fwd_rounds, p.gpu_fwd_schur, p.gpu_fwd_xfer, p.gpu_fwd_backsub, p.gpu_fwd_compact, p.gpu_fwd_compactions, p.gpu_fwd_recoveries);
		std::fprintf(f, "PROFILE gpu search: counts and scan %.3f, lists and candidates %.3f, sequential passes %.3f\n",
			p.gpu_search_count, p.gpu_search_cand, p.gpu_search_greedy);
		if (p.res_ent) {
			std::fprintf(f, "PROFILE result: %zu entries, %zu limbs, den same as previous in row %zu; limbs (num x den, 0..2, 3+):", p.res_ent, p.res_limbs, p.res_same_den);
			for (int a = 0; a < 4; a++)
				for (int b = 0; b < 4; b++)
					std::fprintf(f, " %d%d:%zu", a, b, p.res_hist[a][b]);
			std::fprintf(f, "\n");
		}
		std::fprintf(f, "PROFILE_ROUNDS round pivots leftrows nnz_after search_s schur_s rss_MB\n");
		for (size_t i = 0; i < p.fwd.size(); i++) {
			auto& r = p.fwd[i];
			std::fprintf(f, "PROFILE_ROUND %zu %zu %zu %zu %.6f %.6f %.1f\n", i, r.pivots, r.leftrows, r.nnz, r.search, r.schur, r.rss_kb / 1024.0);
		}
	}
	inline void report_mem(FILE* f) {
		for (auto& m : d().mem)
			std::fprintf(f, "PROFILE_MEM %-28s t %8.3f rss %8.1f MB peak %8.1f MB\n", m.label, m.t, m.rss_kb / 1024.0, m.hwm_kb / 1024.0);
	}
}
#define SPR_T0(name) auto name = SparseRREF::prof::clk::now()
#define SPR_ADD(field, t0) (SparseRREF::prof::d().field += SparseRREF::prof::since(t0))
#define SPR_DO(stmt) stmt
#define SPR_MEM(label) SparseRREF::prof::mem_mark(label)
#else
#define SPR_T0(name) ((void)0)
#define SPR_ADD(field, t0) ((void)0)
#define SPR_DO(stmt) ((void)0)
#define SPR_MEM(label) ((void)0)
#endif
