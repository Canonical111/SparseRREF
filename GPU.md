# GPU acceleration (CUDA cores) and memory

Everything here is optional: without `-DSPARSERREF_USE_CUDA` the library is the CPU code, and with
it the GPU is used only when the caller's policy asks for it (`opt->policy.gpu.enabled`, the CLI's
`--gpu` or `SPARSERREF_GPU=1`) and a device is present. Every GPU path
computes exactly what the CPU computes (same primes, same pivots, same modular arithmetic), so
outputs are identical byte for byte; this was checked by output hash on every change.

## Build

```bash
nvcc -O3 -std=c++17 -arch=sm_120 -c sparse_gpu.cu sparse_gpu_rr.cu sparse_gpu_res.cu   # sm_120: RTX 50xx
g++ -std=c++20 -O3 -march=native -DSPARSERREF_USE_CUDA main.cpp sparse_gpu.o sparse_gpu_rr.o sparse_gpu_res.o \
    -lflint -lmpfr -lgmp -L/usr/local/cuda/lib64 -lcudart -lpthread -o sparserref
./sparserref --gpu -t 16 -F QQ big.mtx
```

Add `-DSPARSERREF_PROFILE` for the phase timers and the memory timeline (`PROFILE ...` and
`PROFILE_MEM ...` lines).

## Rationals: the packed result

Over Q the result is built in packed form (`sparse_qpack.h`): per entry a 32-bit descriptor and the
limbs of numerator and denominator, in chunks of whole rows (about 64 chunks, 4096 to 4M entries
each). An fmpq whose parts exceed 62 bits costs two heap-allocated mpz of about 48 bytes each on top
of its 16 bytes; a packed entry costs its limbs and 4 bytes.

- A chunk keeps the residues of every prime until all its entries reconstruct, then only the values.
  A later prime checks the values (n = d r mod p) instead of storing its residues; a chunk whose
  values disagree goes back to residues (kept alongside the values while all residues fit
  `SPARSERREF_RR_KEEP_MB`, 512 MB, otherwise recomputed from the values). The fraction within the
  bounds is unique, so the stopping prime and every value are those of reconstructing every entry at
  every prime.
- The height check runs on the packed values (lcm with repeated denominators skipped; a product is
  formed only where its size bound can reach the maximum so far).
- `sparse_mat_rref_reconstruct` returns the rational matrix as before (built chunk by chunk from the
  packed form); `sparse_mat_rref_reconstruct_packed` returns the packed form, which the CLI writes to
  the text output directly (the kernel and WXF output still build the rational matrix).

### The reconstruction on the CPU

The per-entry arithmetic of the GPU's thread kernel (Garner's CRT, Wang's half extended Euclid with
FLINT's bounds, Lehmer steps that run up to the stopping point) is shared with the CPU through
`sparse_rr_core.h` (`unsigned __int128`; elsewhere the CPU keeps FLINT's functions). On the CPU it
first tries a common denominator D carried over the entries of a block of rows: with a the CRT value,
y = D a mod m in (-m/2, m/2) and g = gcd(y, D), the fraction (y/g)/(D/g) is FLINT's when both parts
are within the bounds (a fraction within them is unique). The residues are multiplied by D mod p_i
before the CRT, so this costs one CRT and a small gcd; D (at most three limbs) takes in the
denominator of every entry that needs the Euclidean path. In O2L12 99.7 % of the entries have a
denominator dividing that of the entries before them: the CRT and reconstruction took 21.7 s with
FLINT and 6.4 s with this (4 threads, M1 Max), 11.0 s and 3.3 s on an i9-13900K (8 threads).

## What runs on the GPU

| part | file | how |
|---|---|---|
| forward elimination (Z/p, and the first prime of Q) | `sparse_gpu_res.cu` | once the trailing matrix holds `SPARSERREF_GPU_MIN_NNZ` entries the rows move to the GPU and stay there; each round the GPU searches the pivots (method 0, see below), rescales the pivot rows and applies the Schur update; only the pivots come back, and the host's rows are filled once at the end (methods 1 and 2: the CPU searches on the structure, which comes back each round) |
| back substitution | `sparse_gpu_res.cu` | on the resident matrix, in batches of pivots from the end: the batch reduces itself, then the rows that hold its pivot columns (a pivot-column transpose built once) |
| replays for further primes (Q) | `sparse_gpu_res.cu` | the pivots are known: upload once, all levels and the back substitution there; the rows go straight into the residue store |
| CRT + rational reconstruction (Q) | `sparse_gpu_rr.cu` | Garner + Wang's half extended Euclid with FLINT's bounds and Lehmer steps, quotients beyond 62 bits taken in 61-bit pieces; one thread per entry up to 64 primes and 32 limbs, one warp per entry (limbs spread over the lanes, carries resolved by a lookahead across the warp) up to 510 limbs; results come back in the packed descriptor and are copied, not converted |
| Schur update of a row set (fallback) | `sparse_gpu.cu` | rows shipped per call in parts through two streams; used where the matrix is not resident |

Transfers between host and device go through two page-locked buffers of `SPARSERREF_GPU_PIECE`
entries (4M), so the host never holds a page-locked copy of the matrix.

### Failures

A failed GPU step in the forward elimination hands the rows back to the CPU, which redoes the step
(rescaling a rescaled row and updating an updated row change nothing, since an updated row holds no
pivot column of the round) and goes on from the same round. A failed back substitution leaves valid
rows, which the CPU completes. A failed replay or reconstruction leaves the host data untouched and
the CPU does that prime or the remaining chunks. Only when the device's state is lost does the
forward elimination start again on the CPU: from the rational input for Q (no copy kept), from a
copy of the input for Z/p (taken only when the GPU is enabled). In a build with `-DSPARSERREF_TEST_HOOKS`,
`SPARSERREF_GPU_FAIL=site:n[,...]`
injects failures for testing (sites `rescale`, `schur`, `backsub`, `lost`, `replay`, `rr`).

### Pivot search

The search of each round is a left look over the columns and a right look over the rows, both
greedy. What an item would take does not depend on the earlier choices, only whether it may (no
taken row holds the column; the row holds no taken column), so each look runs in chunks (256
doubling to an eighth of the items): marks record what the pivots taken so far block, the candidates
of a chunk's unblocked items are found in parallel, and a sequential pass takes those still
unblocked. The pivots are those of the sequential search, in the same order.

On the GPU (method 0, the default; `SPARSERREF_GPU_SEARCH=0` keeps it on the CPU) each round builds
the rows of each column (counting sort over the rows still to eliminate), the column counts, each
column's candidate row (one atomicMin over packed (entries, row) keys) and each row's candidate column
(a warp per row), and then runs the two sequential passes in one block of 1024 threads. A pass
decides 1024 items at a time in steps: `mark[x]` holds the smallest position of a taken item blocking
key `x` (a mark blocks only the items after it); the undecided unblocked items stamp the keys they
would block with their position (earliest wins); an item whose own key carries its own stamp cannot
be blocked by an earlier undecided item and is taken, as in the sequential order. When the stamps
would cost more than 16384 entries, only the earliest undecided item is taken in that step (its marks
usually block most of the rest). A failed search hands the rows back and the CPU rebuilds its column
lists.

### Pivot order

A level is applied in one pass and the back substitution goes from the last pivot backwards; both
need no pivot row to hold the column of an earlier pivot. The forward elimination produces such an
order. For a pivot list in another order, `backsub_order` finds one (Kahn's algorithm, keeping a
valid order as it is), on the CPU and on the GPU (a device check per level, reordered on the host
only when it fails). A level that cannot go in one pass (its pivot block not triangular in any
order, or pivot rows that get their entry only from other pivots, as when levels are merged) is
applied in batches: each the longest prefix of the level whose rows hold their pivot columns and no
column of an earlier pivot of the batch, applied to the rest of the level too. If a pivot entry
cancels on the way, the level starts again in reverse order, then with greedy batches, and
otherwise `std::invalid_argument` is thrown. The RREF for a given pivot set is unique, so the order
does not change the result.

## Policy fields

The tunables are fields of `exec_policy` / `gpu_policy` (`sparse_rref.h`); the library reads no
environment variable. The CLI maps its flags and environment variables onto them.

| field | default | CLI | meaning |
|---|---|---|---|
| `gpu.enabled` | false | `--gpu`, `SPARSERREF_GPU=1` | the GPU paths (CUDA builds) |
| `gpu.min_nnz` | 1048576 | `SPARSERREF_GPU_MIN_NNZ` | smallest work (entries) for the GPU; CUDA is not initialised below it |
| `gpu.forward`, `.search`, `.backsub`, `.replay`, `.reconstruct` | true | `SPARSERREF_GPU_FORWARD`, `_SEARCH`, `_BACKSUB`, `_REPLAY`, `_RR` (`0`) | that part on the GPU |
| `gpu.backsub_split` | 0 (max(rank/64, 2048)) | `SPARSERREF_GPU_BS_SPLIT` | back-substitution batch |
| `gpu.transfer_piece` | 4194304 (lean 1048576) | `SPARSERREF_GPU_PIECE` | entries per host-device transfer piece |
| `gpu.rr_pinned_mb` | 512 (lean 128) | `SPARSERREF_GPU_RR_PINNED_MB` | page-locked memory of the reconstruction's two slots |
| `gpu.call_min_rows`, `_min_pivots`, `_max_arena`, `_chunk`, `_large_u` | | `SPARSERREF_GPU_MIN_ROWS`, `_MIN_PIVOTS`, `_MAX_ARENA`, `_CHUNK`, `_LARGE` | the per-call Schur path |
| `rr_chunk` | 0 (nent/64 in [4096, 4M]) | `SPARSERREF_RR_CHUNK` | entries per chunk of the packed result |
| `rr_keep_mb` | 512 (lean 0) | `SPARSERREF_RR_KEEP_MB` | residues kept with the values while all of them fit |
| `trim_heap` | true (fast false) | `SPARSERREF_TRIM` | give freed heap pages back after rounds that shrank the matrix |
| `memory` | balanced | `--policy` | lean also trims the rows after the forward elimination; fast keeps one prime's matrix for the next |

The CLI alone also reads `SPARSERREF_GPU_WARMUP` (`1`: create the CUDA context while the input is
read, which saves its 0.15-0.2 s when the GPU is used and costs about as much and 100 MB when it is
not; `0`: never; default: inputs of 64 MB or more). Testing hooks, compiled only with
`-DSPARSERREF_TEST_HOOKS`: `SPARSERREF_GPU_FAIL` (above), `SPARSERREF_GPU_RES_CAP` (arena size in
entries, to test compaction), `SPARSERREF_GPU_RR_WARP=1` (the warp kernel for every size).

## Measured

Upstream `58ea093` against this version, built the same way (static FLINT 3.5, `g++ -O3 -march=native`, CUDA
13.0 for `sm_120`) on one machine: Intel i9-13900K, 12 threads, NVIDIA RTX 5080 (16 GB, PCIe 5.0 x8). Wall time
and peak resident set (`/usr/bin/time`), best of two runs. Every output is identical to upstream's.

| system | upstream | CPU build | CUDA build, `--gpu` |
|---|---|---|---|
| O2L12 (1.7e5 x 9.6e4, result 6.8e7 entries) over Q | 45.1 s, 23.0 GB | 13.8 s, 4.3 GB | 3.9 s, 4.5 GB |
| O2L12 over Z/p | 8.9 s, 3.0 GB | 6.2 s, 1.2 GB | 1.7 s, 1.2 GB |
| bench12 (result 1e6 entries) over Q | 1.53 s, 164 MB | 0.50 s, 145 MB | 0.50 s (CPU: too small) |
| bench12 over Z/p | 1.11 s, 106 MB | 0.41 s, 107 MB | 0.40 s (CPU: too small) |
| r600 (dense 600 x 601, 418 primes) over Q | 7.4 s, 264 MB | 4.1 s, 135 MB | 4.6 s |

Where the time went over Q, for O2L12 on the CPU: the rational result as `fmpq` and its reconstruction (now
packed limbs, the fixed-size CRT and the common denominator), the residues of every prime held at once (now
released chunk by chunk as values replace them), and the text writer (now a bounded window). On the GPU the
forward elimination of O2L12 over Z/p takes about 1 s: Schur updates 0.37 s, the pivot search 0.17 s, transfers
0.15 s, the back substitution 0.04 s, CUDA start-up and per-round bookkeeping the rest.

## Limits

- The Schur updates, the pivot search and the back substitution of the forward elimination run on
  the GPU; what stays on the host per round is bookkeeping (the lists of rows and columns still to
  eliminate) and a few synchronisations. The GPU search pays for itself on large rounds; for small
  matrices everything stays on the CPU (`SPARSERREF_GPU_MIN_NNZ`).
- The GPU reconstruction handles moduli up to 510 limbs (about 540 primes of 60 bits); beyond that
  the CPU does it, and also for entries that would overflow the kernel's limb capacity (none in the
  tests). The warp kernel pays only for large moduli with many entries; below 33 limbs the thread
  kernel is used.
- GPU paths need primes below 2^63 (the CLI already limits primes to that range).
- A pivot list whose levels cannot be applied in the given order, its reverse or greedily (a pivot
  entry cancels on the way) raises std::invalid_argument; the forward elimination never produces
  such levels.
