# Formation kernel: four models, one exact speedup task

## Problem

Ordering 100,000 agents into a shape cost 75.97 ms on one thread: 27.47 ms to lay the slots (`shape_slots`) and
48.50 ms to match agents to slots (`assign`). That is more than four 60 Hz frames spent in one step.

We gave the same task to four models: make both functions as fast as possible on one thread, with output bit-identical
to the original for every input. This folder holds everything needed to rerun the comparison: the task, the original,
the exactness checks, the four solutions, the build script and the benchmark.

## Who this concerns

- Anyone comparing models on low-level, exactness-bound C++ work.
- Anyone with a recursive median split built on `std::nth_element`, or a point-in-silhouette grid scan, on the hot path.
- Platform: x64, Windows, MSVC `/O2 /fp:precise`, C++20, no `/arch` flag, one thread. There is no upstream fix; the
  code is ours.

## What we changed

We wrote the task, the original kernel and the checks. The four models wrote the solutions:

| Folder | Model |
|---|---|
| `solutions/V/` | GPT-6.2 Astra |
| `solutions/N/` | GPT-6.2 Sol |
| `solutions/L/` | GPT-6.1 Sol |
| `solutions/A/` | GPT-6 Astra |

Each changed only `formation.cpp`. Every other file in each model's tree was byte-identical to the original's.

The common tricks, from the solutions' own comments:

- `shape_slots`: instead of testing every grid point against the silhouette, find each row's inside intervals from the
  outline's primitives, and decide every boundary point with the original rounded predicate, so points on an edge get
  the same answer. The even spread of n out of m points uses a running remainder (N, L, A) or precomputed ranges (V)
  instead of a division per point.
- `assign`: sort each axis once with stable radix passes into the original's (value, index) order, then split each node
  at its median rank and stably partition the other axis. This selects the same two sets as `nth_element` with the
  original comparator. In V, N and A, NaN inputs fall back to the original code path.
- V and L use SSE2 intrinsics (part of every x64 CPU), guarded by `_M_X64` or `__SSE2__`.

Comments in V and N say "see README": that is the model's own write-up, which is not included here.

## Result

Measured 18 Nov 2026 (`results/bench_run2.json:2`), 9 rounds (`:3`).

| Build | slots ms (x) | assign ms (x) | total ms (x) |
|---|---|---|---|
| original | 27.47 | 48.50 | 75.97 |
| V, GPT-6.2 Astra | 0.0665 (413x) | 2.885 (16.8x) | 2.95 (25.7x) |
| N, GPT-6.2 Sol | 0.3515 (78x) | 5.327 (9.1x) | 5.68 (13.4x) |
| L, GPT-6.1 Sol | 0.3058 (90x) | 7.037 (6.9x) | 7.34 (10.3x) |
| A, GPT-6 Astra | 0.5811 (47x) | 8.890 (5.5x) | 9.47 (8.0x) |

Method (`bench_all.py`): each build's `bench_formation` runs once per round, interleaved with the others, for 9 rounds.
Each process reports, per benchmark, the median of 21 samples (each sample runs at least 2 ms of calls, after at least
50 ms of warm-up, on a thread pinned to one performance core). A build's figure is the median of its 9 per-process
medians. The total is the slots median plus the assign median; each speedup is the original's time over the build's.
We recomputed every figure above from the per-round values in `results/bench_run2.json`.

Exactness, beside the speed:

- All five builds return the same benchmark checksums in every round (`00000000000186a0` for slots,
  `00000000001af566` for assign; `results/bench_run2.json`, `"checksums"`).
- Every solution passes the unit tests, `check/exact_formation.cpp` (438 fixed cases, 0 mismatches) and
  `check/exact_wide.cpp` (1,518 random cases with ties, signed zeros, subnormals, infinities and aliased inputs,
  0 mismatches). The models saw only the first check. Both programs print their own case count: 360 slot cases, 77
  assign cases and 1 length-mismatch case make 438; 600 slot cases, 900 assign cases and 18 self-aliased assign cases
  make 1,518.
- NaN inputs are left out of both checks. The original's comparator is not a strict weak order with NaN, so its result
  is not defined (`check/exact_wide.cpp:3-4`).

## How to run

Prerequisites (Windows x64):

- Visual Studio 2022 or Build Tools for Visual Studio 2022, with the "Desktop development with C++" workload (the x64
  C++ tools). `kbuild.py` finds it with `vswhere.exe`, so `cl` need not be on PATH.
- Python 3.8 or later. No packages.

Build, test and check one build (about a minute):

```
py -3 kbuild.py original
py -3 kbuild.py solutions/V
```

Expected output, ending (with MSVC it starts with a `kbuild: MSVC tools <version>` line):

```
PASS slots_fill_a_shape_with_exactly_n
  mean walk: rank split <a> cm, shuffled <b> cm
PASS rank_split_is_a_bijection_and_far_better_than_shuffled
2 tests, 0 failed
exact_formation: 438 cases, 0 mismatches
exact_wide: 1518 cases, 0 mismatches
kbuild: tests and exactness pass
```

Bench one build: `py -3 kbuild.py solutions/V --bench --runs 5`. Bench all five side by side, as in the table:
`py -3 bench_all.py 9` (writes `bench_all.json`). Close other heavy programs first.

Try your own solution: copy `original/formation.cpp` into a new folder, change it, and run `py -3 kbuild.py <folder>`.
A `formation.h` in that folder is used instead of the original's.

g++ or clang (untested; we have not built it this way):

```
python3 kbuild.py original --cxx g++
python3 bench_all.py 9 --cxx clang++
```

This compiles with `-std=c++20 -O2 -DNDEBUG`. ISO mode keeps FMA contraction and fast-math off, but the solutions were
written and checked against MSVC only. Outside Windows the benchmark thread is not pinned.

## Files

- `TASK.md`: the task as the models got it, with our paths, logging and harness instructions removed.
- `original/`: the original `formation.h` and `formation.cpp`. They need only the C++ standard library.
- `reference/`: the copy of the original that the checks compare against (renamed into `namespace ref_formation` at
  build time). It is identical to `original/`.
- `solutions/{V,N,L,A}/formation.cpp`: the four deliveries.
- `check/`: the two exactness checks. Their inputs are generated in the source (fixed lists and seeded generators).
- `tests/`, `include/`: the unit test, the benchmark and their header-only harnesses.
- `kbuild.py`, `bench_all.py`: build and benchmark scripts.
- `results/bench_run2.json`: the measured run behind the table, with two host-status lines removed.

We edited comments only in the C++ sources (lab references), and trimmed the test and benchmark files (below). We made
no code change to any `formation.cpp`. In `include/lumen_bench.h` we added one `#include <x86intrin.h>` for g++ and
clang, outside the MSVC path.

## Measured on

- CPU: Intel Core i5-13600K (6 P-cores and 8 E-cores); the benchmark thread pinned to one P-core at high priority.
- OS: Windows 11.
- Compiler: MSVC, x64, `/O2 /fp:precise /std:c++20` (full flags in `kbuild.py`). The MSVC version was not recorded;
  `kbuild.py` now prints it.
- Python 3; the exact version was not recorded.

## Not verified

- The files in this folder were not compiled after packaging: no C++ compiler was on PATH. The changes since the
  measured build are comment edits, the trimmed test and benchmark, the added g++ include and the rewritten build
  scripts. Run `py -3 kbuild.py original` first.
- The original test file had a third test (a swarm pours into the dragon), and the benchmark file two swarm-step
  benchmarks that ran after the two measured ones. All three need our swarm core and are not included. The two
  measured benchmarks are unchanged and ran first in the same process.
- One run per model. A second run of the same model could change close results (L's slots path is faster than N's).
- A GPU render ran on the same machine during the measured run. All builds were interleaved, so it affected them alike,
  but it was not a quiet machine.
- No g++, clang, sanitizer or 32-bit build. NaN inputs are not checked.
