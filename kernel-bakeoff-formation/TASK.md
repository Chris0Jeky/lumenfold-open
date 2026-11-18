# Task: make a hot kernel faster, exactly

This is the task each model was given, with our machine paths, logging and harness instructions removed. The rules are
unchanged. Each model worked alone, in a fresh context, with a 90-minute cap.

`lumen::formation` is one engine-free C++20 core of a game codebase (`formation.h` and `formation.cpp`; read the
header's comment first), with its test (`tests/test_formation.cpp`) and its benchmark (`tests/bench_formation.cpp`).
Two of its benchmarks, at 100,000 agents, are what the game pays when a swarm is ordered into a shape:

- `formation.slots.dragon.n100000`: `shape_slots`, laying exactly n slots on a grid inside a silhouette;
- `formation.assign.n100000`: `assign`, the hierarchical rank-split matching of agents to slots.

Make both as fast as you can, under these rules:

- **Exact.** For every input, both functions must give bit-identical results to the original: the same return value,
  the same `sx`, `sy` floats bit for bit, the same `slot_of` permutation, the same exception on a length mismatch.
  `check/exact_formation.cpp` checks this against a renamed copy of the original over a fixed set of inputs. A wider
  check with inputs you do not see (`check/exact_wide.cpp`) also runs, so exactness must hold by construction, not only
  on that set.
- **One thread.** No threads, thread pools, OpenMP or parallel algorithms. SIMD intrinsics are allowed if the default
  x64 MSVC build (no `/arch` flag) runs them on any x64 CPU.
- **Same build.** The compiler flags are fixed (MSVC, `/O2 /fp:precise`, C++20; see `kbuild.py`). No pragmas or
  attributes that change floating-point semantics. No new dependencies.
- **Same interface.** Change only `formation.cpp`, and `formation.h` only in comments or private details that keep
  every caller's source and behaviour the same. The test file must still pass unchanged.
- Keep the code readable: a short comment says what each trick does and why it is exact.

Build, test, check and bench (about a minute):

```
py -3 kbuild.py <your folder> [--bench] [--runs 5]
```

It compiles with the fixed flags, runs the test and the exactness checks, and with `--bench` runs the benchmark
`--runs` times and prints each benchmark's median. `py -3 kbuild.py original --bench` measures the original.

Deliver the changed `formation.cpp` (and `formation.h` if changed) and a short write-up: what you changed and why each
change is exact, a before-and-after table from your own `--bench` runs, what you tried that did not pay, and where the
remaining time goes.

Differences from the task as given: the models never saw `exact_wide.cpp`; we ran it on each delivery after the task. The test file they ran also held a third test (a swarm pours into the dragon), and the
benchmark file two swarm-step benchmarks that run after the two above; both need our swarm core and are left out here.
