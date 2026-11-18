# Fast mesh UV unwrap for ComfyUI

CPU numbers; GPU job not verified.

## Problem

ComfyUI's `UnwrapMesh` node takes minutes on a TRELLIS mesh and holds up the whole job while the GPU idles. In two
measured ComfyUI v0.43.2 jobs it took 268 s of a 470 s job and 177 s of a 254 s job, all on the CPU.

Causes, all in ComfyUI's `comfy_extras/mesh3d/uv_unwrap`:

- `parameterize.lscm_charts_batch` solves thousands of small charts as dense padded normal equations (chunks of up to
  128 charts, forced to the CPU on Windows ROCm). Each chart's least-squares problem has about 6 non-zeros per row.
- `segment.cluster_charts_pec` and `_merge_small_charts` rebuild the chart adjacency from all 3F face adjacencies on
  every merge iteration.
- `MeshSmoothNormals` with a crease angle (`nodes_mesh_postprocess._compute_vertex_face_normals`) runs per-face Python
  loops.

## Who this concerns

- Anyone running ComfyUI v0.43.2's `UnwrapMesh` or `MeshSmoothNormals` (crease path) on large meshes. The work is on
  the CPU on every platform; on Windows ROCm the dense LSCM path is forced to the CPU.
- Written against ComfyUI v0.43.2. Each patch checks itself against upstream at install and installs nothing on a
  mismatch. We know of no upstream release that changes these functions; we did not check releases after v0.43.2.

## What we changed

`lf_unwrap_fast/` is a ComfyUI custom node that adds no nodes. On load it patches upstream functions in place:

- `unwrap_fast.py`, `lscm_charts_sparse`: the same LSCM rows (ABF for valid faces, plain LSCM for degenerate ones), the
  same two farthest-point pins per chart and the same regularisation, solved as one block-diagonal sparse system with
  SuperLU, in groups of 2048 charts to bound memory. Charts above upstream's 256-vertex limit stay on upstream's
  per-chart path. Any failure falls back to upstream's function.
- `unwrap_fast.py`, `cluster_charts_pec_fast` and `merge_small_charts_fast`: the chart-pair list is built once and
  mapped through each merge's remap instead of rebuilt from the faces.
- `batch_fast.py`: PEC clustering that computes only the cone cost for every edge (axis, sine and cosine only for
  winning edges), tracks live chart labels on the host, remaps edges with a sort and adjacent-duplicate removal, and
  extracts boundary loops by scanning only boundary edges. A runtime error runs upstream's whole clustering path.
- `normals_fast.py`: the crease-normal path vectorised (edge keys, a sparse connected-components pass, one output
  vertex per original vertex and smoothing group). Bit-identical to upstream.

Equivalence rule for the LSCM solve: identical chart labels and topology, and chart UV error below 1e-5 of each chart's
span (about 84 float32 ulps, for sparse against dense elimination order). PEC clustering and normals must be identical.

## Result

CPU, i5-13600K, 472,523-face TRELLIS mesh, four torch threads, warmed numba packer, node defaults (PEC, 1024
resolution, padding 1, automatic welding).

| Run (8 November 2026) | Upstream | Original patch | This version |
| --- | ---: | ---: | ---: |
| Three-way run (with a large-chart batching step since dropped) | 180.97 s | 20.03 s | 17.59 s |
| Paired run, current code | not run | 16.48 s | 15.38 s |

- Three-way run: all 25,954 chart assignments matched upstream; maximum chart-relative UV error 1.39e-8 against
  upstream.
- Paired run: every equivalence delta 0.0 between the two patched versions (chart assignment, topology, chart UVs,
  integer layout, atlas 955x940, UV area, stretch).
- PEC incremental adjacency alone (7 November 2026): identical chart assignment on 5 meshes; 2.9x to 3.8x on a
  194K-face mesh (4.16 s to 1.09 s); two deliberate mutants were caught by the equality check.
- Crease normals (7 November 2026): 4.5x on 319K faces (1.20 s to 0.26 s), bit-identical.

## How to run

Prerequisites: a ComfyUI v0.43.2 source tree and its Python with torch (a CPU build is enough), numpy, scipy and
trimesh. numba is optional; upstream's atlas packer uses it when installed, and our timings used it. No GPU is needed:
the benchmark and test hide all GPUs.

Tests (CPU only):

```
set COMFYUI_ROOT=<ComfyUI root>
python -s test_unwrap_fast.py -v
```

Expected: every test passes (`OK`); they skip when torch or the ComfyUI source is missing.

Benchmark on your own mesh:

```
set COMFYUI_ROOT=<ComfyUI root>
python -s lf_unwrap_fast/bench_unwrap.py <mesh.glb> --output lf_unwrap_fast/_bench_results/run.json
```

The harness does not ship or copy upstream code. It loads `uv_unwrap` from `COMFYUI_ROOT` and compiles
`_uv_weld_vertices` and `_uv_unwrap` from your installed `comfy_extras/nodes_mesh_postprocess.py` unchanged. It
replaces only progress reporting and the packer's device provider with CPU stand-ins, so no ComfyUI server or
model-management module is imported. It runs upstream, the original patch (`unwrap_fast`), then the current version
(`unwrap_fast` plus `batch_fast`), prints one JSON line per mode and an equivalence report, and exits 0 when every
equivalence check passes. `--modes current candidate` runs the shorter paired comparison; `--profile` prints CPU
profiles. Generated files and the numba cache stay in `lf_unwrap_fast/_bench_results/` (gitignored).

Install in ComfyUI: copy `lf_unwrap_fast/` to `<ComfyUI root>/custom_nodes/` and restart with an empty queue. The log
shows `[lf_unwrap_fast] sparse batched LSCM installed, self-test max rel diff ...`, the PEC line and the normals line.
Undo: delete the folder and restart.

## Measured on

i5-13600K (6 P-cores, 8 E-cores), 32 GB RAM, Windows 11. ComfyUI v0.43.2 on its embedded CPython 3.12.10, torch
2.9.1+rocm7.2.1 used on the CPU, numpy 2.5.2, scipy 1.18.1, numba 0.68.1, trimesh 5.1.0. The machine's GPU (Radeon
RX 9070 XT, Adrenalin 32.0.31041.1004) was hidden. Full stack:
[../docs/windows-rocm-stack.md](../docs/windows-rocm-stack.md).

## Not verified

- GPU tensors and a full ComfyUI job: these CPU timings do not predict a GPU job's wall time.
- Production job timings with the node installed.
- Other meshes at full scale: the 472,523-face numbers are from one mesh.
- ComfyUI versions after v0.43.2. There is no source-hash pin; protection is the `lscm_charts_batch` signature check
  and the install self-tests.
- `test_unwrap_fast.py` was not re-run on this copy. It covers the LSCM solve, PEC clustering, boundary loops, edge
  remap and the install order; the crease-normal path is checked only by its install self-test against upstream.

## Upstream and licence

- ComfyUI v0.43.2 (GPL-3.0): `comfy_extras/mesh3d/uv_unwrap/{mesh,segment,parameterize,pack}.py` and
  `comfy_extras/nodes_mesh_postprocess.py`.
- `lf_unwrap_fast/unwrap_fast.py` and `lf_unwrap_fast/batch_fast.py` contain modified copies of, or code adapted from,
  ComfyUI v0.43.2 functions (named at the top of each file), so those two files are GPL-3.0
  (`lf_unwrap_fast/LICENSE-GPL-3.0.txt`). The other files here are MIT.
- numba 0.68.1 (BSD), numpy 2.5.2, scipy 1.18.1, trimesh 5.1.0.
