# HIP collapse checks for QEM decimation

## Problem

On AMD GPUs, ComfyUI's `DecimateMesh` node spends most of its time in a slow fallback. QEM decimation checks every
candidate edge collapse for normal flips, skinny faces and the link condition. comfy-kitchen ships an exact kernel for
this (`edge_collapse_checks`) only in its CUDA backend, and ComfyUI's `qem_decimate._kitchen_op` asks only that backend.
On ROCm every iteration therefore takes the torch fallback `_quality_checks_fused`, which rebuilds a padded
vertex-to-face table each iteration and caps vertex degree at 16. Faces past that cap are skipped, so the fallback's
checks are also inexact on meshes with high-degree vertices.

In one measured `DecimateMesh` call of 44.9 s, the fallback took 31.7 s over 26 iterations: `_quality_checks_fused`
19.3 s and `_build_vert_to_faces_pad` 12.4 s.

## Who this concerns

- AMD GPUs on Windows with torch's ROCm wheels, running ComfyUI's mesh post-processing (`DecimateMesh`).
- `hipkernel.py` loads `hiprtc0702.dll` and `amdhip64_7.dll` through `ctypes.WinDLL`, so this code is Windows-only as
  written.
- ComfyUI v0.43.2 with comfy-kitchen 0.2.40. We know of no upstream release that ships a non-CUDA
  `edge_collapse_checks`; we did not check releases after these.

## What we changed

`lf_qem_hip/` is a ComfyUI custom node that adds no nodes.

- `qem_hip.py`: `edge_collapse_checks` as HIP kernels compiled at load time with hiprtc. Incident-face lists are built
  as compressed rows (atomic count, cumulative sum, atomic fill), with no sort and no degree cap. One thread per edge
  runs the same three checks as the fallback. An endpoint with more than 256 incident faces is refused outright
  (no collapse), which bounds the link check's per-thread work. Inputs are validated (shapes, device, index range)
  before any launch. On install it runs a self-test against the torch fallback and installs nothing on a mismatch; it
  then routes only `edge_collapse_checks` on the current GPU to the kernel.
- `hipkernel.py`: compile and launch a HIP kernel from Python with hiprtc and the HIP module API, no build system.
  Code objects are cached under `%LOCALAPPDATA%\lf-hiprtc` (override with `LF_HIPRTC_CACHE`).
- `edges_lowpeak.py`: QEM's sorted half-edge table (`_sorted_edge_halfedges`) without its (3F, 2) temporaries. Same
  three output tensors. It installs only when upstream's function source matches a SHA-256 pin and a CPU self-test is
  exact.

## Result

Measured 7 November 2026. Exactness is against the torch fallback with its degree cap raised to 64; speed is against the
fallback at its default cap of 16.

| Check | Result |
| --- | --- |
| Install self-test (closed torus grid plus a link violation) | 0 flip mismatches, 0 link mismatches, skinny difference 1.2e-7 |
| One call, 410K-triangle prop mesh, 635K edges, maximum vertex degree 58 | 0 flip mismatches, 0 link mismatches, skinny difference 3e-7 |
| One call, time | 6.7 ms against 78 ms: 11.8x |
| One call, a later re-run | exact, 20.6x |
| `qem_simplify` end to end, fallback then kernel in one process | 4.97x in one run, 3.87x in another |
| `qem_simplify` to 100,000 faces, one path per process | fallback 4.42 s, 95,658 faces, Chamfer to input 0.0106; kernel 3.60 s, 91,392 faces, Chamfer 0.0111; kernel with `AMD_SERIALIZE_KERNEL=3` 4.63 s |

The end-to-end speedup varied between runs; we have not explained the spread. Face counts differ
because the kernel's checks are exact where the fallback skips faces past degree 16.

Native-crash caveat: in the first full run, `qem_simplify` crashed natively (no traceback, exit code lost) about 25% into
the torch-fallback run, in a process that had already run the kernel. A code audit found no kernel cause; the only gap
found was unchecked indices, now validated. Two later runs of the same configuration and one run with each path in its
own process did not crash. The cause is not established.

`edges_lowpeak.py`: identical outputs to upstream on the CPU test meshes. Its peak-memory saving on the GPU was not
measured.

## How to run

CPU check (no GPU; needs torch, CPU build is enough, and a ComfyUI v0.43.2 source tree):

```
set COMFYUI_ROOT=<ComfyUI root>
python test_edges_lowpeak.py
```

It compiles upstream's `_sorted_edge_halfedges` from your install, checks the SHA-256 pin, and compares outputs. Expected:
`Ran 4 tests ... OK` (all skipped when `COMFYUI_ROOT` is not set).

GPU checks (need an AMD GPU and torch's ROCm wheels on Windows):

1. Copy `lf_qem_hip/` to `<ComfyUI root>/custom_nodes/` and restart ComfyUI with an empty queue. The log shows
   `[lf_qem_hip] installed: ... (flip mismatches, max skinny diff, link mismatches) = (0, <diff>, 0)` with `<diff>` at
   most 1e-4, and `[lf_qem_hip] edges installed: ...`. Undo: delete the folder and restart.
2. Benchmark on your own mesh, saved as an `.npz` with `pos` (V, 3) and `tri` (F, 3), with ComfyUI's embedded Python:

   ```
   set COMFYUI_ROOT=<ComfyUI root>
   python -s bench_qem_hip.py <mesh.npz> --target 100000
   ```

   It prints `QEMHIP {json}` lines: the self-test, one call (mismatches, maximum degree, both times), then
   `qem_simplify` with both paths. `--mode fallback` or `--mode kernel` runs one path per process.

## Measured on

Radeon RX 9070 XT (gfx1201, 16 GB), Adrenalin 32.0.31041.1004, Windows 11, i5-13600K. torch 2.9.1+rocm7.2.1 (TheRock
wheels, ROCm SDK 7.2.1: `hiprtc0702.dll`, `amdhip64_7.dll`), ComfyUI v0.43.2 on its embedded CPython 3.12.10,
comfy-kitchen 0.2.40. Full stack: [../docs/windows-rocm-stack.md](../docs/windows-rocm-stack.md).

## Not verified

- Any GPU other than gfx1201, and Linux.
- The cause of the native crash above.
- `edges_lowpeak.py` on the GPU, and its memory saving.
- ComfyUI versions after v0.43.2: `qem_hip.py` patches `_kitchen_op` without a source pin (its self-test still guards
  the result); `edges_lowpeak.py` refuses to install when upstream's source changes.
- Full-size job timings with the node installed.

## Upstream and licence

- ComfyUI v0.43.2 (GPL-3.0): `comfy_extras/mesh3d/postprocess/qem_decimate.py`, functions `_quality_checks_fused`,
  `_build_vert_to_faces_pad`, `_sorted_edge_halfedges`, `_kitchen_op`, `qem_simplify`. They are imported or, in the
  test, compiled from your install; none is copied here.
- comfy-kitchen 0.2.40: its CUDA backend's `edge_collapse_checks` is the op this replaces on ROCm.
- The HIP kernel computes the same checks with the same formulas and thresholds as `_quality_checks_fused` (the 1e-6
  usable-face test, shape quality `2*sqrt(3)*|n| / sum of squared edge lengths`, the link rule). It is our own
  implementation in HIP C++ with a different data layout and link-counting method; only the summation order of the
  skinny term differs. All files here are MIT.
