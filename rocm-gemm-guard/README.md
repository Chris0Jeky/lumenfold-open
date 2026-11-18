# ROCm GEMM guard (hip_gemm_chunk)

**Fixed upstream in torch 2.11.0+rocm7.13.0.** On the same card and machine it gives 0 wrong rows at 8,388,609 fp32
rows (2.9.1+rocm7.2.1 gives 7.86 million), and bmm is correct. **Still needed** on gfx12 GPUs (RDNA4: gfx1200, gfx1201)
with ROCm 7.2.x: we measured torch 2.9.1+rocm7.2.1, and ROCm 7.2.4 with torch 2.14.1+rocm7.2 is reported to still
reproduce. To check your own stack, run `python probe_quick.py` (GPU, about 6 GB of VRAM). Versions and sources:
[../docs/windows-rocm-stack.md](../docs/windows-rocm-stack.md).

## Problem

Large matrix multiplies on the GPU return wrong values with no error and no warning. On an RX 9070 XT with torch
2.9.1+rocm7.2.1:

- `F.linear`, `matmul` and `mm` in fp32 are wrong from output row 524,288 (2^19): errors of 30-260%, against 1e-7
  below that row.
- In fp16 and bf16 the limit depends on the kernel: from row 1,048,576 at K = N = 16, from row 4,194,304 at K = 64,
  N = 32.
- `bmm` is wrong from batch index 65,536 in fp32, bf16 and fp16.
- `addmm` with a 2-D bias (beta 1) past the row limit is wrong in every row, from row 0 (measured at 1,048,576 and
  1,048,577 rows): the wrapped tiles add into the first rows a second time.
- Batched 3x3 `linalg.solve` past the batch limit fails with "HIP error: invalid configuration argument".

Mesh decoders such as TRELLIS.2 run linear layers over millions of voxel rows, so their meshes and textures come out
broken.

## Who this concerns

- GPU: gfx12 (RDNA4). The faulty kernels are hipBLASLt's gfx12 GEMM kernels. We measured gfx1201 only.
- OS: we measured Windows 11. The bug is in hipBLASLt, not in Windows; an upstream bmm report reproduced it on WSL2.
- Bad: torch 2.9.1+rocm7.2.1 (measured); any ROCm 7.2.x release without the backport
  https://github.com/ROCm/rocm-libraries/pull/15172, which was open on 11 Nov 2026.
- Fixed: torch 2.11.0+rocm7.13.0 (measured). Fix: https://github.com/ROCm/rocm-libraries/pull/2681.
- On a fixed stack, do not install the guard: it fixes nothing there and costs speed (see Result).

## Mechanism

hipBLASLt's gfx12 kernels launch a 2-D or 3-D grid, and gfx12 reads the Y and Z workgroup IDs from 16-bit fields. A
launch with more than 65,536 workgroups in Y or Z wraps: workgroup 65,536 runs as workgroup 0, so its output tile is
never written and tile 0 is written twice (https://github.com/ROCm/TheRock/issues/9270). Rows map to Y, so **the first
wrong row is exactly 65,536 x the selected kernel's row tile**. Batches map to Z, so the first wrong batch is exactly
65,536. The threshold is not a byte or element count: a 4.5 GiB fp32 operand under the row limit is exact.

Measured with `gemm_boundary.py`, unguarded `F.linear` (subset of the sweep):

| dtype | K | N | first wrong row | row tile |
|---|---|---|---|---|
| fp32 | 64 | 32 | 524,288 | 8 |
| fp32 | 3 | 3 | 524,288 | 8 |
| fp32 | 256 | 256 | 1,048,576 | 16 |
| fp32 | 128 | 128 | 1,048,576 | 16 |
| bf16, fp16 | 64 | 32 | 4,194,304 | 64 |
| bf16, fp16 | 16 | 16 | 1,048,576 | 16 |
| bf16 | 1024 | 1024 | none up to 1,572,864 rows | 32 or more |

## What we changed

`hip_gemm_chunk/__init__.py` wraps the Python entry points `F.linear`, `torch.matmul`, `torch.mm`, `torch.addmm`,
`torch.bmm`, `torch.baddbmm`, `Tensor.bmm`, `Tensor.__matmul__`, batched `torch.einsum`, `torch.linalg.solve`,
`torch.det` and `torch.linalg.det`, and splits any GPU call that would cross the limits:

- fp32 rows in chunks of 2^18 (2^19 would be exact for an 8-row tile, but was slower; see Result);
- fp16 and bf16 rows unchunked up to 2^20 (65,536 x their smallest measured row tile, 16), chunked beyond;
- batches, including broadcast and multi-dimensional batches, in pieces under 2^15;
- an N-D input to a row GEMM is flattened to 2-D rows, so it never runs as an internal bmm with a large batch.

At import it installs these wrappers and checks itself against the CPU just past each limit. It does nothing on a
build without ROCm or without a GPU. Inside ComfyUI it also sets the TRELLIS.2 shape and texture VAEs to bf16
(`LF_TRELLIS_VAE_FP32=1` keeps fp32); fp32 GEMMs at these sizes ran at 1-3 TFLOPS against about 100 for bf16.

Not covered: direct `torch.ops.aten` calls, C++ extensions, and code that bound `torch.matmul` (or another wrapped
function) to its own name before the guard was imported.

## Result

- Mechanism (11 Nov 2026): reference = the same op on 2^16-row slices on the GPU, itself checked against the CPU in
  fp64 on sampled rows (max relative error 5e-8 to 3e-3). In every configuration the first wrong row is exactly
  65,536 x the row tile and every row below it is right. At the fp32 edge, 524,287 and 524,288 rows are right, 524,289
  rows give exactly one wrong row (row 524,288), 524,296 give 8 and 524,297 give 9.
- `probe_quick.py` output, (first wrong row or batch, number wrong), same machine:

  | case | 2.9.1+rocm7.2.1 | 2.11.0+rocm7.13.0 |
  |---|---|---|
  | linear fp32 M=524297 K=64 N=32 | (524288, 9) | (None, 0) |
  | linear bf16 M=1048585 K=16 N=16 | (1048576, 9) | (None, 0) |
  | linear fp16 M=1048585 K=16 N=16 | (1048576, 9) | (None, 0) |
  | addmm 2-D bias M=2^20 | (0, 1048576) | (None, 0) |
  | bmm fp32 batch 65537 | (65536, 1) | (None, 0) |
  | bmm bf16 batch 65537 | (65536, 1) | (None, 0) |

- Guard self-test at load (14 Nov 2026, inside ComfyUI): max relative error 1.89e-04 against the CPU over linear,
  bmm, `@`, einsum, solve and det just past the limits, plus a bf16 K = N = 16 case. A wrapped tile gives errors of
  order 1. The module logs this at warning level, because its warning threshold is 1e-4.
- Cost (11 Nov 2026, fp32 `F.linear`, 4,194,304 x 256 x 256, median of 5): the unguarded single call takes 162 ms
  and is wrong; the guard's 2^18-row chunks take 720 ms; 2^19-row chunks took 1,200 ms; one transposed call (rows on
  the 32-bit X axis) took 1,348 ms. Chunking bf16 rows cost 25-45%, which is why half precision passes through
  unchunked below its own limit.

## How to run

Prerequisites: Windows 11, an RDNA4 GPU with AMD's Adrenalin driver, Python 3.12 with a ROCm build of torch (in the
ComfyUI Windows portable build, use its embedded Python).

1. Check your stack, without the guard loaded:

   ```
   python probe_quick.py
   ```

   A broken stack prints rows like the 2.9.1+rocm7.2.1 column above; a fixed one prints `(None, 0)` on every line.

2. Install as a ComfyUI custom node: copy `hip_gemm_chunk/` to `<ComfyUI root>/custom_nodes/hip_gemm_chunk/` and
   restart ComfyUI. The log shows `[hip_gemm_chunk] large-GEMM guard installed, self-test max rel err ...`. Delete
   the folder and restart to undo.

3. Or use it from plain torch: import it once, before the workload.

   ```python
   import sys
   sys.path.insert(0, r"<path to rocm-gemm-guard>")
   import hip_gemm_chunk  # on a ROCm GPU build: patches torch and runs the self-test; elsewhere: does nothing
   ```

   Outside ComfyUI it also prints `TRELLIS.2 VAE dtype patch not applied: No module named 'comfy'`, which is harmless.
   Import it only once; a second `install()` would wrap the wrappers.

4. Checks:

   | Command | Needs | Expected |
   |---|---|---|
   | `python test_cpu.py` | torch (a CPU build is enough), no GPU | `OK` |
   | `python test_guard.py` | ROCm GPU | every line `ok`, then `selftest <error>`, exit 0 |
   | `python solve_probe.py 200000` | ROCm GPU | a `raw ok` or `raw FAIL` line, then a `guarded err` line |
   | `python gemm_boundary.py` | ROCm GPU, 16 GB | JSON lines; `first_wrong` per case, `null` where the stack is right |
   | `python gemm_boundary.py guard` | ROCm GPU, 16 GB | JSON lines; `"n_wrong": 0` for every `guard_case` |

All checks except `test_cpu.py` need the GPU. `gemm_boundary.py` and `probe_quick.py` cap their own process at 0.85 of
VRAM, because past VRAM Windows backs torch memory with host RAM instead of failing (an uncapped probe "allocated"
28 GiB on the 16 GB card and took the host to 0.9 GB free).

## Measured on

RX 9070 XT (gfx1201, 16 GB), Adrenalin 32.0.31041.1004, Windows 11, torch 2.9.1+rocm7.2.1 (git 12d3edd5),
HIP 7.2.53211, hipBLASLt 1.2.2, Python 3.12.10, ComfyUI v0.43.2. Fixed-stack comparison: torch 2.11.0+rocm7.13.0,
HIP 7.13.99004. Full table: [../docs/windows-rocm-stack.md](../docs/windows-rocm-stack.md).

Dependencies: torch (ROCm build). ComfyUI v0.43.2 only for the custom-node route and the TRELLIS.2 VAE hooks
(`comfy.sd.VAE.__init__`, `comfy.ldm.trellis2.vae.ShapeVae`, `TextureVae.decode_tex_slat`); no ComfyUI code is
copied. MIT licence (top-level `LICENSE`).

## Not verified

- Linux, gfx1200, other drivers and other 7.2.x builds: not measured by us.
- The guard on a fixed stack (2.11.0+rocm7.13.0): not measured.
- Which op gives the 1.89e-04 self-test error is not recorded.
- `test_cpu.py` has not been run (it was checked to compile only).
- The GPU scripts were not re-run for this folder. They are our scripts with comments changed; `gemm_boundary.py`
  lost one section that read a shape list from our own traces, and `probe_quick.py` adds the 0.85 VRAM cap line to the
  script that produced the table above.
