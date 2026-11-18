# Windows ROCm stack

The machine and software the folders in this repository were measured on, and the versions known to be bad or fixed.

## Measured on

| Part | Version |
|---|---|
| GPU | AMD Radeon RX 9070 XT, gfx1201 (RDNA4), 16 GB; torch reports 15.92 GiB |
| Driver | AMD Software: Adrenalin Edition 32.0.31041.1004 (driver dated 17 Aug 2026) |
| OS | Windows 11 (WDDM) |
| CPU and RAM | Intel Core i5-13600K (6 P-cores, 8 E-cores), 32 GB |
| Python | 3.12.10, the embedded Python of the ComfyUI Windows portable build |
| torch | 2.9.1+rocm7.2.1, git 12d3edd5 |
| HIP | 7.2.53211-158bd99533 |
| ROCm libraries | TheRock wheels `rocm-sdk-core` and `rocm-sdk-libraries-custom` 7.2.1: hipBLASLt 1.2.2 and rocBLAS 5.2.0 (per the ROCm 7.2.1 release notes) |
| BLAS backend | torch's preferred BLAS library is Cublaslt, which is hipBLASLt on ROCm. hipBLASLt ships tuned gfx1201 kernels; rocBLAS ships only `*_fallback_gfx1201` libraries |
| ComfyUI | v0.43.2, commit `423cbd20511becb1f16122ab518e92e58bfa03d9` |

Comparison stack, same machine, separate virtual environment: torch 2.11.0+rocm7.13.0 from AMD's
`repo.amd.com/rocm/whl/gfx120X-all` index, HIP 7.13.99004.

Print the same facts for your own stack:

```python
import sys, torch
print(sys.version, torch.__version__, torch.version.git_version[:8], torch.version.hip,
      torch.cuda.get_device_name(0), torch.backends.cuda.preferred_blas_library())
```

The driver version is in AMD Software under System, or in Device Manager.

## Silent wrong GEMMs (rocm-gemm-guard)

Cause: hipBLASLt's gfx12 GEMM kernels read the grid's Y and Z workgroup IDs as 16 bits, so a launch with more than
65,536 workgroups in Y or Z wraps. Fix: ROCm/rocm-libraries#2681 (commit 507afd7 on `develop`), which folds Y and Z into
X. The bug is in hipBLASLt, not in Windows; a bmm report reproduced it on WSL2. We measured Windows only.

| Stack | State | Source |
|---|---|---|
| torch 2.9.1+rocm7.2.1, HIP 7.2.53211, gfx1201 | Wrong: fp32 rows from 524,288, fp16/bf16 rows from 1,048,576 (K = N = 16), bmm from batch 65,536 | Measured here, 11 Nov 2026 |
| ROCm 7.2.0 with a torch 2.12 dev build | Wrong (11.98% zeros) | Reported in ROCm/rocm-libraries#5706 |
| ROCm 7.2.4, torch 2.14.1+rocm7.2 | Still reproduces | Reported in ROCm/rocm-libraries#13243 |
| Any ROCm 7.2.x | The 7.2 backport, ROCm/rocm-libraries#15172, was still open on 11 Nov 2026. A first cherry-pick (ROCm/rocm-libraries#5146) was reverted for gfx1200/gfx1201 segfaults (ROCm/rocm-libraries#5292) | Upstream pull requests |
| TheRock nightly 7.13.0a20260323 (gfx120X-all) | 0% bad on an RX 9070 XT | Reported in ROCm/rocm-libraries#5706 |
| "ROCm 7.11 and later" | Fixed, according to an AMD maintainer; we found no release tag that confirms it | ROCm/TheRock#9270 |
| torch 2.11.0+rocm7.13.0, HIP 7.13.99004, gfx1201 | Fixed: 0 wrong rows at 8,388,609 fp32 rows, where 2.9.1+rocm7.2.1 gives 7.86 million; bmm correct | Measured here, 11 Nov 2026 |

From reading the PyTorch v2.9.1 source (not tested): `blasPreferredBackend()` picks hipBLASLt for gfx1200 and gfx1201,
and on Windows (`_MSC_VER`) `setBlasPreferredBackend` only warns, so `TORCH_BLAS_PREFER_HIPBLASLT=0` probably cannot
move gfx1201 off hipBLASLt there. `DISABLE_ADDMM_CUDA_LT=1` disables only addmm's bias epilogue path. fp64 never goes
to hipBLASLt.

## VRAM past the card (rocm-vram-cap, windows-gpu-spill-guard)

On this stack a torch allocation does not fail at the card's size. Windows backs the excess with host RAM (shared GPU
memory), and it does the same below the card's size once a process passes its WDDM video-memory budget.

- Uncapped, 18.0 GiB allocated on the 15.92 GiB card; host free memory fell 17.9 to 14.8 GiB (7 Nov 2026).
- A ComfyUI prompt that wanted 21 GiB drained host free memory to 0.02-0.07 GB twice; the second time the PC
  bugchecked (stop code 0x7E).
- The WDDM budget for a lone process measured about 14.75 GiB. With other GPU processes running it was about
  11.5 GB. A fresh ComfyUI process filled 15.3 GB before spilling; the same process after a TRELLIS.2 job only 12.1 GB.

We know of no upstream fix. We have not tested whether torch 2.11.0+rocm7.13.0 or a newer driver behaves differently.

## Upstream links

- ROCm/TheRock#9270 (row case, diagnosed by AMD): https://github.com/ROCm/TheRock/issues/9270
- ROCm/TheRock#8554: https://github.com/ROCm/TheRock/issues/8554
- ROCm/TheRock#9552 (bmm past batch 65,536): https://github.com/ROCm/TheRock/issues/9552
- ROCm/rocm-libraries#2681 (fix): https://github.com/ROCm/rocm-libraries/pull/2681
- ROCm/rocm-libraries#15172 (7.2 backport): https://github.com/ROCm/rocm-libraries/pull/15172
- ROCm/rocm-libraries#5146 (reverted cherry-pick): https://github.com/ROCm/rocm-libraries/issues/5146
- ROCm/rocm-libraries#5292 (gfx1200/gfx1201 segfaults): https://github.com/ROCm/rocm-libraries/issues/5292
- ROCm/rocm-libraries#5706: https://github.com/ROCm/rocm-libraries/issues/5706
- ROCm/rocm-libraries#13243: https://github.com/ROCm/rocm-libraries/issues/13243
- Comfy-Org/ComfyUI#17401 (symptom in ComfyUI): https://github.com/Comfy-Org/ComfyUI/issues/17401
- PyTorch v2.9.1 BLAS backend selection: https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/Context.cpp
- ROCm 7.2.1 release notes: https://rocm.docs.amd.com/en/docs-7.2.1/about/release-notes.html
