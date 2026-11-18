# lumenfold-open

Fixes, guards and tools from the Lumenfold lab. Each folder states the problem, what we changed, the measured result
with its exactness check, how to run it and what is not verified.

Most of it comes from running ComfyUI, torch and Unreal Engine 5.8 on one Windows machine with an AMD RX 9070 XT
(gfx1201) and ROCm. The stack these were measured on is in [docs/windows-rocm-stack.md](docs/windows-rocm-stack.md).

## Windows ROCm and ComfyUI

| Folder | What it does | Result |
|---|---|---|
| [rocm-gemm-guard](rocm-gemm-guard/) | Chunks large fp32 matmul and bmm calls that hipBLASLt gets silently wrong on gfx12; includes a probe for your own stack | At 8,388,609 fp32 rows: 7.86 million wrong rows on torch 2.9.1+rocm7.2.1, 0 on 2.11.0+rocm7.13.0 (fixed upstream). Guard self-test: max relative error 1.89e-4 against the CPU |
| [rocm-vram-cap](rocm-vram-cap/) | Caps torch GPU memory per process, so a job fails with an out-of-memory error instead of filling host RAM | Uncapped, 18.0 GiB was allocated on a 15.92 GiB card and spilled into host RAM; capped, a HIP out-of-memory error at 15.5 GiB with host memory untouched |
| [windows-gpu-spill-guard](windows-gpu-spill-guard/) | Stops a prompt when Windows starts backing GPU memory with host RAM; stdlib DXGI budget reader included | Tripped at 0.51 GB shared, host low point 11.66 GB |
| [windows-proactor-accept-guard](windows-proactor-accept-guard/) | Keeps an asyncio server on Windows alive after a transient `WinError 10055` in `accept()` | Repro: stock listener dead after one error, guarded listener alive (CPython 3.12.10; 3.14.3 still affected) |
| [hip-qem-decimate](hip-qem-decimate/) | HIP kernel for the QEM edge-collapse checks that ship only for CUDA | 11.8x per call on a 410K-triangle mesh, 0 mismatches against the torch reference |
| [fast-mesh-unwrap](fast-mesh-unwrap/) | Faster UV unwrap for ComfyUI meshes (sparse batched LSCM) | 180.97 s to 17.59 s on 472,523 faces, all 25,954 charts identical, UV error 1.39e-8 (CPU) |

## Engines and kernels

| Folder | What it does | Result |
|---|---|---|
| [kernel-bakeoff-formation](kernel-bakeoff-formation/) | One C++ task, four model-written solutions, bit-exact checks and the bench to rerun them | 75.97 ms to 2.95 ms (25.7x) at 100,000 agents on one thread, 0 mismatches on 1,518 hidden cases |
| [ue58-pitfalls](ue58-pitfalls/) | Unreal Engine 5.8 calls and importers that fail without an error, each with symptom and fix | List |

## Assets

| Folder | What it does | Result |
|---|---|---|
| [polyhaven-bulk-download](polyhaven-bulk-download/) | Bulk Poly Haven download with an md5 check per file, resume and a manifest | 1,016 assets, 5,578 files, 64.1 GB at 4K |

## Notes

- Numbers are from one machine. Each folder lists what it was measured on and what was not verified.
- MIT licence, except files that carry an upstream notice (CPython PSF, ComfyUI GPL-3.0); their folder says which.
- Third-party code appears only as a diff or as a marked, modified function with its upstream licence.
