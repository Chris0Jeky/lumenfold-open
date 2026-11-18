# ROCm VRAM cap (lf_vram_cap)

A hard cap on torch's GPU memory for a process on Windows ROCm, so a job that wants more than the card fails with a HIP
out-of-memory error instead of filling host RAM. It is one torch call; this folder packages it as a ComfyUI custom node.

## Problem

On Windows with a ROCm build of torch, allocations do not stop at the card's size. Past the card, Windows backs torch
memory with host RAM, and nothing fails until host memory runs out.

- Uncapped, 18.0 GiB allocated on a 15.92 GiB RX 9070 XT; host free memory fell 17.9 to 14.8 GiB (7 Nov 2026).
- A ComfyUI image-to-3D prompt that wanted 21 GiB drained host free memory to 0.02-0.07 GB twice. The second time the
  PC bugchecked (stop code 0x7E).
- A standalone GPU probe without a cap "allocated" 28 GiB on the 16 GB card and took the host to 0.9 GB free
  (11 Nov 2026).

## Who this concerns

- Windows (WDDM) with a ROCm build of torch. Measured on torch 2.9.1+rocm7.2.1, Adrenalin 32.0.31041.1004, an
  RX 9070 XT (16 GB) and 32 GB of host RAM.
- Any process that can ask for more VRAM than the card has: large mesh decoders, big batches, long sequences.
- Upstream: we know of no fix. Not tested on torch 2.11.0+rocm7.13.0, on newer drivers or on Linux.

## What we changed

At import, on a ROCm build with a GPU:

```python
torch.cuda.set_per_process_memory_fraction(0.92, 0)
```

The cap must sit at or below the WDDM per-process budget, not at the card's size: past the budget, Windows backs torch
memory with host memory even below the cap. The budget for a lone process measured about 14.75 GiB, so the default is
0.92 (14.6 GiB of 15.92). ComfyUI's `--reserve-vram 2` already plans within it (15.92 - 2 = 13.9 GiB).
`LF_VRAM_CAP_FRACTION` overrides the fraction; `0` turns the cap off.

When other GPU processes shrink the budget further, the cap is not enough: on 7 Nov 2026 a prompt reached 11.5 GiB
dedicated and spilled 6.3 GB to the host under a 1.00 cap. [../windows-gpu-spill-guard](../windows-gpu-spill-guard)
covers that case.

## Result

- With `torch.cuda.set_per_process_memory_fraction(1.0)`, the same allocations that reached 18.0 GiB uncapped stopped
  with a HIP OOM at 15.5 GiB, and host memory was untouched (7 Nov 2026). The check is the host free memory: it did
  not move.
- The 0.92 default follows from the measured lone-process budget (about 14.75 GiB). Its own OOM point was not
  recorded.

## How to run

Prerequisites: Windows 11, an AMD GPU with the Adrenalin driver, Python 3.12 with a ROCm build of torch.

- ComfyUI: copy `lf_vram_cap/` to `<ComfyUI root>/custom_nodes/lf_vram_cap/` and restart ComfyUI. The log shows
  `[lf_vram_cap] lf_vram_cap: torch allocations capped at 0.92 x 15.92 GiB; beyond that a HIP OOM, not a host spill`
  (the second number is your card's size). Delete the folder and restart to undo.
- Plain torch: put this folder on `sys.path` and `import lf_vram_cap` before the first large allocation, or make the
  call above yourself.
- Check (GPU, Windows only, fills the whole card; close other GPU programs first):

  ```
  python test_vram_cap.py
  ```

  Expected: `PASS: oom=True allocated=<GiB> GiB of <card GiB>, host free drop <GiB> GiB`, exit 0. It passes when the
  cap stops the allocations with a torch OOM below the card's size and host free memory drops by less than 1 GiB. With
  `LF_VRAM_CAP_FRACTION=0` it must print `FAIL` and exit 1; it stops filling once host free memory drops by 2 GiB.

There is no CPU check: the logic is one torch call.

## Measured on

RX 9070 XT (gfx1201, 16 GB), Adrenalin 32.0.31041.1004, Windows 11, Intel Core i5-13600K with 32 GB RAM,
torch 2.9.1+rocm7.2.1 (git 12d3edd5), HIP 7.2.53211, Python 3.12.10, ComfyUI v0.43.2. Full table:
[../docs/windows-rocm-stack.md](../docs/windows-rocm-stack.md).

Dependencies: torch (ROCm build). ComfyUI v0.43.2 only for the custom-node route. MIT licence (top-level `LICENSE`).

## Not verified

- The output of `test_vram_cap.py` at the 0.92 default was not archived; the recorded OOM at 15.5 GiB is at 1.0.
- Other cards, other drivers, Linux, and torch builds other than 2.9.1+rocm7.2.1.
- Whether the budget of about 14.75 GiB holds on other machines; it changes with other GPU processes.
