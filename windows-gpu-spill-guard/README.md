# Windows GPU spill guard (lf_spill_guard)

Stops a ComfyUI prompt as soon as Windows starts backing its GPU memory with host RAM, before host memory runs out.
`spill_core.py` and `dxgi_budget.py` need no torch and no ComfyUI: stdlib `ctypes` only, Windows only.

## Problem

Windows gives each process a video-memory budget (WDDM). Past it, new GPU allocations are backed by shared host memory,
even when the process is under its torch memory cap. Nothing fails until host memory is gone.

Measured on 7 Nov 2026 in ComfyUI, an image-to-3D prompt with other GPU programs running:

- WDDM gave the ComfyUI process a budget of about 11.5 GB, under the 15.92 GiB torch cap.
- Past it, the process's shared GPU memory rose 0.1 to 6.3 GB in 20 s. Then host available memory fell 5.2 to
  0.1 GB in 4 s. The torch cap never fired.
- After an out-of-memory error, ComfyUI's own handler offloads every model to the CPU: the process's working set went
  2.7 to 7.7 GB in 2 s and host free memory fell to 1.9 GB.

The budget moves: a fresh ComfyUI process filled 15.3 GB before spilling, the same process after a TRELLIS.2 job only
12.1 GB.

## Who this concerns

- Windows (WDDM) with a GPU build of torch that lets Windows back VRAM with host memory. Measured on torch
  2.9.1+rocm7.2.1, Adrenalin 32.0.31041.1004, an RX 9070 XT (16 GB) and 32 GB of host RAM.
- The node is for ComfyUI v0.43.2. The core works in any Python process on Windows.
- Upstream: we know of no fix. Not tested with other torch builds or drivers.

## What we changed

- `lf_spill_guard/spill_core.py` (no torch, no ComfyUI): reads this process's `\GPU Process Memory(*)\Dedicated Usage`
  and `Shared Usage` performance counters through PDH, per adapter, and runs the trip and restore state machine with
  callbacks. It reads OS counters only and opens no GPU device.
- `lf_spill_guard/dxgi_budget.py` (no torch, no ComfyUI): reads this process's WDDM budget per adapter with
  `IDXGIAdapter3::QueryVideoMemoryInfo` through ctypes COM. It creates no device.
- `lf_spill_guard/spill_guard.py`, the ComfyUI wiring. A daemon thread reads shared usage every 0.25 s on the adapter
  where the process holds the most dedicated memory. While a prompt runs and shared usage is more than
  `LF_SPILL_GUARD_GB` (default 1.0) above what was left in the last idle spell, it trips:
  - torch's memory fraction is clamped to what torch has reserved now, so the next allocation is a HIP OOM;
  - each running prompt is interrupted by id, and the queue's `unload_models` and `free_memory` flags are set;
  - a JSON line goes to `LF_SPILL_GUARD_LOG` (default `<ComfyUI root>/user/lf_spill_guard.jsonl`), followed by a
    record with the WDDM budget.

  It restores the previous fraction once those prompts have ended and shared usage has fallen, never trips while idle,
  and retries a trip or restore that raised. An unreadable counter changes nothing.
- `lf_spill_guard/oom_drop.py`: after an OOM seen by ComfyUI's node handler, or a trip, the main loop's unload moves
  model weights to the `meta` device and resets the executor cache in the same step, instead of copying the models to
  host RAM. Models reload from disk when next needed. Other unloads keep ComfyUI's behaviour. It arms only when
  ComfyUI was started through its `main.py`.

No ComfyUI code is copied. The node calls these ComfyUI v0.43.2 internals by name: `PromptServer.instance.prompt_queue`
(`get_current_queue_volatile`, `interrupt_if_running`, `set_flag`, `mutex`), `comfy.model_management` (`is_oom`,
`unload_all_models`, `current_loaded_models`, `soft_empty_cache`, `interrupt_current_processing`),
`folder_paths.get_user_directory`, and the local name `e` (the prompt executor) in `main.py`'s worker frame. A ComfyUI
change to any of these can break it.

## Result

- GPU bench, 7 Nov 2026 ([results/spill-bench-2026-11-07.json](results/spill-bench-2026-11-07.json)): a helper process
  held 4 GB of VRAM to shrink the budget, the bench filled VRAM in 128 MB steps, threshold 0.5 GB, guard polled every
  1 s. The guard tripped at 0.51 GB shared with 11.37 GB dedicated and 11.62 GiB reserved; the next allocation failed
  ("OOM after the trip") at 11.62 GiB; the host available low point was 11.66 GB.
- The same bench on an otherwise empty card: the process reached 15.88 GiB with 0.38 GB shared, so nothing spilled,
  nothing tripped, and the cap's OOM ended the fill.
- Poll interval: in a production run a 1 s poll let 2.86 GB spill before the trip. The node now polls every 0.25 s.
- CPU tests, 18 Nov 2026: `test_spill_guard.py`, 49 tests, OK on Windows 11 with Python 3.14.3. They include the measured 7 Nov trace (it
  trips once, at the first reading over 1 GB), and the PDH reader against `typeperf` on the live machine.

## How to run

Prerequisites: Windows 11, Python 3.12 (the CPU tests were also run on 3.14). For the node:
ComfyUI v0.43.2 with a ROCm build of torch. For the GPU bench: torch (ROCm build) and `psutil`.

- CPU check (no GPU, no torch, no ComfyUI):

  ```
  python test_spill_guard.py
  ```

  Expected: `Ran 49 tests in <s>` and `OK`. The install tests use stand-in torch and ComfyUI modules, so real torch is
  never imported. On a system other than Windows the live PDH tests are skipped. One test reads
  `../rocm-vram-cap/lf_vram_cap/__init__.py` for the cap's default and skips when that folder is absent.
- Read your own WDDM budget (no torch):

  ```
  python lf_spill_guard/dxgi_budget.py
  ```

  One JSON line per hardware adapter with `Budget`, `CurrentUsage`, `AvailableForReservation` and
  `CurrentReservation` in GiB, for that Python process.
- ComfyUI: copy `lf_spill_guard/` to `<ComfyUI root>/custom_nodes/lf_spill_guard/` and restart ComfyUI with an empty
  queue. The log shows `[lf_spill_guard] installed: threshold 1.0 GB shared, every 0.25 s, OOM drop armed on ComfyUI's
  OOM handler and guard trips; ...`. `NOT installed` means there is no protection. Use it with
  [../rocm-vram-cap](../rocm-vram-cap). Delete the folder and restart to undo.
- GPU check (stop ComfyUI first; needs torch and psutil):

  ```
  python -s bench_spill_guard.py --step-mb 128 --pause-s 0.4 --threshold-gb 0.5 --hog-gb 4
  ```

  Pass: the `SPILL {...}` line has `"end": "OOM after the trip"`. It stops filling below 6 GB of host available memory
  whatever the guard does (`--host-floor-gb`).

## Measured on

RX 9070 XT (gfx1201, 16 GB), Adrenalin 32.0.31041.1004, Windows 11, Intel Core i5-13600K with 32 GB RAM,
torch 2.9.1+rocm7.2.1 (git 12d3edd5), HIP 7.2.53211, Python 3.12.10, ComfyUI v0.43.2. Full table:
[../docs/windows-rocm-stack.md](../docs/windows-rocm-stack.md).

Dependencies: Python stdlib for the core. ComfyUI v0.43.2 and torch for the node; torch and psutil (version not
recorded) for the bench. MIT licence (top-level `LICENSE`).

## Not verified

- The 0.25 s poll has not been measured on the GPU; the recorded bench polled every 1 s.
- The recorded bench run's `--pause-s` was not recorded; the command above uses the rate we planned for it. The bench
  script may differ in detail from the one that produced the result file.
- The OOM drop's effect on host memory in a live ComfyUI prompt was not measured; the tests cover it with stand-ins.
- More than one GPU: adapter selection is covered by tests with fakes only.
- ComfyUI versions other than v0.43.2, torch builds other than 2.9.1+rocm7.2.1, and NVIDIA GPUs.
