"""Hard VRAM cap for a ComfyUI process on Windows ROCm (7 Nov 2026). Loaded only on ROCm builds with a GPU; adds no nodes.

Works as a ComfyUI custom node (copy this folder into custom_nodes/) or from plain torch (`import lf_vram_cap` with this
folder's parent on sys.path, before the first large allocation).

On this Windows ROCm build torch lets allocations run past the card: measured 7 Nov, uncapped, 18.0 GiB allocated on the
15.92 GiB RX 9070 XT with the overflow backed by host memory (host free 17.9 -> 14.8 GiB). An image-to-3D prompt that
wanted 21 GiB drained the host to 0.02-0.07 GB twice, and the second time the PC bugchecked (0x7E). With
torch.cuda.set_per_process_memory_fraction(1.0) the same allocations stop with a HIP OOM at 15.5 GiB and the host is
untouched, so a prompt that wants more than the card fails fast instead of taking the machine down.

The cap must sit at or below the WDDM per-process budget, not the card size: past the budget, Windows backs torch memory
with host memory even below the cap (7 Nov, a prompt reached 11.5 GiB dedicated and spilled 6.3 GB to the host under a
1.00 cap). The budget for a lone process measured about 14.75 GiB, so the default is 0.92 (14.6 GiB). ComfyUI's
--reserve-vram 2 already plans within it (15.92 - 2 = 13.9 GiB). When other GPU processes shrink the budget, the spill
guard (windows-gpu-spill-guard in this repository) is the backstop.

LF_VRAM_CAP_FRACTION overrides the fraction of device memory (default 0.92); 0 disables the cap.
Remove this folder (and restart ComfyUI) to undo.
"""
import logging, os
import torch

log = logging.getLogger("lf_vram_cap")
FRACTION = float(os.environ.get("LF_VRAM_CAP_FRACTION", "0.92"))

if torch.version.hip and torch.cuda.is_available() and FRACTION > 0:
    torch.cuda.set_per_process_memory_fraction(FRACTION, 0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    msg = f"lf_vram_cap: torch allocations capped at {FRACTION:.2f} x {total:.2f} GiB; beyond that a HIP OOM, not a host spill"
    log.warning(msg); print(f"[lf_vram_cap] {msg}", flush=True)

NODE_CLASS_MAPPINGS = {}
