# GPU check of lf_vram_cap: loads the node, then allocates 0.5 GiB blocks past the card. Passes when the cap stops them
# with a torch OOM before the card's total and host free memory stays within 1 GiB of its start. Windows only (it reads
# host memory through kernel32). Fills the whole GPU: run it with nothing else on the GPU.
#   python test_vram_cap.py [node __init__.py]             expected: PASS (exit 0)
#   LF_VRAM_CAP_FRACTION=0 python test_vram_cap.py          cap off: expected FAIL (exit 1); the loop stops at a 2 GiB host drop
import ctypes, sys
try:
    import torch
except ImportError:
    print("SKIP: torch is not installed (run with ComfyUI's embedded Python)"); sys.exit(0)
if not (torch.version.hip and torch.cuda.is_available()):
    print("SKIP: needs a ROCm GPU build of torch"); sys.exit(0)
import importlib.util
from pathlib import Path


class MS(ctypes.Structure):
    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                ("ave", ctypes.c_ulonglong)]


def host_free_gb():
    m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ap / 2**30


node = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "lf_vram_cap" / "__init__.py")
spec = importlib.util.spec_from_file_location("lf_vram_cap", node); spec.loader.exec_module(importlib.util.module_from_spec(spec))
total = torch.cuda.get_device_properties(0).total_memory / 2**30
start, blocks, oom = host_free_gb(), [], False
try:
    while len(blocks) * 0.5 < total + 4:
        if host_free_gb() < start - 2: break  # spilling into the host: the cap is not working; stop before it hurts
        b = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda"); b.fill_(1); blocks.append(b)
except torch.OutOfMemoryError:
    oom = True
got, drop = torch.cuda.memory_allocated() / 2**30, start - host_free_gb()
del blocks; torch.cuda.empty_cache()
ok = oom and got <= total and drop < 1
print(f"{'PASS' if ok else 'FAIL'}: oom={oom} allocated={got:.2f} GiB of {total:.2f}, host free drop {drop:.2f} GiB")
sys.exit(0 if ok else 1)
