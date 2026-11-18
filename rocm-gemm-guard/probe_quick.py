"""Does this torch/ROCm stack get large GEMMs silently wrong? About 6 GB of VRAM, under a minute. GPU only.

    python probe_quick.py

Run it WITHOUT the guard loaded (not from inside ComfyUI). Each line prints (first wrong row or batch, number wrong).
Reference: the same op on slices that stay under 65536 tiles. (None, 0) everywhere means this stack does not need the
guard for these shapes. See README.md for the output on a broken and a fixed stack.
"""
import torch, torch.nn.functional as F

assert torch.version.hip and torch.cuda.is_available(), "needs a ROCm build of torch and a GPU"
# Past VRAM, Windows backs torch memory with host RAM instead of failing: cap this process so it fails fast instead.
torch.cuda.set_per_process_memory_fraction(0.85)
torch.manual_seed(0); d = "cuda"


def first_wrong(out, ref):
    e = ((out.float() - ref.float()).abs().reshape(out.shape[0], -1).amax(1) / ref.float().abs().max())
    bad = (torch.nan_to_num(e, nan=1.0) > 1e-2).nonzero().flatten()
    return (int(bad[0]), len(bad)) if len(bad) else (None, 0)


def by_slices(fn, *xs, step=1 << 15):
    return torch.cat([fn(*(x[i:i + step] for x in xs)) for i in range(0, xs[0].shape[0], step)])


print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))
for dt, M, K, N in ((torch.float32, (1 << 19) + 9, 64, 32), (torch.bfloat16, (1 << 20) + 9, 16, 16), (torch.float16, (1 << 20) + 9, 16, 16)):
    x, w = torch.randn(M, K, device=d, dtype=dt), torch.randn(N, K, device=d, dtype=dt)
    print(f"linear {str(dt)[6:]:8} M={M} K={K} N={N}:", first_wrong(F.linear(x, w), by_slices(lambda a: F.linear(a, w), x)))
c, a, b = torch.randn(1 << 20, 32, device=d), torch.randn(1 << 20, 64, device=d), torch.randn(64, 32, device=d)
print("addmm 2-D bias M=2^20:", first_wrong(torch.addmm(c, a, b), by_slices(lambda c_, a_: torch.addmm(c_, a_, b), c, a)))
for dt in (torch.float32, torch.bfloat16):
    a, b = torch.randn(65537, 4, 4, device=d, dtype=dt), torch.randn(65537, 4, 4, device=d, dtype=dt)
    print(f"bmm {str(dt)[6:]} batch 65537:", first_wrong(torch.bmm(a, b), by_slices(torch.bmm, a, b)))
