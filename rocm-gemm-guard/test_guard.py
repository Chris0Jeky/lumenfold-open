# GPU check of the guard against the CPU. Usage: python test_guard.py [path/to/hip_gemm_chunk/__init__.py]
# Allocates a few GB of GPU memory: run it with nothing else on the GPU. Exit 0 when every case passes.
import os, sys
if os.environ.get("HIP_VISIBLE_DEVICES") == "-1" or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    print("SKIP: GPU visibility disabled (CPU-only run)"); sys.exit(0)
try:
    import torch
except ImportError:
    print("SKIP: torch is not installed (run with ComfyUI's embedded Python)"); sys.exit(0)
if not (torch.version.hip and torch.cuda.is_available()):
    print("SKIP: needs a ROCm GPU build of torch"); sys.exit(0)
import importlib.util
from pathlib import Path
guard = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / 'hip_gemm_chunk' / '__init__.py')
spec = importlib.util.spec_from_file_location("hip_gemm_chunk", guard); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
F = torch.nn.functional; torch.manual_seed(1)
def rel(g, r): return ((g.float().cpu() - r.float()).abs().max() / r.float().abs().max()).item()
x = torch.randn(1 << 20, 48); w = torch.randn(40, 48); bb = torch.randn(40)
checks = {
 "linear 2^20 rows": (F.linear(x.cuda(), w.cuda(), bb.cuda()), F.linear(x, w, bb)),
 "linear 3D (4, 2^18+5, 48)": (F.linear(x[:4*(2**18+5)].reshape(4, -1, 48).cuda(), w.cuda()), F.linear(x[:4*(2**18+5)].reshape(4, -1, 48), w)),
 "matmul 2^20 x (48,40)": (torch.matmul(x.cuda(), w.T.contiguous().cuda()), x @ w.T),
 "@ operator": (x.cuda() @ w.T.cuda(), x @ w.T),
 "mm": (torch.mm(x.cuda(), w.T.cuda()), x @ w.T),
 "addmm": (torch.addmm(bb.cuda(), x.cuda(), w.T.cuda()), torch.addmm(bb, x, w.T)),
}
a = torch.randn(100000, 4, 4); b = torch.randn(100000, 4, 4)
checks["bmm 100000"] = (torch.bmm(a.cuda(), b.cuda()), torch.bmm(a, b))
checks["matmul batched 100000"] = (torch.matmul(a.cuda(), b.cuda()), torch.matmul(a, b))
checks["einsum bij,bjk->bik"] = (torch.einsum("bij,bjk->bik", a.cuda(), b.cuda()), torch.einsum("bij,bjk->bik", a, b))
v = torch.randn(100000, 4); checks["einsum bi,bij->bj"] = (torch.einsum("bi,bij->bj", v.cuda(), a.cuda()), torch.einsum("bi,bij->bj", v, a))
xb, wb, bh = x.bfloat16(), w.bfloat16(), bb.bfloat16()  # bf16 rows pass through unchunked: must stay exact
checks["bf16 linear 2^20 rows (unchunked)"] = (F.linear(xb.cuda(), wb.cuda(), bh.cuda()), F.linear(xb.double(), wb.double(), bh.double()))
checks["bf16 addmm 2^20 rows (unchunked)"] = (torch.addmm(bh.cuda(), xb.cuda(), wb.T.cuda()), torch.addmm(bh.double(), xb.double(), wb.T.double()))
ab, bbm = a.bfloat16(), b.bfloat16()
checks["bf16 bmm 100000 (chunked)"] = (torch.bmm(ab.cuda(), bbm.cuda()), torch.bmm(ab.double(), bbm.double()))
# broadcast and multi-dim batches (internal bmm of 257*256 = 65,792), baddbmm, Tensor.bmm
p, q = torch.randn(257, 256, 4, 4), torch.randn(1, 256, 4, 4)
checks["matmul broadcast [257,256]x[1,256]"] = (torch.matmul(p.cuda(), q.cuda()), torch.matmul(p, q))
checks["einsum abij,abjk->abik broadcast"] = (torch.einsum("abij,abjk->abik", p.cuda(), q.cuda()), torch.einsum("abij,abjk->abik", p, q))
c = torch.randn(100000, 4, 4)
checks["baddbmm 100000 beta/alpha"] = (torch.baddbmm(c.cuda(), a.cuda(), b.cuda(), beta=0.5, alpha=2.0), torch.baddbmm(c, a, b, beta=0.5, alpha=2.0))
checks["Tensor.bmm 100000"] = (a.cuda().bmm(b.cuda()), a.bmm(b))
checks["small matmul unchanged"] =(torch.randn(3, 3).cuda() @ torch.eye(3).cuda(), None)
bad = 0
for k, (g, r) in checks.items():
    e = 0.0 if r is None else rel(g, r); ok = e < (2e-2 if k.startswith("bf16") else 1e-4); bad += not ok; print(f"{'ok ' if ok else 'BAD'} {k}: {e:.2e}")
print("selftest", m.selftest()); sys.exit(1 if bad else 0)
