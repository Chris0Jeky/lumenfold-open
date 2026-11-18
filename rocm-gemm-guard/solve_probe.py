# Batched 3x3 linalg.solve past the batch limit, unguarded then guarded. Usage: python solve_probe.py <batch>   (GPU)
# Unguarded, a batch above 65536 can fail with "HIP error: invalid configuration argument"; guarded it must match the CPU.
import sys
try:
    import torch
except ImportError:
    print("SKIP: torch is not installed (run with ComfyUI's embedded Python)"); sys.exit(0)
if not (torch.version.hip and torch.cuda.is_available()):
    print("SKIP: needs a ROCm GPU build of torch"); sys.exit(0)
n = int(sys.argv[1]); M = torch.randn(n, 3, 3); A = M @ M.transpose(1, 2) + torch.eye(3); y = torch.randn(n, 3, 1)
try: r = torch.linalg.solve(A.cuda(), y.cuda()); torch.cuda.synchronize(); print(n, "raw ok err", ((r.cpu() - torch.linalg.solve(A, y)).abs().max()).item())
except Exception as e: print(n, "raw FAIL", str(e).splitlines()[0])
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent)); import hip_gemm_chunk  # installs the guard and runs its self-test
r = torch.linalg.solve(A.cuda(), y.cuda()); torch.cuda.synchronize(); print(n, "guarded err", ((r.cpu() - torch.linalg.solve(A, y)).abs().max()).item())
