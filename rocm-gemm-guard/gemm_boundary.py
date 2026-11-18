"""Where exactly do GEMMs go wrong on gfx12, and what is the narrowest guard? (11 Nov 2026)

Upstream (https://github.com/ROCm/TheRock/issues/9270, fix https://github.com/ROCm/rocm-libraries/pull/2681, 7.2
backport https://github.com/ROCm/rocm-libraries/pull/15172): hipBLASLt's gfx12 kernels launch a 2-D/3-D grid and gfx12
workgroup IDs in Y and Z are 16 bits, so tile 65536 wraps to tile 0. Predictions this script tests, unguarded:
  P1 rows: the first wrong output row is exactly 65536 * MT (MT = the kernel's row tile), and rows below are right.
  P2 addmm with a 2-D bias (beta=1): the wrapped tiles add a second time into the first rows, so rows 0.. go wrong too.
  P3 bmm: the first wrong batch is exactly 65536 (Z), in every dtype.
  P4 byte size alone does nothing: a >4 GiB operand under the row threshold is right.
Reference: the same op on 2^16-row (or 2^15-batch) slices on the GPU, checked against the CPU in fp64 on sampled rows.
Times: no guard (where correct), the guard in ./hip_gemm_chunk, and candidate narrower chunk sizes, median of 5.

    python gemm_boundary.py [rows,edge,addmm,bmm,bytes,tiles,guard,timing]   (default: all but guard)

Output: JSON lines on stdout. GPU only; sized for a 16 GB card. Run it without the guard loaded ("guard" loads it).
"""
import json, sys, time
import torch, torch.nn.functional as F

D = "cuda"; G = torch.Generator(device=D)
SMALL = 1 << 16


def med(fn, n=5):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return round(1000 * sorted(ts)[n // 2], 3)


def rows_ref(fn, a, *rest):
    return torch.cat([fn(a[i:i + SMALL], *rest) for i in range(0, a.shape[0], SMALL)])


def bad_rows(out, ref):
    scale = ref.abs().amax().float().clamp_min(1e-30)
    e = torch.cat([((o.float() - r.float()).abs() / scale).reshape(o.shape[0], -1).amax(1)  # in slices: a full float
                   for o, r in zip(out.split(SMALL), ref.split(SMALL))])                   # copy of a big output OOMs
    e = torch.nan_to_num(e, nan=1e9, posinf=1e9)
    wrong = (e > 1e-2).nonzero().flatten()
    return wrong, e.max().item()


def emit(**k): print(json.dumps(k), flush=True)


def cpu_check(fn, a, rest, ref, rows=256):
    idx = torch.randint(0, a.shape[0], (rows,), device=D)
    r = fn(a[idx].double().cpu(), *[x.double().cpu() for x in rest])
    return ((ref[idx].double().cpu() - r).abs().max() / r.abs().max()).item()


def linear_case(dt, M, K, N, layout="c", bias=None):
    torch.manual_seed(0)
    x = torch.randn(M, K, device=D, dtype=dt) if layout == "c" else torch.randn(K, M, device=D, dtype=dt).t()
    w = torch.randn(N, K, device=D, dtype=dt)
    b = None if bias is None else torch.randn(N, device=D, dtype=dt)
    fn = lambda a, w_, *b_: F.linear(a, w_, *b_)
    rest = [w] if b is None else [w, b]
    out = fn(x, *rest); ref = rows_ref(fn, x, *rest)
    wrong, err = bad_rows(out, ref)
    emit(test="linear", dtype=str(dt)[6:], M=M, K=K, N=N, layout=layout, bias=bias is not None,
         bytes_x=x.numel() * x.element_size(), first_wrong=int(wrong[0]) if len(wrong) else None, n_wrong=len(wrong),
         last_wrong=int(wrong[-1]) if len(wrong) else None, max_rel=round(err, 4), ref_vs_cpu=cpu_check(fn, x, rest, ref))
    del x, w, out, ref; torch.cuda.empty_cache()


def addmm2d_case(dt, M, K, N):
    torch.manual_seed(0)
    c, a, b = (torch.randn(M, N, device=D, dtype=dt), torch.randn(M, K, device=D, dtype=dt), torch.randn(K, N, device=D, dtype=dt))
    out = torch.addmm(c, a, b)
    ref = torch.cat([torch.addmm(c[i:i + SMALL], a[i:i + SMALL], b) for i in range(0, M, SMALL)])
    wrong, err = bad_rows(out, ref)
    emit(test="addmm2d", dtype=str(dt)[6:], M=M, K=K, N=N, first_wrong=int(wrong[0]) if len(wrong) else None,
         n_wrong=len(wrong), max_rel=round(err, 4))
    del c, a, b, out, ref; torch.cuda.empty_cache()


def bmm_case(dt, B, m=4):
    torch.manual_seed(0)
    a, b = torch.randn(B, m, m, device=D, dtype=dt), torch.randn(B, m, m, device=D, dtype=dt)
    out = torch.bmm(a, b); ref = torch.cat([torch.bmm(a[i:i + 32768], b[i:i + 32768]) for i in range(0, B, 32768)])
    wrong, err = bad_rows(out, ref)
    emit(test="bmm", dtype=str(dt)[6:], B=B, m=m, first_wrong=int(wrong[0]) if len(wrong) else None, n_wrong=len(wrong),
         max_rel=round(err, 4))


def timing(dt, M, K, N, guard_rows):
    """F.linear unguarded vs chunked at each candidate size (all chunk sizes under the measured limit are exact)."""
    torch.manual_seed(0)
    x, w = torch.randn(M, K, device=D, dtype=dt), torch.randn(N, K, device=D, dtype=dt)
    def chunked(rows):
        def f():
            out = torch.empty(M, N, device=D, dtype=dt)
            for i in range(0, M, rows): out[i:i + rows] = F.linear(x[i:i + rows], w)
            return out
        return f
    res = {"test": "timing", "dtype": str(dt)[6:], "M": M, "K": K, "N": N, "unguarded_ms": med(lambda: F.linear(x, w))}
    ref = chunked(SMALL)()
    for r in guard_rows:
        o = chunked(r)(); res[f"chunk_{r}_ms"] = med(chunked(r)); res[f"chunk_{r}_exact"] = bool(torch.equal(o, ref))
    o = F.linear(w, x).t()  # the transpose route: rows become the BLAS M (grid X, 32-bit), one call, no chunking
    res["transposed_ms"] = med(lambda: F.linear(w, x).t().contiguous()); res["transposed_maxrel"] = round(((o.float() - ref.float()).abs().max() / ref.float().abs().max()).item(), 6)
    emit(**res)
    del x, w; torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.version.hip and torch.cuda.is_available(), "needs the ROCm GPU"
    # a standalone process has no VRAM cap: past VRAM, WDDM pages into host RAM instead of failing (11 Nov: 28 GiB
    # "allocated" on the 16 GB card took the host to 0.9 GB free). Fail fast with a HIP OOM instead.
    torch.cuda.set_per_process_memory_fraction(0.85)
    emit(test="env", torch=torch.__version__, hip=torch.version.hip, device=torch.cuda.get_device_name(0),
         blas=str(torch.backends.cuda.preferred_blas_library()))
    parts = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"all"}
    f32, bf, fp = torch.float32, torch.bfloat16, torch.float16
    if parts & {"all", "rows"}:
        # P1: one big call per config; the first wrong row gives 65536 * MT directly
        for dt, K, N in ((f32, 64, 32), (f32, 64, 3), (f32, 256, 256), (f32, 1024, 1024), (bf, 64, 32), (bf, 1024, 1024),
                         (fp, 64, 32), (bf, 3072, 768), (f32, 3, 3)):
            M = min((1 << 23) + 1, int(3.0 * 2**30 / (K * torch.tensor([], dtype=dt).element_size())))
            linear_case(dt, M, K, N)
    if parts & {"all", "rows", "edge"}:
        linear_case(f32, (1 << 21) + 1, 64, 32, layout="t")      # K-major (transposed) input: another kernel layout
        linear_case(f32, (1 << 21) + 1, 64, 32, bias=True)       # 1-D bias: the gemm_and_bias epilogue path
        for M in ((1 << 19) - 1, 1 << 19, (1 << 19) + 1, (1 << 19) + 8, (1 << 19) + 9):  # the boundary itself
            linear_case(f32, M, 64, 32)
    if parts & {"all", "addmm"}:
        addmm2d_case(f32, (1 << 20) + 1, 64, 32)                # P2
    if parts & {"all", "bmm"}:
        for dt in (f32, bf, fp):
            for B in (65535, 65536, 65537, 70000): bmm_case(dt, B)   # P3
    if parts & {"all", "bytes"}:
        linear_case(f32, (1 << 19) - 8, 2304, 64)               # P4: 4.5 GiB x, rows under the threshold
    if parts & {"all", "tiles"}:  # the smallest row tile per dtype bounds the narrowest safe chunk
        for dt in (bf, fp, f32):
            for K, N in ((32, 64), (64, 384), (128, 128), (256, 1024), (512, 512), (1024, 64), (16, 16)):
                isz = torch.tensor([], dtype=dt).element_size()
                M = min((1 << 22) + 1, int(2.5 * 2**30 / ((K + 3 * N) * isz)))  # x, out, ref, diff under the 0.85 cap
                linear_case(dt, M, K, N)
    if parts & {"guard"}:  # ./hip_gemm_chunk (imported = installed): exact past the half-precision limit, and its cost
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("hgc", Path(__file__).resolve().parent / "hip_gemm_chunk" / "__init__.py")
        raw_linear = F.linear; g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
        emit(test="guard_selftest", max_rel=g.selftest(), lowp_max_rows=g.LOWP_MAX_ROWS)
        for dt, M, K, N in ((bf, (1 << 21) - 3, 64, 7), (bf, (1 << 21) - 3, 64, 6), (bf, 1500000, 64, 7), (bf, (1 << 21) - 3, 16, 16),
                            (fp, (1 << 20) + 5, 16, 16), (bf, 1500000, 128, 128)):  # 64->7/6: TRELLIS.2 VAE heads
            torch.manual_seed(0); x, w = torch.randn(M, K, device=D, dtype=dt), torch.randn(N, K, device=D, dtype=dt)
            ref = rows_ref(raw_linear, x, w); out = F.linear(x, w)
            wrong, err = bad_rows(out, ref); raw_wrong, _ = bad_rows(raw_linear(x, w), ref)
            emit(test="guard_case", dtype=str(dt)[6:], M=M, K=K, N=N, n_wrong=len(wrong), exact=bool(torch.equal(out, ref)),
                 unguarded_first_wrong=int(raw_wrong[0]) if len(raw_wrong) else None, unguarded_n_wrong=len(raw_wrong),
                 guarded_ms=med(lambda: F.linear(x, w)), unguarded_ms=med(lambda: raw_linear(x, w)))
            del x, w, ref, out; torch.cuda.empty_cache()
    if parts & {"all", "timing"}:
        timing(f32, 1 << 22, 256, 256, (1 << 18, 1 << 19))
        timing(bf, 1 << 22, 256, 256, (1 << 18, 1 << 20))
        timing(bf, (1 << 21) + 8, 64, 7, (1 << 18, 1 << 20))  # the TRELLIS.2 VAE head shape
    emit(test="done")
