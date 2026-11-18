"""HIP large-GEMM guard. Loaded only on ROCm builds with a GPU; adds no ComfyUI nodes.

Works two ways: as a ComfyUI custom node (copy this folder into custom_nodes/), or from plain torch (put this folder's
parent on sys.path and `import hip_gemm_chunk` once, before the workload). Either way the import installs the guard.

torch 2.9.1+rocm7.2.1 on RDNA4 (gfx1201) returns wrong results for fp32 matmuls with more than 2^19 rows and for batched
matmuls past batch 65536 (measured on an RX 9070 XT, 4 Nov 2026: errors of 30-260% from row 524288 / batch 65536,
1e-7 below). TRELLIS.2 and Pixal3D decode on millions of voxel rows, so their meshes and textures come out broken.
Symptom report: https://github.com/Comfy-Org/ComfyUI/issues/17401
This wraps F.linear, matmul, mm, addmm, bmm, baddbmm, Tensor.bmm, Tensor.__matmul__, batch einsum, linalg.solve and det so no single call crosses the
limits, then checks itself against the CPU at load. Remove this folder (and restart ComfyUI) to undo.
Direct aten calls are outside this Python-entrypoint guard.
Full einsum batch analysis covers explicit two-operand equations with unique labels. More operands retain only the
legacy shared leading-label split; ellipses, repeated labels, implicit outputs and sublist notation pass through.

6 Nov 2026: the row limit is an fp32 bug. fp16 and bf16 row GEMMs are correct past 2^19 rows at K >= 32 (measured to
2^21+7 rows, guard off, against fp64), so row chunking passes them through unchunked up to LOWP_MAX_ROWS (chunking cost
bf16 25-45%). The batch limit hits every dtype (bf16/fp16 bmm wrong past 65536 too,
https://github.com/ROCm/TheRock/issues/9552): batch chunking stays for all. fp32 GEMMs at these sizes also run at 1-3
TFLOPS against ~100 for bf16, so the TRELLIS.2 VAEs decode in bf16 here (fp16 decodes to NaN; bf16 has fp32's range).
Set LF_TRELLIS_VAE_FP32=1 to restore the fp32 VAEs.

11 Nov 2026: the cause is hipBLASLt's gfx12 kernels reading grid Y/Z workgroup IDs as 16 bits
(https://github.com/ROCm/TheRock/issues/9270, fix https://github.com/ROCm/rocm-libraries/pull/2681, not in 7.2.x).
The first wrong row is exactly 65536 x the selected kernel's row tile: fp32 tiles 8 or 16 (wrong from 2^19), and
fp16/bf16 tiles 64 at K=32-64 but 16 at K=N=16, so half-precision rows go wrong from 2^20, not 2^21. bmm goes wrong
from batch 65536 exactly. LOWP_MAX_ROWS is 2^20; fp32 keeps 2^18 chunks (2^19 chunks were slower: 1200 vs 720 ms at
4M x 256 x 256).
"""
import logging, math, os, torch
import torch.nn.functional as F

ROWS, BATCH = 1 << 18, 1 << 15  # well under the 2^19-row and 65536-batch limits
LOWP = (torch.float16, torch.bfloat16)  # row GEMMs exact past 2^19 rows in these dtypes (6 Nov sweep) ...
LOWP_MAX_ROWS = 1 << 20  # ... but only to 65536 x their smallest row tile measured (16, at K=N=16; 11 Nov)
log = logging.getLogger("hip_gemm_chunk")


def _batch_chunked(fn, ops, batch, axes, output_axes, kw):
    """Slice broadcast batch axes as views, never slice a singleton, and keep each leaf below BATCH.

    axes maps each effective batch axis to an operand axis (None for an absent axis). Recursing also handles a
    batch whose remaining dimensions alone exceed the limit. No reduction axis is split, so autograd is retained.
    """
    n = math.prod(batch)
    if n < BATCH: return fn(*ops, **kw)
    dim = next(i for i, size in enumerate(batch) if size > 1)
    step = max(1, (BATCH - 1) // (n // batch[dim]))
    out = kw.get("out")
    parts = []
    for start in range(0, batch[dim], step):
        size = min(step, batch[dim] - start)
        sliced = [op.narrow(axis[dim], start, size) if axis[dim] is not None and op.shape[axis[dim]] != 1
                  else op for op, axis in zip(ops, axes)]
        piece = (*batch[:dim], size, *batch[dim + 1:])
        piece_kw = dict(kw, out=out.narrow(output_axes[dim], start, size)) if out is not None else kw
        result = _batch_chunked(fn, sliced, piece, axes, output_axes, piece_kw)
        if out is None: parts.append(result)
    return out if out is not None else torch.cat(parts, dim=output_axes[dim])


def _batch_out(kw, shape):
    """Prepare an out= buffer once; leaves use views so the native op validates dtype/device/autograd."""
    out = kw.get("out")
    if out is not None and tuple(out.shape) != tuple(shape): out.resize_(shape)


def _einsum_batch(eq, ops):
    """Only shared, retained labels are batch axes. Leave other equation forms to native einsum."""
    if not isinstance(eq, str) or len(ops) < 2 or not all(isinstance(o, torch.Tensor) for o in ops): return None
    ins, arrow, out = eq.replace(" ", "").partition("->")
    terms = ins.split(",")
    if not arrow or len(terms) != len(ops): return None
    if any(not all(c.isascii() and c.isalpha() for c in t) or len(set(t)) != len(t) for t in (*terms, out)): return None
    if any(len(t) != o.dim() for t, o in zip(terms, ops)) or not set(out) <= set(ins): return None
    labels = [c for c in out if all(c in t for t in terms)]
    if len(ops) > 2:
        # Preserve the old safe leading-batch case. Other intermediate batches depend on contraction order.
        lead = out[:1]
        if not lead or not all(t.startswith(lead) for t in terms) or len({o.shape[0] for o in ops}) != 1: return None
        labels = [lead]
    axes = [tuple(t.index(c) for c in labels) for t in terms]
    batch = []
    for dims in zip(*axes):
        sizes = [o.shape[d] for o, d in zip(ops, dims)]
        non_singleton = {size for size in sizes if size != 1}
        if len(non_singleton) > 1: return None
        batch.append(next(iter(non_singleton), 1))
    return tuple(batch), axes, tuple(out.index(c) for c in labels)


def _rows_chunked(fn, a, *rest, **kw):
    """Split the leading (row) dims of `a`; every row's result depends only on that row."""
    lead = a.shape[:-1]; n = a.numel() // max(a.shape[-1], 1)
    if a.dim() < 2 or (a.dim() == 2 and n <= ROWS): return fn(a, *rest, **kw)
    # N-D inputs always go in as 2-D rows: a non-contiguous (..., k) input otherwise runs as an internal bmm with batch =
    # its leading dim, and past 65536 that is wrong in every dtype (measured 6 Nov: [70000, 2, 48] fp32 rel err 1.2,
    # bf16 NaN). reshape is a view for contiguous inputs and a copy otherwise.
    flat = a.reshape(n, a.shape[-1])
    if n <= ROWS or (a.dtype in LOWP and n <= LOWP_MAX_ROWS):  # one unchunked call
        out = fn(flat, *rest, **kw); return out.reshape(*lead, out.shape[-1])
    first = fn(flat[:ROWS], *rest, **kw)
    out = first.new_empty((n, first.shape[-1])); out[:ROWS] = first; del first  # preallocated: torch.cat held every chunk
    for i in range(ROWS, n, ROWS): out[i:i + ROWS] = fn(flat[i:i + ROWS], *rest, **kw)  # and the result at once (OOM, 5 Nov)
    return out.reshape(*lead, out.shape[-1])


def install():
    lin, mm, addmm, bmm, matmul, einsum, tmatmul = F.linear, torch.mm, torch.addmm, torch.bmm, torch.matmul, torch.einsum, torch.Tensor.__matmul__
    baddbmm, tbmm = torch.baddbmm, torch.Tensor.bmm

    def g_linear(*a, **kw):  # F.linear(input, weight, bias=None); callers pass bias (and sometimes all three) by keyword
        x = a[0] if a else kw.get("input")
        if not (isinstance(x, torch.Tensor) and x.is_cuda): return lin(*a, **kw)
        w = a[1] if len(a) > 1 else kw["weight"]; b = a[2] if len(a) > 2 else kw.get("bias")
        return _rows_chunked(lin, x, w, b)
    def g_mm(a, b, **kw): return _rows_chunked(mm, a, b, **kw) if a.is_cuda and a.dim() == 2 else mm(a, b, **kw)
    def g_addmm(c, a, b, **kw):
        if not (a.is_cuda and a.shape[0] > ROWS) or (a.dtype in LOWP and a.shape[0] <= LOWP_MAX_ROWS): return addmm(c, a, b, **kw)
        c = c.expand(a.shape[0], b.shape[1])
        return torch.cat([addmm(c[i:i + ROWS], a[i:i + ROWS], b, **kw) for i in range(0, a.shape[0], ROWS)])
    def batched_mm(fn, a, b, kw, c=None):
        ops = (a, b) if c is None else (c, a, b)
        if not (a.is_cuda and a.dim() == b.dim() == 3 and a.shape[0] == b.shape[0] and a.shape[0] >= BATCH):
            return fn(*ops, **kw)
        shape = (a.shape[0], a.shape[1], b.shape[2])
        if c is not None:
            out = kw.get("out")
            if out is not None and c.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
                c = c.clone()  # an input aliasing out would be overwritten by earlier chunks before later ones read it
            ops = (c.expand(shape), a, b)
        _batch_out(kw, shape)
        return _batch_chunked(fn, ops, (a.shape[0],), [(0,)] * len(ops), (0,), kw)
    def g_bmm(input, mat2, **kw): return batched_mm(bmm, input, mat2, kw)
    def g_tbmm(input, mat2, **kw):
        if "out" in kw: return tbmm(input, mat2, **kw)  # Tensor.bmm has no out= overload; retain its native error.
        return batched_mm(tbmm, input, mat2, kw)
    def g_baddbmm(input, batch1, batch2, **kw): return batched_mm(baddbmm, batch1, batch2, kw, input)
    def g_matmul(input, other, **kw):
        a, b = input, other
        if not (isinstance(a, torch.Tensor) and a.is_cuda): return matmul(a, b, **kw)
        if b.dim() == 2 and a.dim() >= 2: return _rows_chunked(matmul, a, b, **kw)  # (..., n, k) @ (k, m)
        if a.dim() >= 1 and b.dim() >= 1:
            try: batch = tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
            except RuntimeError: return matmul(a, b, **kw)  # native error for incompatible shapes
            if math.prod(batch) >= BATCH:
                axes = [tuple([None] * (len(batch) - max(o.dim() - 2, 0)) + list(range(max(o.dim() - 2, 0)))) for o in (a, b)]
                shape = batch + (a.shape[-2:-1] if a.dim() > 1 else ()) + (b.shape[-1:] if b.dim() > 1 else ())
                _batch_out(kw, shape)
                return _batch_chunked(matmul, (a, b), batch, axes, tuple(range(len(batch))), kw)
        return matmul(a, b, **kw)
    def g_tmatmul(a, b): return g_matmul(a, b) if isinstance(b, torch.Tensor) else tmatmul(a, b)
    def g_einsum(eq, *ops, **kw):
        ops_ = ops[0] if len(ops) == 1 and isinstance(ops[0], (list, tuple)) else ops
        if not (ops_ and all(isinstance(o, torch.Tensor) and o.is_cuda for o in ops_)): return einsum(eq, *ops, **kw)
        plan = _einsum_batch(eq, ops_)
        if plan is None:  # forms the full analysis skips (ellipsis, repeated labels) keep the old shared leading-label split
            ins, _, out = eq.replace(" ", "").partition("->") if isinstance(eq, str) else ("", "", "")
            lead, n = ins[:1], ops_[0].shape[0] if ops_[0].dim() else 0
            if (out and lead.isalpha() and out[0] == lead and all(t[:1] == lead for t in ins.split(","))
                    and all(o.dim() and o.shape[0] == n for o in ops_)):
                plan = (n,), [(0,)] * len(ops_), (0,)
        if plan is not None and math.prod(plan[0]) >= BATCH:
            batch, axes, output_axes = plan
            return _batch_chunked(lambda *xs, **kwargs: einsum(eq, *xs, **kwargs), ops_, batch, axes, output_axes, kw)
        return einsum(eq, *ops, **kw)

    F.linear, torch.mm, torch.addmm, torch.bmm, torch.matmul, torch.einsum = g_linear, g_mm, g_addmm, g_bmm, g_matmul, g_einsum
    torch.Tensor.__matmul__ = g_tmatmul
    torch.baddbmm, torch.Tensor.bmm = g_baddbmm, g_tbmm

    # Batched small solves hit the same batch limit as a launch error ("HIP error: invalid configuration argument"):
    # ComfyUI's QEM DecimateMesh solves one 3x3 system per edge, ~1M at once (5 Nov 2026).
    solve, det, ldet = torch.linalg.solve, torch.det, torch.linalg.det
    def _batched(fn, a, *rest, **kw):
        if not (isinstance(a, torch.Tensor) and a.is_cuda and a.dim() >= 3 and a.shape[0] > BATCH) or "out" in kw: return fn(a, *rest, **kw)
        n = a.shape[0]; cut = lambda t, i: t[i:i + BATCH] if isinstance(t, torch.Tensor) and t.dim() and t.shape[0] == n else t
        return torch.cat([fn(a[i:i + BATCH], *[cut(r, i) for r in rest], **kw) for i in range(0, n, BATCH)])
    torch.linalg.solve = lambda A, B, *a, **kw: _batched(solve, A, B, *a, **kw)
    torch.det = lambda A: _batched(det, A); torch.linalg.det = lambda A: _batched(ldet, A)


def trellis_vae_dtype():
    """TRELLIS.2 shape/texture VAEs only, without changing image VAEs for every other workflow: bf16 (6 Nov; fp32 GEMMs at
    their sizes run at 1-3 TFLOPS here), or fp32 with LF_TRELLIS_VAE_FP32=1. Never fp16: the shape VAE decodes to NaN.
    The texture decode hands BakeTextureFromVoxel fp32 features (it calls .numpy(), which has no bf16).
    Needs ComfyUI; from plain torch the import fails and the guard prints one "not applied" line, which is harmless."""
    import comfy.sd, comfy.ldm.trellis2.vae as tv
    dt = torch.float32 if os.environ.get("LF_TRELLIS_VAE_FP32") == "1" else torch.bfloat16
    init = comfy.sd.VAE.__init__
    def g_init(self, *a, **kw):
        init(self, *a, **kw)
        if isinstance(getattr(self, "first_stage_model", None), (tv.ShapeVae, tv.TextureVae)) and self.vae_dtype != dt:
            self.vae_dtype = dt; self.first_stage_model.to(dt)
            print(f"[hip_gemm_chunk] {type(self.first_stage_model).__name__} set to {str(dt).split('.')[-1]}", flush=True)
    comfy.sd.VAE.__init__ = g_init
    dec = tv.TextureVae.decode_tex_slat
    def g_decode_tex_slat(self, *a, **kw):
        out = dec(self, *a, **kw)
        return out.float() if out.feats.dtype != torch.float32 else out
    tv.TextureVae.decode_tex_slat = g_decode_tex_slat


def selftest():
    """The boundary probe through the guarded ops: GPU vs CPU just past each limit."""
    torch.manual_seed(0); worst = 0.0
    x, w = torch.randn((1 << 19) + 1024, 64), torch.randn(32, 64)
    h, hw = torch.randn((1 << 20) + 1024, 16), torch.randn(16, 16)  # bf16 at K=N=16: a 16-row tile, wrong from 2^20
    a, b = torch.randn(65537 + 512, 4, 4), torch.randn(65537 + 512, 4, 4)
    A = torch.randn(65537 + 512, 3, 3) + 3 * torch.eye(3); y = torch.randn(65537 + 512, 3, 1)
    for gpu, ref in ((F.linear(x.cuda(), w.cuda()), F.linear(x, w)), (torch.bmm(a.cuda(), b.cuda()), torch.bmm(a, b)),
                     (x.cuda() @ w.cuda().T, x @ w.T), (torch.einsum("bij,bjk->bik", a.cuda(), b.cuda()), torch.einsum("bij,bjk->bik", a, b)),
                     (torch.linalg.solve(A.cuda(), y.cuda()), torch.linalg.solve(A, y)), (torch.det(A.cuda()), torch.det(A))):
        e = ((gpu.cpu() - ref).abs().max() / ref.abs().max()).item(); worst = max(worst, e if e == e else float("inf"))  # NaN fails
    lo = F.linear(h.cuda().bfloat16(), hw.cuda().bfloat16()).float().cpu(); ref = F.linear(h.bfloat16().float(), hw.bfloat16().float())
    lo_err = ((lo - ref).abs().max() / ref.abs().max()).item()  # bf16 rounding is ~4e-3; a wrapped tile is ~1
    return max(worst, 0.0 if lo_err <= 2e-2 else (lo_err if lo_err == lo_err else float("inf")))  # NaN fails too (inf)


if torch.version.hip and torch.cuda.is_available():
    install()
    try: trellis_vae_dtype()
    except Exception as e: print(f"[hip_gemm_chunk] TRELLIS.2 VAE dtype patch not applied: {e}", flush=True)
    try:
        err = selftest(); (log.warning if err > 1e-4 else log.info)(f"hip_gemm_chunk: guard installed, self-test max rel err {err:.2e}")
        print(f"[hip_gemm_chunk] large-GEMM guard installed, self-test max rel err {err:.2e}", flush=True)
    except Exception as e: print(f"[hip_gemm_chunk] guard installed, self-test failed to run: {e}", flush=True)

NODE_CLASS_MAPPINGS = {}
