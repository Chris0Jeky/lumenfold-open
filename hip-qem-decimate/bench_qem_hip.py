"""lf_qem_hip proof (needs the GPU): exactness against ComfyUI's torch fallback, and speed, on a real high-poly mesh.

  set COMFYUI_ROOT=<ComfyUI root>
  python -s bench_qem_hip.py <mesh.npz with pos, tri> [--target 100000] [--mode all|onecall|fallback|kernel]

1. self_test (closed torus grid, degree cap raised so the fallback is exact).
2. One collapse-check call on every edge of the mesh with random targets: mismatches against the fallback at cap 64, and
   time against the fallback at the default cap (16).
3. qem_simplify to --target faces, fallback vs kernel: time, faces, Chamfer between the two results.
Prints `QEMHIP {json}` lines.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.argv[1:], args = [], sys.argv[1:]  # comfy.cli_args parses sys.argv at import
sys.path.insert(0, os.environ["COMFYUI_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parent / "lf_qem_hip"))
import numpy as np
import torch
import comfy_extras.mesh3d.postprocess.qem_decimate as q
import qem_hip


def emit(**kw): print("QEMHIP " + json.dumps(kw), flush=True)


def timed(fn, reps=3):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): out = fn()
    torch.cuda.synchronize(); return out, (time.perf_counter() - t) / reps


def chamfer(v1, v2, n=20000):
    g = torch.Generator(device=v1.device).manual_seed(0)
    a = v1[torch.randint(len(v1), (min(n, len(v1)),), device=v1.device, generator=g)]
    b = v2[torch.randint(len(v2), (min(n, len(v2)),), device=v2.device, generator=g)]
    d = lambda x, y: torch.cat([torch.cdist(x[i:i + 4096], y).min(1).values for i in range(0, len(x), 4096)]).mean()
    return float((d(a, b) + d(b, a)) / 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mesh", type=Path); ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--mode", choices=("all", "onecall", "fallback", "kernel"), default="all")
    a = ap.parse_args(args)
    if a.mode in ("all", "onecall"): emit(step="self_test", result=list(qem_hip.self_test()))
    m = np.load(a.mesh)
    verts = torch.tensor(m["pos"], dtype=torch.float32, device="cuda")
    faces = torch.tensor(m["tri"], dtype=torch.int64, device="cuda")
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = torch.unique(torch.sort(e, 1).values, dim=0)
    el = (verts[edges[:, 1]] - verts[edges[:, 0]]).norm(dim=1, keepdim=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    opt = (verts[edges[:, 0]] + verts[edges[:, 1]]) / 2 + el * torch.randn(edges.shape[0], 3, device="cuda", generator=g)

    if a.mode not in ("all", "onecall"): return simplify(a, verts, faces)
    (f1, s1, l1), t_k = timed(lambda: qem_hip.edge_collapse_checks(verts, faces, edges, opt, 0.0))
    v2f64 = q._build_vert_to_faces_pad(faces, verts.shape[0], 64)
    f2, s2, l2 = q._quality_checks_fused(verts, faces, edges, opt, v2f64, cos_threshold=0.0, want_link=True)
    _, t_f = timed(lambda: q._quality_checks_fused(verts, faces, edges, opt,
                                                    q._build_vert_to_faces_pad(faces, verts.shape[0], 16),
                                                    cos_threshold=0.0, want_link=True))
    emit(step="one_call", verts=verts.shape[0], faces=faces.shape[0], edges=edges.shape[0],
         flip_mismatch=int((f1 != f2).sum()), skinny_maxdiff=float((s1 - s2).abs().max()), link_mismatch=int((l1 != l2).sum()),
         max_degree=int(torch.bincount(faces.flatten()).max()),
         kernel_s=round(t_k, 4), fallback_cap16_s=round(t_f, 4), speedup=round(t_f / t_k, 1))

    if a.mode == "onecall": return
    simplify(a, verts, faces)


def simplify(a, verts, faces):
    orig = q._kitchen_op
    if a.mode in ("fallback", "kernel"):  # one path per process, to isolate a crash
        if a.mode == "kernel": q._kitchen_op = lambda n, t: qem_hip.edge_collapse_checks if n == "edge_collapse_checks" else orig(n, t)
        torch.cuda.synchronize(); t = time.perf_counter()
        v, f, _ = q.qem_simplify(verts, faces, a.target)
        torch.cuda.synchronize()
        emit(step="qem_simplify_" + a.mode, target=a.target, seconds=round(time.perf_counter() - t, 2), faces=int(f.shape[0]),
             chamfer_to_input=chamfer(v, verts))
        torch.save({"v": v.cpu(), "f": f.cpu()}, Path(a.mesh).with_name(f"qem-{a.mode}.pt"))
        return
    results = {}
    for name, op in (("fallback", orig), ("kernel", lambda n, t: qem_hip.edge_collapse_checks if n == "edge_collapse_checks" else orig(n, t))):
        q._kitchen_op = op
        torch.cuda.synchronize(); t = time.perf_counter()
        v, f, _ = q.qem_simplify(verts, faces, a.target)
        torch.cuda.synchronize(); results[name] = (v, f, time.perf_counter() - t)
    q._kitchen_op = orig
    (vf, ff, tf), (vk, fk, tk) = results["fallback"], results["kernel"]
    emit(step="qem_simplify", target=a.target, fallback_s=round(tf, 2), kernel_s=round(tk, 2), speedup=round(tf / tk, 2),
         fallback_faces=int(ff.shape[0]), kernel_faces=int(fk.shape[0]), chamfer_between=chamfer(vf, vk),
         chamfer_kernel_to_input=chamfer(vk, verts), chamfer_fallback_to_input=chamfer(vf, verts))


if __name__ == "__main__":
    main()
