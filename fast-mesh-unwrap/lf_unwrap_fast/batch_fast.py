# This file contains modified copies of functions from ComfyUI v0.43.2,
# comfy_extras/mesh3d/uv_unwrap/segment.py (cluster_charts_pec -> cluster_charts, _combine_normal_cones -> cone_half).
# ComfyUI is licensed under the GNU General Public License version 3; so is this file. The licence text is in
# LICENSE-GPL-3.0.txt beside this file. Changes: see the docstrings of the functions named above.
"""Low-overhead PEC clustering, chart boundary loops, normal-cone and edge-remap helpers, with checked installation.

Large-chart batching and the copied unwrap driver were dropped: batching saved nothing measurable, and the
copied driver could only disable these patches after a ComfyUI upgrade. Each installed entry point falls back to upstream
on an execution error; PEC's fallback runs upstream's whole clustering path (unwrap_fast.pec_guard).
"""
import functools

import numpy as np
import torch

try:
    from . import unwrap_fast as U
except ImportError:  # standalone benchmark/test imports
    import unwrap_fast as U


def boundary_loops(faces_subset, face_face_subset):
    """Vectorize the F*3 edge scan, retaining upstream insertion/overwrite order.

    Walking the boundary is sequential. Only boundary edges enter Python, usually
    tens/hundreds of edges instead of thousands of interior edges per chart.
    """
    faces = faces_subset.detach().cpu().numpy()
    ff = face_face_subset.detach().cpu().numpy()
    f, e = np.nonzero(ff == -1)
    next_v = dict(zip(faces[f, e].tolist(), faces[f, (e + 1) % 3].tolist()))
    loops, visited = [], set()
    for start in next_v:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = next_v.get(start)
        while cur is not None and cur != start:
            if cur in visited:
                break
            loop.append(cur)
            visited.add(cur)
            cur = next_v.get(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def remap_edges(edges, m, V):
    """NumPy's integer unique is cheaper than torch.unique for the CPU edge list."""
    if edges.device.type != "cpu" or m.device.type != "cpu":
        return remap_edges_tensor(edges, m, V)
    a, b = m.numpy()[edges.numpy()].T
    keep = a != b
    a, b = a[keep], b[keep]
    # np.unique's hash pass is very expensive on these almost-unique keys in
    # the installed NumPy 2.5.2. Sort and remove adjacent duplicates directly.
    key = np.sort(np.minimum(a, b) * V + np.maximum(a, b))
    key = key[np.r_[True, key[1:] != key[:-1]]] if key.size else key
    return torch.from_numpy(np.stack((key // V, key % V), axis=1))


def remap_edges_tensor(edges, m, V):
    """One label gather and one mask instead of two of each; same sorted keys."""
    a, b = m[edges].unbind(dim=1)
    key = torch.minimum(a, b) * V + torch.maximum(a, b)
    key = torch.unique(key[a != b], sorted=True)
    return torch.stack((key // V, key % V), dim=1)


def cone_half(axis_a, half_a, axis_b, half_b):
    """The cost-only portion of upstream _combine_normal_cones, exactly in torch.

    The upstream routine already vectorizes cones, but computes a normalized axis,
    sin and cos even when PEC discards it. Only winning edges need that work.
    """
    angle = torch.acos((axis_a * axis_b).sum(dim=-1).clamp(-1., 1.))
    low = torch.minimum(-half_a, angle - half_b)
    high = torch.maximum(half_a, angle + half_b)
    return (high - low) * .5


def cluster_charts(mesh, max_cost=.7, max_iters=1024, min_faces=8, progress_callback=None):
    """PEC with exact torch arithmetic, cheaper cost/edge updates and no .item()."""
    from comfy_extras.mesh3d.uv_unwrap import segment as S
    device = mesh.faces.device
    F = mesh.faces.shape[0]
    chart_id = torch.arange(F, dtype=torch.long, device=device)
    axis = mesh.face_normal.to(torch.float32).clone()
    half = torch.zeros(F, dtype=torch.float32, device=device)
    fel = S.face_edge_lengths(mesh.vertices.to(torch.float32), mesh.faces.to(torch.long))
    ff = mesh.face_face.to(torch.long)
    edges, _ = S._build_chart_edges(ff, chart_id, fel)
    # Winner labels are disjoint and always merge high into low. Track live
    # labels on the host, transferring only the small winner list once/round.
    # This is exactly max(chart_id)+1 without two device scalar reads/round.
    live = np.ones(F, dtype=bool)
    N = F
    active = F
    for _ in range(max_iters):
        if edges.shape[0] == 0:
            break
        a, b = edges[:, 0], edges[:, 1]
        cost = cone_half(axis[a], half[a], axis[b], half[b])
        E = edges.shape[0]
        key = (torch.clamp(cost * 1e6, max=2e9).to(torch.int64) << 32) | torch.arange(E, device=device)
        best = torch.full((N,), 2**62, dtype=torch.long, device=device)
        best.scatter_reduce_(0, a, key, reduce="amin", include_self=True)
        best.scatter_reduce_(0, b, key, reduce="amin", include_self=True)
        winners = (best[a] == key) & (best[b] == key) & (cost <= max_cost)
        wa, wb = a[winners], b[winners]
        if wa.numel() == 0:
            break
        active -= wa.numel()
        if progress_callback is not None:
            # Actual merges give monotonic progress without a device scalar read.
            progress_callback(F - active, F)
        new_axis, new_half, _ = S._combine_normal_cones(axis[wa], half[wa], axis[wb], half[wb])
        axis[wa], half[wa] = new_axis, new_half
        remap = torch.arange(N, dtype=torch.long, device=device)
        remap[wb] = wa
        chart_id = remap[chart_id]
        edges = remap_edges(edges, remap, N)
        live[wb.detach().cpu().numpy()] = False
        while N and not live[N - 1]:
            N -= 1
    if min_faces > 1:
        chart_id = S._merge_small_charts(chart_id, mesh.face_normal.to(torch.float32),
                                         mesh.face_area.to(torch.float32), ff, fel, min_faces, 2 * max_cost)
    return torch.unique(chart_id, sorted=True, return_inverse=True)[1]


def guarded(fast, orig):
    @functools.wraps(orig)
    def call(*a, **kw):
        try:
            return fast(*a, **kw)
        except Exception as e:
            msg = f"[lf_unwrap_fast] {orig.__name__} failed ({type(e).__name__}: {e}); upstream"
        # the retry runs after the handler, so the failed attempt's frames and arrays are released first
        print(msg, flush=True)
        return orig(*a, **kw)
    call._lf_batch_fast = True
    return call


def install():
    """Check every entry point before publishing any of this extension's patches; returns the number published (0 when
    already installed)."""
    from comfy_extras.mesh3d.uv_unwrap import mesh as M, segment as S
    if getattr(S.cluster_charts_pec, "_lf_batch_fast", False):
        return 0
    mesh = U.demo_mesh(12)
    up_cluster = getattr(S.cluster_charts_pec, "__wrapped__", S.cluster_charts_pec)
    up_merge = getattr(S._merge_small_charts, "__wrapped__", S._merge_small_charts)
    current = S._merge_small_charts
    S._merge_small_charts = up_merge
    try:
        ref = up_cluster(mesh, max_cost=1.)
    finally:
        S._merge_small_charts = current
    if not torch.equal(ref, cluster_charts(mesh, max_cost=1.)):
        raise RuntimeError("PEC self-test differs")
    if boundary_loops(mesh.faces, mesh.face_face) != M.chart_boundary_loops(mesh.faces, mesh.face_face):
        raise RuntimeError("boundary self-test differs")
    axes = mesh.face_normal
    h = torch.linspace(0., 1., len(axes))
    if not torch.equal(cone_half(axes, h, axes.roll(1, 0), h.roll(1)),
                       S._combine_normal_cones(axes, h, axes.roll(1, 0), h.roll(1))[1]):
        raise RuntimeError("cone-half self-test differs")
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [0, 1]])
    m = torch.tensor([0, 0, 2, 3])
    if not torch.equal(remap_edges(edges, m, 4), U._remap_edges(edges, m, 4)):
        raise RuntimeError("edge-remap self-test differs")
    if not torch.equal(remap_edges_tensor(edges, m, 4), U._remap_edges(edges, m, 4)):
        raise RuntimeError("tensor edge-remap self-test differs")
    M.chart_boundary_loops = guarded(boundary_loops, M.chart_boundary_loops)
    pec = U.pec_guard(cluster_charts, up_cluster, up_merge, S)
    pec._lf_batch_fast = pec._lf_unwrap_fast = True  # also marks unwrap_fast.install_pec as done (no downgrade on a re-run)
    S.cluster_charts_pec = pec
    return 2
