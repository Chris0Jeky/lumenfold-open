# This file contains code adapted from ComfyUI v0.43.2: comfy_extras/mesh3d/uv_unwrap/parameterize.py (lscm_charts_batch:
# pin selection and row coefficients, reused in _sparse) and comfy_extras/mesh3d/uv_unwrap/segment.py (cluster_charts_pec
# -> cluster_charts_pec_fast, _merge_small_charts -> merge_small_charts_fast).
# ComfyUI is licensed under the GNU General Public License version 3; so is this file. The licence text is in
# LICENSE-GPL-3.0.txt beside this file. Upstream copyright: the ComfyUI authors (github.com/comfyanonymous/ComfyUI). Changes: see the docstrings of the functions named above.
"""Faster UnwrapMesh for ComfyUI 0.43.2, as a drop-in patch: the batched LSCM solved sparse instead of dense.

ComfyUI's UV unwrap (`comfy_extras/mesh3d/uv_unwrap`) spends most of a TRELLIS mesh job on the CPU while the GPU
waits (0.43.2 log: 268 s of a 470 s job, 177 s of 254 s). The largest phase is `lscm_charts_batch`: thousands
of small charts, each an ABF/LSCM least-squares problem with about 6 non-zeros per row, solved as dense padded normal
equations in chunks of up to 128 charts x 512 x 512 (forced to the CPU on Windows ROCm). That is roughly 100x more
arithmetic than the problem has.

`lscm_charts_sparse` builds the same rows (ABF for valid faces, plain LSCM for degenerate ones), the same two
farthest-point pins per chart and the same 1e-12 x median-diagonal regularisation, then solves every chart at once as
one block-diagonal sparse system (SuperLU). Same equations, different elimination order: results agree to round-off
(see README.md), not bit for bit. If the sparse factorisation fails, it falls back to the original function.

    import unwrap_fast; unwrap_fast.install()            # before UnwrapMesh runs

The PEC chart clustering (`segment.cluster_charts_pec`, `_merge_small_charts`) rebuilt the chart adjacency from all 3F
face adjacencies every merge iteration (about 2.3 s per job). `cluster_charts_pec_fast` builds it once and then maps
the chart-pair list through each merge's remap instead: the adjacency after a merge is exactly the image of the previous
one minus self-loops, and the same key sort keeps the edge order, so face_chart is identical (the min-edge tie-break
uses the edge's row index). Callers read `_uv_seg.cluster_charts_pec` off the module at call time, so the swap reaches them.
"""
import numpy as np
try:
    import torch  # the PEC path only (it runs inside ComfyUI); the LSCM path and its CI test need numpy and scipy alone
except ImportError:
    torch = None
import scipy.sparse as sp
import scipy.sparse.linalg as spla

GROUP_CHARTS = 2048  # charts per sparse solve: bounds memory and commit at any mesh size


def lscm_charts_sparse(verts, uv_pins, faces_gl, face_pos, chart_of_face, chart_of_vert, vert_offsets, chart_ids,
                       n_charts, max_bucket_verts=256, device=None, *, _orig=None):
    """Drop-in for `parameterize.lscm_charts_batch` (0.43.2): {chart_id: (Vc, 2) float32} for charts with at most
    `max_bucket_verts` vertices; larger charts are left to the caller, as upstream does. Any failure (assembly memory,
    a singular factorisation, a non-finite result) falls back to `_orig`, the upstream function, when given."""
    args = (verts, uv_pins, faces_gl, face_pos, chart_of_face, chart_of_vert, vert_offsets, chart_ids, n_charts,
            max_bucket_verts, device)
    try:
        return _sparse(*args)
    except Exception as e:
        if _orig is None:
            raise
        print(f"[lf_unwrap_fast] sparse LSCM failed ({type(e).__name__}: {e}); upstream dense path for {chart_ids.size} charts",
              flush=True)
    return _orig(*args)  # outside the except: the failed attempt's arrays are freed first


def _sparse(verts, uv_pins, faces_gl, face_pos, chart_of_face, chart_of_vert, vert_offsets, chart_ids, n_charts,
            max_bucket_verts, device, *, _pins=None, _regularize=True):
    from comfy_extras.mesh3d.uv_unwrap import parameterize as P
    out = {}
    vcounts = np.diff(vert_offsets)
    small = chart_ids[vcounts[chart_ids] <= max_bucket_verts] if chart_ids.size else chart_ids
    if small.size == 0:
        return out
    sel = np.zeros(n_charts, bool); sel[small] = True

    # vertices of the selected charts, in global order; compact index k <-> global index vg[k]
    vg = np.nonzero(sel[chart_of_vert])[0]
    V = vg.size
    comp = np.full(chart_of_vert.size, -1, np.int64); comp[vg] = np.arange(V)

    # pins: upstream's two farthest-point passes, per chart
    cv = chart_of_vert[vg]
    if _pins is None:
        first = vert_offsets[:-1]
        d0 = ((verts[vg] - verts[first[cv]]) ** 2).sum(1)
        pin_a = P._segment_argmax(d0, cv, n_charts); pin_a = np.where(pin_a >= 0, vg[pin_a.clip(min=0)], -1)
        d1 = ((verts[vg] - verts[pin_a.clip(min=0)[cv]]) ** 2).sum(1)
        pin_b = P._segment_argmax(d1, cv, n_charts); pin_b = np.where(pin_b >= 0, vg[pin_b.clip(min=0)], -1)
        alt = np.where(pin_a == first, first + 1, first)
        pin_b = np.where(pin_a == pin_b, alt, pin_b)
    else:
        # Large-chart caller supplies upstream's boundary-extremal pins.
        pin_a, pin_b = _pins

    # rows: two per face, ABF where valid, plain LSCM where degenerate (upstream's coefficients and signs)
    f_ids = np.nonzero(sel[chart_of_face])[0]
    ids, cosf, sinf, valid = P._abf_face_coefficients(verts, faces_gl[f_ids])
    rr = 2 * np.arange(f_ids.size)
    R, C, X = [], [], []
    def put(rows, cols, vals): R.append(rows); C.append(cols); X.append(vals)
    i = np.nonzero(valid)[0]
    if i.size:
        k = comp[ids[i]]; c, s, r, one = cosf[i], sinf[i], rr[i], np.ones(i.size)
        for cols, vals in ((k[:, 0], c - 1.0), (k[:, 0] + V, -s), (k[:, 1], -c), (k[:, 1] + V, s), (k[:, 2], one)):
            put(r, cols, vals)
        for cols, vals in ((k[:, 0], s), (k[:, 0] + V, c - 1.0), (k[:, 1], -s), (k[:, 1] + V, -c), (k[:, 2] + V, one)):
            put(r + 1, cols, vals)
    j = np.nonzero(~valid)[0]
    if j.size:
        fj = faces_gl[f_ids[j]]
        t = P._triangle_local_2d(verts, fj)
        w = 1.0 / np.sqrt(2.0 * np.abs(t[:, 1, 0] * t[:, 2, 1] - t[:, 1, 1] * t[:, 2, 0]).clip(min=1e-20))
        k, r = comp[fj], rr[j]
        for q in range(3):
            q1, q2 = (q + 1) % 3, (q + 2) % 3
            aq = (t[:, q1, 0] - t[:, q2, 0]) * w; bq = (t[:, q1, 1] - t[:, q2, 1]) * w
            put(r, k[:, q], aq); put(r, k[:, q] + V, -bq); put(r + 1, k[:, q], bq); put(r + 1, k[:, q] + V, aq)
    A = sp.csr_matrix((np.concatenate(X), (np.concatenate(R), np.concatenate(C))), shape=(2 * f_ids.size, 2 * V))
    del R, C, X

    # pins move to the right-hand side; every other column is free
    pa, pb = comp[pin_a[small]], comp[pin_b[small]]
    pin_cols = np.concatenate([pa, pb, pa + V, pb + V])
    pin_vals = np.concatenate([uv_pins[pin_a[small], 0], uv_pins[pin_b[small], 0],
                               uv_pins[pin_a[small], 1], uv_pins[pin_b[small], 1]])
    free = np.ones(2 * V, bool); free[pin_cols] = False
    fc = np.nonzero(free)[0]
    Af = A[:, fc].tocsc()
    rhs = -(A[:, pin_cols] @ pin_vals)
    AtA = (Af.T @ Af).tocsc()
    Atb = Af.T @ rhs
    del A, Af

    # regularise free columns by 1e-12 x the median positive diagonal of their chart (upstream's eps)
    ch = np.concatenate([cv, cv])[fc]
    if _regularize:
        dg = AtA.diagonal()
        pos = dg > 0
        order = np.lexsort((dg[pos], ch[pos]))
        vals, chs = dg[pos][order], ch[pos][order]
        starts = np.searchsorted(chs, np.arange(n_charts)); ends = np.searchsorted(chs, np.arange(n_charts), "right")
        n = ends - starts
        lo = np.clip(starts + (n - 1) // 2, 0, max(vals.size - 1, 0)); hi = np.clip(starts + n // 2, 0, max(vals.size - 1, 0))
        med = np.where(n > 0, 0.5 * (vals[lo] + vals[hi]) if vals.size else 0.0, np.nan)
        dsc = 1e-12 * np.nan_to_num(med, nan=1e-8).clip(min=1e-20)  # upstream's order: clamp the median, then scale
        AtA = AtA + sp.diags(dsc[ch], format="csc")

    # The system is block diagonal (charts never couple), so solving it in groups of charts gives the same answer while
    # bounding SuperLU's up-front reservation (about 30x the non-zeros, all of it Windows commit charge).
    x = np.empty(fc.size)
    order = np.argsort(ch, kind="stable")                   # free columns, chart-major
    if _regularize:
        bounds = np.searchsorted(ch[order], np.arange(0, n_charts + GROUP_CHARTS, GROUP_CHARTS))
    else:
        # Large chart IDs are sparse in the full atlas. Bound by the number of
        # selected charts, not their largest label, to actually batch them.
        bounds = np.r_[np.searchsorted(ch[order], np.sort(small)[::GROUP_CHARTS]), fc.size]
    for g0, g1 in zip(bounds[:-1], bounds[1:]):
        if g1 > g0:
            idx = order[g0:g1]
            xg = spla.spsolve(AtA[idx][:, idx].tocsc(), Atb[idx])
            if not np.all(np.isfinite(xg)):
                raise RuntimeError("non-finite sparse LSCM solution")
            x[idx] = xg
    full = np.empty(2 * V); full[fc] = x; full[pin_cols] = pin_vals
    uv = np.stack([full[:V], full[V:]], 1).astype(np.float32)
    for c in small:  # each chart's vertices are contiguous in compact order
        k0 = comp[vert_offsets[c]]
        out[int(c)] = uv[k0:k0 + vcounts[c]]
    return out


# ---- PEC chart clustering: incremental chart-pair list --------------------------------------------------------------

def _remap_edges(edges, m, V):
    """Image of a chart-pair list under the chart map `m` (ids < V afterwards): self-loops dropped, rows normalised to
    (lo, hi), unique and sorted by lo*V+hi, exactly the order `_build_chart_edges` yields."""
    a, b = m[edges[:, 0]], m[edges[:, 1]]
    d = a != b
    a, b = a[d], b[d]
    key = torch.unique(torch.minimum(a, b) * V + torch.maximum(a, b))
    return torch.stack([key // V, key % V], 1)


def merge_small_charts_fast(chart_id, face_normal, face_area, face_face, face_edge_len, min_faces, cost_cap):
    """`segment._merge_small_charts` with the chart-pair list built once and mapped through each round's remap and id
    compaction (upstream rebuilds it from the faces every round). Same result."""
    from comfy_extras.mesh3d.uv_unwrap import segment as S
    if chart_id.numel() == 0:
        return chart_id
    device = chart_id.device
    edges = None
    for _ in range(16):
        N = int(chart_id.max().item()) + 1
        sizes = torch.bincount(chart_id, minlength=N)
        axis = torch.zeros(N, 3, dtype=torch.float32, device=device)
        axis.index_add_(0, chart_id, face_normal * face_area[:, None])
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dev = torch.acos((face_normal * axis[chart_id]).sum(1).clamp(-1.0, 1.0))
        half = torch.zeros(N, dtype=torch.float32, device=device)
        half.scatter_reduce_(0, chart_id, dev, reduce="amax")

        if edges is None:
            edges, _ = S._build_chart_edges(face_face, chart_id, face_edge_len)
        if edges.shape[0] == 0:
            break
        a, b = edges[:, 0], edges[:, 1]
        _, new_half, _ = S._combine_normal_cones(axis[a], half[a], axis[b], half[b])
        ok = new_half <= cost_cap
        E = edges.shape[0]
        key = (torch.clamp(new_half * 1e6, max=2e9).to(torch.int64) << 32) \
            | torch.arange(E, dtype=torch.long, device=device)
        best = torch.full((N,), 1 << 62, dtype=torch.long, device=device)
        va = (sizes[a] < min_faces) & ok
        vb = (sizes[b] < min_faces) & ok
        best.scatter_reduce_(0, a[va], key[va], reduce="amin")
        best.scatter_reduce_(0, b[vb], key[vb], reduce="amin")
        src = (best < (1 << 62)).nonzero(as_tuple=True)[0]
        if src.numel() == 0:
            break
        eid = best[src] & 0xFFFFFFFF
        ea, eb = a[eid], b[eid]
        tgt = torch.where(ea == src, eb, ea)
        prop = torch.arange(N, dtype=torch.long, device=device)
        prop[src] = tgt
        keepm = (prop[tgt] == tgt) | (src > tgt)
        remap = torch.arange(N, dtype=torch.long, device=device)
        remap[src[keepm]] = tgt[keepm]
        for _ in range(32):
            nr = remap[remap]
            if torch.equal(nr, remap):
                break
            remap = nr
        chart_id = remap[chart_id]
        uniq, chart_id = torch.unique(chart_id, return_inverse=True)
        # every chart that has an edge has faces, so its remapped id is in uniq: searchsorted is the compaction map
        edges = _remap_edges(edges, torch.searchsorted(uniq, remap).clamp_max(uniq.numel() - 1), uniq.numel())
    return chart_id


def cluster_charts_pec_fast(mesh, max_cost=0.7, max_iters=1024, min_faces=8, progress_callback=None):
    """Drop-in for `segment.cluster_charts_pec` (0.43.2): same signature, same face_chart, bit for bit."""
    from comfy_extras.mesh3d.uv_unwrap import segment as S
    device = mesh.faces.device
    F = mesh.faces.shape[0]
    faces = mesh.faces.to(torch.long)
    vertices = mesh.vertices.to(torch.float32)
    face_normal = mesh.face_normal.to(torch.float32)
    face_face = mesh.face_face.to(torch.long)
    face_edge_len = S.face_edge_lengths(vertices, faces)

    chart_id = torch.arange(F, dtype=torch.long, device=device)
    chart_axis = face_normal.clone()
    chart_half = torch.zeros(F, dtype=torch.float32, device=device)
    edges = None

    for it in range(max_iters):
        if edges is None:
            edges, _ = S._build_chart_edges(face_face, chart_id, face_edge_len)
        if edges.shape[0] == 0:
            break
        a = edges[:, 0]
        b = edges[:, 1]
        _, new_half, _ = S._combine_normal_cones(chart_axis[a], chart_half[a], chart_axis[b], chart_half[b])
        cost = new_half.clone()

        E = edges.shape[0]
        N = int(chart_id.max().item()) + 1
        edge_ids = torch.arange(E, dtype=torch.long, device=device)
        key = (torch.clamp(cost * 1e6, max=2e9).to(torch.int64) << 32) | edge_ids
        chart_min = torch.full((N,), (2**62), dtype=torch.long, device=device)
        chart_min.scatter_reduce_(0, a, key, reduce="amin", include_self=True)
        chart_min.scatter_reduce_(0, b, key, reduce="amin", include_self=True)

        winners = (chart_min[a] == key) & (chart_min[b] == key) & (cost <= max_cost)
        n_merge = int(winners.sum().item())
        if n_merge == 0:
            break
        if progress_callback is not None:
            progress_callback(F - N + n_merge, F)

        win_a = a[winners]
        win_b = b[winners]
        new_axis, new_half_w, _ = S._combine_normal_cones(
            chart_axis[win_a], chart_half[win_a], chart_axis[win_b], chart_half[win_b])
        chart_axis[win_a] = new_axis
        chart_half[win_a] = new_half_w

        remap = torch.arange(N, dtype=torch.long, device=device)
        remap[win_b] = win_a
        chart_id = remap[chart_id]
        edges = _remap_edges(edges, remap, N)

    if min_faces > 1:
        chart_id = S._merge_small_charts(chart_id, face_normal, mesh.face_area.to(torch.float32),
                                         face_face, face_edge_len, min_faces, 2.0 * max_cost)
    _, inverse = torch.unique(chart_id, sorted=True, return_inverse=True)
    return inverse


def grid_mesh(v, wrap_rows=False):
    """MeshData of an (R, C, 3) vertex grid wrapped around the columns (and the rows when `wrap_rows`): a tube, or a
    torus. Two triangles per cell."""
    from comfy_extras.mesh3d.uv_unwrap.mesh import build_mesh
    R, C = v.shape[:2]
    idx = np.arange(R * C).reshape(R, C)
    a = idx if wrap_rows else idx[:-1]
    b = np.roll(idx, -1, 0) if wrap_rows else idx[1:]
    a2, b2 = np.roll(a, -1, 1), np.roll(b, -1, 1)
    f = np.concatenate([np.stack([a, b, a2], -1).reshape(-1, 3), np.stack([a2, b, b2], -1).reshape(-1, 3)])
    return build_mesh(torch.from_numpy(v.reshape(-1, 3).astype(np.float32)), torch.from_numpy(f.astype(np.int64)))


def demo_mesh(n=24):
    """A slightly noisy UV sphere (about 4n^2 faces) as a MeshData, for the load-time self-test."""
    rng = np.random.default_rng(1)
    th, ph = np.meshgrid(np.linspace(0.05, np.pi - 0.05, n), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), indexing="ij")
    v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    return grid_mesh(v * (1 + rng.normal(0, 0.03, (n, 2 * n, 1))))


def pec_selftest(S):
    """True when the fast clustering (with the fast small-chart merge) equals upstream's on `demo_mesh`; leaves S as found."""
    mesh = demo_mesh()
    ref = S.cluster_charts_pec(mesh, max_cost=1.0)
    orig_merge = S._merge_small_charts
    S._merge_small_charts = merge_small_charts_fast
    try:
        new = cluster_charts_pec_fast(mesh, max_cost=1.0)
    finally:
        S._merge_small_charts = orig_merge
    return bool(torch.equal(ref, new)) and int(ref.max()) > 1


def pec_guard(fast, up_cluster, up_merge, S):
    """`fast` as segment.cluster_charts_pec, with upstream's whole path as the fallback: on an error, upstream's
    cluster_charts_pec runs with upstream's _merge_small_charts swapped back in for that call (it looks the merge up on the
    module, where install_pec put the fast one, so a plain retry would hit the same failure)."""
    import functools

    @functools.wraps(up_cluster)
    def cluster_charts_pec(*a, **kw):
        try:
            return fast(*a, **kw)
        except Exception as e:
            msg = f"[lf_unwrap_fast] fast PEC clustering failed ({type(e).__name__}: {e}); upstream"
        print(msg, flush=True)  # the retry runs after the handler, so the failed attempt's frames are released first
        current = S._merge_small_charts
        S._merge_small_charts = up_merge
        try:
            return up_cluster(*a, **kw)
        finally:
            S._merge_small_charts = current
    cluster_charts_pec.__wrapped__ = up_cluster
    return cluster_charts_pec


def install_pec():
    """Patch `segment.cluster_charts_pec` and `._merge_small_charts` (idempotent), only on an identical self-test result."""
    from comfy_extras.mesh3d.uv_unwrap import segment as S
    if getattr(S.cluster_charts_pec, "_lf_unwrap_fast", False):
        return True
    if not pec_selftest(S):
        raise RuntimeError("PEC self-test: fast clustering differs from upstream")
    up_cluster, up_merge = S.cluster_charts_pec, S._merge_small_charts
    merge_small_charts_fast.__wrapped__ = up_merge
    guarded = pec_guard(cluster_charts_pec_fast, up_cluster, up_merge, S)
    guarded._lf_unwrap_fast = True
    S._merge_small_charts = merge_small_charts_fast
    S.cluster_charts_pec = guarded
    return True


EXPECTED = ("verts", "uv_pins", "faces_gl", "face_pos", "chart_of_face", "chart_of_vert", "vert_offsets", "chart_ids",
            "n_charts", "max_bucket_verts", "device")


def demo_charts(seed=0, sizes=(4, 7, 12)):
    """Three bent, jittered grid charts in the layout `lscm_charts_batch` takes (for the load-time self-test)."""
    rng = np.random.default_rng(seed)
    verts, faces, cof, cov, offs = [], [], [], [], [0]
    for c, n in enumerate(sizes):
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        verts.append(np.stack([u, v, 0.3 * np.sin(3 * u + c) * v], -1).reshape(-1, 3) + rng.normal(0, 0.01, (n * n, 3)))
        b = offs[-1]
        for i in range(n - 1):
            for j in range(n - 1):
                a = b + i * n + j
                faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]; cof += [c, c]
        cov += [c] * (n * n); offs.append(b + n * n)
    verts = np.concatenate(verts); faces = np.asarray(faces, np.int64); cof = np.asarray(cof, np.int64)
    first = np.searchsorted(cof, np.arange(len(sizes)))
    return (verts, verts[:, :2].copy(), faces, np.arange(len(cof)) - first[cof], cof, np.asarray(cov, np.int64),
            np.asarray(offs, np.int64), np.arange(len(sizes)), len(sizes))


def selftest(orig):
    """Max UV difference between upstream and sparse on `demo_charts`, relative to each chart's span."""
    args = demo_charts()
    ref, new = orig(*args), lscm_charts_sparse(*args)
    if set(ref) != set(new):
        return float("inf")
    return max(float(np.abs(new[c] - u).max() / (np.ptp(u, axis=0).max() or 1.0)) for c, u in ref.items())


def install():
    """Patch ComfyUI's parameterize module in place (idempotent), only if its LSCM signature is the one this was
    written against (0.43.2) and the sparse solve matches upstream on the self-test. Returns the self-test error."""
    import inspect
    try:
        install_pec()
    except Exception as e:  # independent of the LSCM patch: upstream's clustering stays if this fails
        print(f"[lf_unwrap_fast] PEC clustering patch not installed: {type(e).__name__}: {e}", flush=True)
    from comfy_extras.mesh3d.uv_unwrap import parameterize as P
    if getattr(P.lscm_charts_batch, "_lf_unwrap_fast", False):
        return 0.0
    orig = P.lscm_charts_batch
    names = tuple(inspect.signature(orig).parameters)
    if names != EXPECTED:
        raise RuntimeError(f"lscm_charts_batch signature changed: {names}")
    err = selftest(orig)
    if not err < 1e-5:
        raise RuntimeError(f"self-test differs from upstream by {err:.2e}")
    def lscm_charts_batch(*a, **kw):
        return lscm_charts_sparse(*a, _orig=orig, **kw)
    lscm_charts_batch.__wrapped__ = orig; lscm_charts_batch._lf_unwrap_fast = True
    P.lscm_charts_batch = lscm_charts_batch
    return err
