# This file reimplements, vectorised, the behaviour of _compute_vertex_face_normals from ComfyUI v0.43.2,
# comfy_extras/nodes_mesh_postprocess.py, and was written against that function to be bit-identical to it.
# ComfyUI is licensed under the GNU General Public License version 3; so is this file. The licence text is in
# LICENSE-GPL-3.0.txt beside this file. Upstream copyright: the ComfyUI authors (github.com/comfyanonymous/ComfyUI).
"""Vectorised MeshSmoothNormals crease path. Bit-identical to ComfyUI's
`_compute_vertex_face_normals` (comfy_extras/nodes_mesh_postprocess.py, ~line 1037), which spends ~5 s per job in
per-face Python loops. The None / >=180 branch, an empty mesh, vertices other than float32/float64 and negative indices
call through to the original."""
import math
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def compute_vertex_face_normals_fast(vertices_np, faces_np, crease_angle=None, _orig=None):
    faces_i = faces_np.astype(np.int64)
    M = faces_i.shape[0]
    # Only the inputs the node feeds (float32 or float64 vertices, non-negative indices) take the fast path: float16
    # rounding can outrun the tie band below, and negative indices would collide in the edge keys.
    if (crease_angle is None or crease_angle >= 180.0 or M == 0 or vertices_np.dtype not in (np.float32, np.float64)
            or faces_i.min() < 0):
        return _orig(vertices_np, faces_np, crease_angle)

    tris = vertices_np[faces_i]
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = np.linalg.norm(face_n, axis=1, keepdims=True)
    face_unit = face_n / np.where(areas > 1e-12, areas, 1.0)
    cos_thresh = math.cos(math.radians(crease_angle))

    # Edges face-major (edge e belongs to face e // 3), so a stable sort keeps each edge's faces in face order.
    a, b, c = faces_i[:, 0], faces_i[:, 1], faces_i[:, 2]
    u = np.stack([a, b, c], 1).reshape(-1)
    v = np.stack([b, c, a], 1).reshape(-1)
    nv = int(faces_i.max()) + 1
    ekey = np.minimum(u, v) * nv + np.maximum(u, v)
    order = np.argsort(ekey, kind="stable")
    sk = ekey[order]
    start = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    counts = np.diff(np.r_[start, sk.size])
    two = start[counts == 2]                       # the original's `len(fl) == 2` rule
    f0, f1 = order[two] // 3, order[two + 1] // 3
    d = (face_unit[f0].astype(np.float64) * face_unit[f1]).sum(1)
    join = d >= cos_thresh
    # The original's np.dot rounds in the operand dtype; re-decide the razor-thin cases with the same call.
    for i in np.flatnonzero(np.abs(d - cos_thresh) < 1e-5):
        join[i] = float(np.dot(face_unit[f0[i]], face_unit[f1[i]])) >= cos_thresh
    g = coo_matrix((np.ones(int(join.sum()), np.int8), (f0[join], f1[join])), shape=(M, M))
    ngroups, label = connected_components(g, directed=False)

    # One output vertex per (original vertex, group), numbered by first occurrence in the face-major corner scan.
    flat = faces_i.reshape(-1)
    keys = flat * ngroups + np.repeat(label, 3)
    _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(first.size)
    out_faces = rank[inv.reshape(-1)].reshape(M, 3)
    remap = flat[np.sort(first)]

    normals = np.zeros((remap.shape[0], 3), dtype=np.float64)
    for k in range(3):
        np.add.at(normals, out_faces[:, k], face_n)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lens > 1e-12, lens, 1.0)
    return (vertices_np[remap], out_faces.astype(np.uint32), normals.astype(np.float32), remap)


def _selftest_mesh():
    """UV sphere plus a duplicated face and a third face on one edge (non-manifold), float32 like the node feeds."""
    n, m = 12, 16
    th = np.linspace(0, np.pi, n + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, m, endpoint=False)
    v = [[0, 0, 1]] + [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in th for p in ph] + [[0, 0, -1]]
    f = [[0, 1 + j, 1 + (j + 1) % m] for j in range(m)]
    for i in range(n - 2):
        for j in range(m):
            p, q = 1 + i * m + j, 1 + i * m + (j + 1) % m
            f += [[p, p + m, q], [q, p + m, q + m]]
    last = 1 + (n - 2) * m
    f += [[last + j, len(v) - 1, last + (j + 1) % m] for j in range(m)]
    f += [f[5], [f[7][0], f[7][1], 0]]
    return np.asarray(v, np.float32), np.asarray(f, np.int64)


def selftest(orig):
    v, f = _selftest_mesh()
    for ang in (10.0, 30.0, 60.0, 89.0):  # 10: the sphere's edges split; above 30 they join
        a, b = orig(v, f, ang), compute_vertex_face_normals_fast(v, f, ang, orig)
        if not all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b)):
            return False
    return True


def node_modules():
    """Every loaded copy of comfy_extras/nodes_mesh_postprocess.py, whatever its sys.modules key."""
    return [m for m in list(sys.modules.values())
            if str(getattr(m, "__file__", None) or "").replace("\\", "/").lower().endswith("comfy_extras/nodes_mesh_postprocess.py")]


def install():
    """Patch `_compute_vertex_face_normals` in every loaded copy of the module (idempotent); returns the number
    patched. Raises, patching nothing, if no copy is loaded or the fast version differs from upstream on the self-test."""
    mods = node_modules()
    if not mods:
        raise RuntimeError("comfy_extras/nodes_mesh_postprocess.py is not loaded")
    todo = [m for m in mods if not getattr(m._compute_vertex_face_normals, "_lf_normals_fast", False)]
    for m in todo:
        if not selftest(m._compute_vertex_face_normals):
            raise RuntimeError("self-test differs from upstream")
    for m in todo:
        def patched(vertices_np, faces_np, crease_angle=None, _orig=m._compute_vertex_face_normals):
            return compute_vertex_face_normals_fast(vertices_np, faces_np, crease_angle, _orig)
        patched.__wrapped__ = m._compute_vertex_face_normals; patched._lf_normals_fast = True
        m._compute_vertex_face_normals = patched
    return len(todo)
