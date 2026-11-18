"""lf_qem_hip, part 2: QEM's sorted half-edge table without its full-size temporaries.

ComfyUI's `qem_decimate._sorted_edge_halfedges` builds the 3F half-edges as a (3F, 2) int64 tensor, sorts each row into
another, keeps `arange(F).repeat(3)` and `arange(3).repeat_interleave(F)` id tensors, and gathers all three by the key's
argsort. On a ~20M-face mesh that is DecimateMesh's peak: 9.18 GiB of torch memory, 5.3 GiB of it these temporaries
(allocator history), and the job then spills past WDDM's budget.

This returns the same three tensors. Each half-edge's key min(a,b)*(V+1)+max(a,b) is written slot by slot into one 3F
int64 tensor (the same values upstream's row sort gives), argsort runs on that same key tensor, and the face and slot ids
are the sort index modulo and divided by F (upstream's id tensors hold exactly i % F and i // F at position i).
Pinned to upstream's source; checked against it on install (CPU, small meshes). Published on qem_decimate's module global,
which DecimateMesh's collapse loop reads at call time; remesh.py imports the function by name at load, so Remesh keeps
upstream's (same values, the old peak).
"""
import ast
import hashlib
import inspect

import torch

UPSTREAM_SHA256 = "cfc46081699baf073041e2e730e62394ddc6cb408fa695d953e6357cc68ddf37"


def sorted_edge_halfedges(faces, num_verts):
    """3F half-edges sorted by key min(a,b)*(V+1)+max(a,b); returns (sorted_keys, face_ids, slot_ids)."""
    F = faces.shape[0]
    P = num_verts + 1
    key = torch.empty(3 * F, dtype=torch.long, device=faces.device)
    for s in range(3):
        a, b = faces[:, s], faces[:, (s + 1) % 3]
        k = key[s * F:(s + 1) * F]
        torch.mul(torch.minimum(a, b).long(), P, out=k)
        k.add_(torch.maximum(a, b).long())
    sort_idx = torch.argsort(key)
    sorted_keys = key[sort_idx]
    del key, k, a, b  # the slot view k alone would keep the key's storage alive
    if F == 0:
        return sorted_keys, sort_idx.clone(), sort_idx
    slot = torch.div(sort_idx, F, rounding_mode="floor")
    face = sort_idx.remainder_(F)  # in place: the sort index is not needed after this
    return sorted_keys, face, slot


def upstream_sha(fn):
    try:
        src = inspect.getsource(inspect.getmodule(fn))
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == fn.__name__)
        return hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()
    except (OSError, TypeError, StopIteration):
        return None


def selftest(orig, fn=sorted_edge_halfedges):
    """True when fn returns exactly what upstream does on small CPU meshes (shared edges, an empty mesh)."""
    g = torch.Generator().manual_seed(0)
    for F, V in ((0, 4), (1, 3), (257, 40), (4096, 600)):
        faces = torch.randint(0, V, (F, 3), generator=g, dtype=torch.long, device="cpu")  # whatever the default device
        for dtype in (torch.long, torch.int32):
            a, b = orig(faces.to(dtype), V), fn(faces.to(dtype), V)
            if not all(torch.equal(x, y) for x, y in zip(a, b)):
                return False
    return True


def install():
    from comfy_extras.mesh3d.postprocess import qem_decimate as Q
    orig = Q._sorted_edge_halfedges
    if getattr(orig, "lf_edges_lowpeak", False):
        return "already installed"
    if hasattr(orig, "__wrapped__") or upstream_sha(orig) != UPSTREAM_SHA256:
        raise RuntimeError("qem_decimate._sorted_edge_halfedges is not upstream's (patched elsewhere or source changed)")
    if not selftest(orig):
        raise RuntimeError("self-test against upstream failed")
    fn = lambda faces, num_verts: sorted_edge_halfedges(faces, num_verts)
    fn.lf_edges_lowpeak = True
    fn.__wrapped__ = orig
    Q._sorted_edge_halfedges = fn
    return "QEM sorted half-edges without the (3F, 2) temporaries, self-test exact"
