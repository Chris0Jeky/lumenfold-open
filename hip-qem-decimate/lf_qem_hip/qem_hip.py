"""AMD-native `edge_collapse_checks` for ComfyUI's QEM decimation.

comfy-kitchen ships the exact collapse-check kernel only in its CUDA backend, and `qem_decimate._kitchen_op` asks only
that backend, so on ROCm every QEM iteration takes the torch fallback: a padded vertex-to-face table rebuilt each
iteration, gathered (E, 2*16, 3) per chunk, with vertex degree capped at 16. On one mesh that was 31.7 of DecimateMesh's
44.9 s. This module builds the incident-face lists as compressed rows (atomic count,
cumsum, atomic fill: no sort, no degree cap) and runs one thread per edge for the same three checks as the fallback:

  flips[e]   faces around a or b (not both) whose normal turns past cos_threshold when a, b move to opt[e]
  skinny[e]  mean of (1 - shape quality) over those faces, shape = 2*sqrt(3)*|n| / (sum of squared edge lengths)
  link[e]    common neighbours of a and b (excluding a, b) number no more than the faces on edge ab; an endpoint with
             more than LINK_MAX_DEG (256) incident faces is refused outright (link 0: no collapse), bounding the
             O(deg^2) per-thread work, where the fallback instead drops faces past degree 16

Same math and thresholds as `_quality_checks_fused`; only the order of the skinny sum differs (float rounding).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hipkernel import Kernel, I64  # noqa: E402

LINK_MAX_DEG = 256  # a vertex with more incident faces refuses its collapses (the link check is O(deg^2) per thread)

SRC = f"#define LINK_MAX_DEG {LINK_MAX_DEG}\n" + r"""
extern "C" __global__ void vf_count(const long long* __restrict__ faces, long long F, int* __restrict__ deg) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * F) return;
    atomicAdd(&deg[faces[i]], 1);
}

extern "C" __global__ void vf_fill(const long long* __restrict__ faces, long long F, const long long* __restrict__ off,
                                   int* __restrict__ cursor, int* __restrict__ adj) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * F) return;
    long long v = faces[i];
    int slot = atomicAdd(&cursor[v], 1);
    adj[off[v] + slot] = (int)(i / 3);
}

__device__ inline float3 ld(const float* v, long long i) { return make_float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]); }
__device__ inline float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Faces of one endpoint: count flips and skinny terms over faces holding exactly one of (a, b).
__device__ inline void ring(const float* verts, const long long* faces, const long long* off, const int* adj,
                            long long v, long long a, long long b, float3 o, float cos_t,
                            int& flips, float& skin, int& naff) {
    const float k = 2.0f * 1.7320508075688772f;  // 4*sqrt(3)*0.5, as the torch fallback
    for (long long j = off[v]; j < off[v + 1]; ++j) {
        long long f = adj[j];
        long long i0 = faces[3 * f], i1 = faces[3 * f + 1], i2 = faces[3 * f + 2];
        bool s0 = i0 == a || i0 == b, s1 = i1 == a || i1 == b, s2 = i2 == a || i2 == b;
        bool ca = i0 == a || i1 == a || i2 == a, cb = i0 == b || i1 == b || i2 == b;
        if (ca == cb) continue;  // holds both (collapses away) or neither
        float3 p0 = ld(verts, i0), p1 = ld(verts, i1), p2 = ld(verts, i2);
        float3 q0 = s0 ? o : p0, q1 = s1 ? o : p1, q2 = s2 ? o : p2;
        float3 e01 = sub(q1, q0), e02 = sub(q2, q0), e12 = sub(q2, q1);
        float3 nn = cross3(e01, e02);
        float ln = sqrtf(dot3(nn, nn));
        float es = dot3(e01, e01) + dot3(e02, e02) + dot3(e12, e12);
        float3 d01 = sub(p1, p0), d02 = sub(p2, p0), d12 = sub(p2, p1);
        float3 no = cross3(d01, d02);
        float lo = sqrtf(dot3(no, no));
        float os = dot3(d01, d01) + dot3(d02, d02) + dot3(d12, d12);
        bool usable = lo > 1e-6f * os && ln > 1e-6f * es;
        if (usable && dot3(no, nn) < cos_t * lo * ln) flips += 1;
        float shape = k * ln / fmaxf(es, 1e-20f);
        skin += 1.0f - fminf(fmaxf(shape, 0.0f), 1.0f);
        naff += 1;
    }
}

__device__ inline long long other(long long i0, long long i1, long long i2, long long v, int which) {
    // the two vertices of face (i0, i1, i2) other than v, as the fallback picks them
    if (which == 0) return i0 == v ? i1 : i0;
    return i2 == v ? i1 : i2;
}

extern "C" __global__ void collapse_checks(const float* __restrict__ verts, const long long* __restrict__ faces,
                                           const long long* __restrict__ edges, long long E, const float* __restrict__ opt,
                                           const long long* __restrict__ off, const int* __restrict__ adj, float cos_t,
                                           int* __restrict__ flips_out, float* __restrict__ skinny_out,
                                           unsigned char* __restrict__ link_out) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    long long a = edges[2 * e], b = edges[2 * e + 1];
    float3 o = make_float3(opt[3 * e], opt[3 * e + 1], opt[3 * e + 2]);
    int flips = 0, naff = 0; float skin = 0.0f;
    ring(verts, faces, off, adj, a, a, b, o, cos_t, flips, skin, naff);
    ring(verts, faces, off, adj, b, a, b, o, cos_t, flips, skin, naff);
    flips_out[e] = flips;
    skinny_out[e] = skin / (float)(naff > 0 ? naff : 1);

    // Link condition: distinct common neighbours of a and b (not a or b) <= faces of a that hold b. It is O(deg^2) per
    // thread, so a hub vertex is refused outright (no collapse): bounded work, never a GPU watchdog reset.
    if (off[a + 1] - off[a] > LINK_MAX_DEG || off[b + 1] - off[b] > LINK_MAX_DEG) { link_out[e] = 0; return; }
    int count_faces = 0, common = 0;
    for (long long j = off[a]; j < off[a + 1]; ++j) {
        long long f = adj[j];
        long long i0 = faces[3 * f], i1 = faces[3 * f + 1], i2 = faces[3 * f + 2];
        if (i0 == b || i1 == b || i2 == b) count_faces += 1;
        for (int w = 0; w < 2; ++w) {
            long long n = other(i0, i1, i2, a, w);
            if (n == a || n == b) continue;
            // already counted from an earlier face or slot of a?
            bool seen = false;
            for (long long j2 = off[a]; j2 <= j && !seen; ++j2) {
                long long g = adj[j2];
                long long k0 = faces[3 * g], k1 = faces[3 * g + 1], k2 = faces[3 * g + 2];
                for (int w2 = 0; w2 < 2; ++w2) {
                    if (j2 == j && w2 >= w) break;
                    long long m = other(k0, k1, k2, a, w2);
                    if (m == n) { seen = true; break; }
                }
            }
            if (seen) continue;
            bool in_b = false;  // a neighbour of b?
            for (long long jb = off[b]; jb < off[b + 1] && !in_b; ++jb) {
                long long g = adj[jb];
                long long k0 = faces[3 * g], k1 = faces[3 * g + 1], k2 = faces[3 * g + 2];
                long long m0 = other(k0, k1, k2, b, 0), m1 = other(k0, k1, k2, b, 1);
                if ((m0 == n && m0 != a && m0 != b) || (m1 == n && m1 != a && m1 != b)) in_b = true;
            }
            if (in_b) common += 1;
        }
    }
    link_out[e] = common <= count_faces ? 1 : 0;
}
"""

_k = {}


def _kernels(device):
    i = device.index if device.index is not None else torch.cuda.current_device()
    if i not in _k:
        with torch.cuda.device(i):
            _k[i] = {n: Kernel(SRC, n) for n in ("vf_count", "vf_fill", "collapse_checks")}
    return _k[i]


def incident_faces(faces, num_verts):
    """(off [V+1] int64, adj [3F] int32): faces of vertex v are adj[off[v]:off[v+1]], in no particular order."""
    k = _kernels(faces.device)
    F = faces.shape[0]
    deg = torch.zeros(num_verts, dtype=torch.int32, device=faces.device)
    off = torch.zeros(num_verts + 1, dtype=torch.int64, device=faces.device)
    adj = torch.empty(3 * F, dtype=torch.int32, device=faces.device)
    if F == 0:
        return off, adj
    grid = ((3 * F + 255) // 256,)
    k["vf_count"](grid, (256,), faces, I64(F), deg)
    off[1:] = torch.cumsum(deg, 0)
    cursor = torch.zeros(num_verts, dtype=torch.int32, device=faces.device)
    k["vf_fill"](grid, (256,), faces, I64(F), off, cursor, adj)
    return off, adj


def edge_collapse_checks(verts, faces, edges, opt, cos_threshold=0.0):
    """Drop-in for comfy-kitchen's op as `qem_decimate._gated_errors` calls it: (flips int32, skinny float32, link_ok bool)."""
    verts = verts.float().contiguous(); faces = faces.long().contiguous()
    edges = edges.long().contiguous(); opt = opt.float().contiguous()
    E, V = edges.shape[0], verts.shape[0]
    # The kernels index with these values unchecked (an out-of-range face index is an out-of-bounds atomic write), so
    # check shapes, devices and index ranges here.
    if verts.shape[1:] != (3,) or faces.shape[1:] != (3,) or edges.shape[1:] != (2,) or tuple(opt.shape) != (E, 3):
        raise ValueError(f"edge_collapse_checks: shapes verts {tuple(verts.shape)} faces {tuple(faces.shape)} "
                         f"edges {tuple(edges.shape)} opt {tuple(opt.shape)}")
    if len({t.device for t in (verts, faces, edges, opt)}) != 1 or verts.device != torch.device("cuda", torch.cuda.current_device()):
        raise ValueError("edge_collapse_checks: all tensors must be on the current GPU")
    if faces.shape[0] >= 2 ** 31:  # face ids are int32 in the incident-face lists
        raise ValueError(f"edge_collapse_checks: {faces.shape[0]:,} faces, over the 2^31 the kernels index")
    present = [t for t in (faces, edges) if t.numel()]
    if present:  # one host sync for every range check
        lo_hi = torch.stack([x for t in present for x in (t.min(), t.max())]).tolist()
        if min(lo_hi[0::2]) < 0 or max(lo_hi[1::2]) >= V:
            raise ValueError(f"edge_collapse_checks: faces or edges index out of range [0, {V})")
    flips = torch.zeros(E, dtype=torch.int32, device=verts.device)
    skinny = torch.zeros(E, dtype=torch.float32, device=verts.device)
    link = torch.ones(E, dtype=torch.uint8, device=verts.device)
    if E == 0:
        return flips, skinny, link.bool()
    off, adj = incident_faces(faces, verts.shape[0])
    _kernels(verts.device)["collapse_checks"](((E + 255) // 256,), (256,), verts, faces, edges, I64(E), opt, off, adj,
                                              float(cos_threshold), flips, skinny, link)
    return flips, skinny.to(verts.dtype), link.bool()


def self_test(device="cuda"):
    """Compare with ComfyUI's torch fallback (degree cap raised so both are exact) on a closed grid mesh with random
    collapse targets. Returns (flip mismatches, max |skinny diff|, link mismatches)."""
    import comfy_extras.mesh3d.postprocess.qem_decimate as q
    g = torch.Generator().manual_seed(0)
    n = 24  # a torus grid: closed, every vertex degree 6
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    u, v = i.flatten().float() * 6.2831853 / n, j.flatten().float() * 6.2831853 / n
    verts = torch.stack([(2 + torch.cos(v)) * torch.cos(u), (2 + torch.cos(v)) * torch.sin(u), torch.sin(v)], 1)
    idx = lambda a, b: (a % n) * n + (b % n)
    quads = [(idx(a, b), idx(a + 1, b), idx(a + 1, b + 1), idx(a, b + 1)) for a in range(n) for b in range(n)]
    faces = [[p, q2, r] for p, q2, r, s in quads] + [[p, r, s] for p, q2, r, s in quads]
    # a second component with a link violation: a, b share neighbour c with no face abc (common {c, p, q} > 2 faces)
    a_, b_, c_, x_, y_, p_, q_ = range(n * n, n * n + 7)
    faces = torch.tensor(faces + [[a_, c_, x_], [b_, y_, c_], [a_, b_, p_], [b_, a_, q_]])
    verts = torch.cat([verts, torch.tensor([[5., 0, 0], [5., 1, 0], [6., .5, .3], [5.5, -.5, .8], [4.5, 1.5, .8],
                                            [5.5, .5, -1], [4.5, .5, 1]])])
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = torch.unique(torch.sort(e, 1).values, dim=0)
    opt = (verts[edges[:, 0]] + verts[edges[:, 1]]) / 2 + 0.3 * torch.randn(edges.shape[0], 3, generator=g)
    verts, faces, edges, opt = (t.to(device) for t in (verts, faces, edges, opt))
    f1, s1, l1 = edge_collapse_checks(verts, faces, edges, opt, 0.0)
    v2f = q._build_vert_to_faces_pad(faces, verts.shape[0], 64)
    f2, s2, l2 = q._quality_checks_fused(verts, faces, edges, opt, v2f, cos_threshold=0.0, want_link=True)
    if int((~l2).sum()) == 0:  # the gate must see a link violation, or a kernel answering "ok" everywhere would pass
        raise RuntimeError("self-test mesh has no link violation")
    return int((f1 != f2).sum()), float((s1 - s2).abs().max()), int((l1 != l2).sum())


def install():
    """Route `edge_collapse_checks` to this kernel for GPU tensors; everything else asks comfy-kitchen as before.
    Returns the self-test result; raises (and installs nothing) if it does not match the torch fallback."""
    import comfy_extras.mesh3d.postprocess.qem_decimate as q
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("not a ROCm build: comfy-kitchen's CUDA kernel applies")
    result = self_test()
    if result[0] or result[2] or not (result[1] <= 1e-4):  # NaN fails too
        raise RuntimeError(f"self-test mismatch against the torch fallback: {result}")
    orig = q._kitchen_op

    def kitchen_op(name, tensor):
        if name == "edge_collapse_checks" and tensor.is_cuda and tensor.device == torch.device("cuda", torch.cuda.current_device()):
            return edge_collapse_checks  # another GPU keeps upstream's path (the kernels are built for the current one)
        return orig(name, tensor)
    q._kitchen_op = kitchen_op
    return result
