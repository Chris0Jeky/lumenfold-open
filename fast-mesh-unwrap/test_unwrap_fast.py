"""lf_unwrap_fast against ComfyUI's own batched LSCM. Needs the ComfyUI source and a torch build (ComfyUI's embedded
Python) with COMFYUI_ROOT set to the ComfyUI root; skips elsewhere. CPU only: no GPU or model.

  <ComfyUI root>/../python_embeded/python.exe -s test_unwrap_fast.py
"""
import os, sys, unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np

NODE = Path(__file__).resolve().parent / "lf_unwrap_fast"
os.environ["NUMBA_CACHE_DIR"] = str(NODE / "_bench_results/numba_cache")

sys.path.insert(0, str(NODE))
import unwrap_fast as U  # noqa: E402  (numpy and scipy only at import: a broken module fails the run, never skips)

try:  # only a missing torch or ComfyUI skips
    import torch  # noqa: F401
    import bench_unwrap as bench
    N, DEFAULTS = bench.load_upstream()
    P, M, S = N._uv_param, N._uv_mesh, N._uv_seg
    import batch_fast as B
    torch.set_num_threads(2)
    SKIP = None
except ImportError as e:
    SKIP = f"needs ComfyUI's source and torch: {e}"


@unittest.skipIf(SKIP, SKIP or "")
class SparseLSCM(unittest.TestCase):
    def upstream(self):
        return getattr(P.lscm_charts_batch, "__wrapped__", P.lscm_charts_batch)

    def test_matches_upstream(self):
        for seed, group in ((0, U.GROUP_CHARTS), (1, 1), (2, 2)):  # one solve, and the chart-group split
            args = U.demo_charts(seed)
            saved, U.GROUP_CHARTS = U.GROUP_CHARTS, group
            try:
                ref, new = self.upstream()(*args), U.lscm_charts_sparse(*args)
            finally:
                U.GROUP_CHARTS = saved
            self.assertEqual(set(ref), set(new))
            for c, u in ref.items():
                self.assertLess(float(np.abs(new[c] - u).max() / np.ptp(u, axis=0).max()), 1e-5, (seed, c))

    def test_large_charts_left_to_caller(self):
        args = list(U.demo_charts()); args.append(100)  # max_bucket_verts: the 144-vertex chart is too big
        ref, new = self.upstream()(*args), U.lscm_charts_sparse(*args)
        self.assertEqual(set(ref), {0, 1}); self.assertEqual(set(new), {0, 1})

    def test_empty(self):
        args = list(U.demo_charts()); args[7] = np.zeros(0, np.int64)
        self.assertEqual(U.lscm_charts_sparse(*args), {})


@unittest.skipIf(SKIP, SKIP or "")
class BatchExtension(unittest.TestCase):
    def test_boundary_loops(self):
        # Open chart, empty chart, closed torus, and conflicting directed edges.
        fixtures = [(torch.empty((0, 3), dtype=torch.long), torch.empty((0, 3), dtype=torch.long))]
        mesh = U.demo_mesh(8)
        fixtures.append((mesh.faces, mesh.face_face))
        fixtures.append((torch.tensor([[0, 1, 2], [0, 3, 4], [1, 2, 5]]), torch.full((3, 3), -1)))
        for f, ff in fixtures:
            self.assertEqual(B.boundary_loops(f, ff), M.chart_boundary_loops(f, ff))

    def test_cone_half_exact(self):
        generator = torch.Generator().manual_seed(3)
        a = torch.randn((500, 3), generator=generator)
        b = torch.randn((500, 3), generator=generator)
        a /= a.norm(dim=1, keepdim=True)
        b /= b.norm(dim=1, keepdim=True)
        a[:3] = torch.tensor([[1., 0., 0.]]).expand(3, 3)
        b[:3] = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        ha, hb = torch.rand(500, generator=generator), torch.rand(500, generator=generator)
        self.assertTrue(torch.equal(B.cone_half(a, ha, b, hb), S._combine_normal_cones(a, ha, b, hb)[1]))

    def test_remap_edges(self):
        rng = np.random.default_rng(4)
        for size in (0, 1, 10, 1000):
            edges = torch.from_numpy(rng.integers(0, 50, (size, 2)))
            mapping = torch.from_numpy(rng.integers(0, 20, 50))
            self.assertTrue(torch.equal(B.remap_edges(edges, mapping, 20), U._remap_edges(edges, mapping, 20)))
            self.assertTrue(torch.equal(B.remap_edges_tensor(edges, mapping, 20), U._remap_edges(edges, mapping, 20)))
        self.assertEqual(B.remap_edges(torch.tensor([[0, 1]]), torch.zeros(2, dtype=torch.long), 1).shape, (0, 2))

    def test_pec_chart_assignment(self):
        for size, iters, min_faces in ((8, 1024, 8), (12, 7, 1), (16, 1024, 8)):
            mesh = U.demo_mesh(size)
            for cost in (.7, 1.):
                ref = S.cluster_charts_pec(mesh, cost, iters, min_faces)
                new = B.cluster_charts(mesh, cost, iters, min_faces)
                self.assertTrue(torch.equal(ref, new), (size, cost, iters, min_faces))

    def chart_fixture(self):
        args = U.demo_charts(2, sizes=(4, 17, 21))
        v, uv, gl, pos, cof, cov, voff, ids, n = args
        foff = np.r_[0, np.cumsum(np.bincount(cof, minlength=n))]
        local = gl - voff[cof, None]
        ff = np.full_like(local, -1)
        for c in ids:
            f0, f1 = foff[c:c+2]
            # Actual adjacency of each submesh; includes an interior vertex so
            # boundary-extremal pins differ from small-batch farthest-point pins.
            vv = torch.from_numpy(v[voff[c]:voff[c+1]].astype(np.float32))
            ff[f0:f1] = M.build_mesh(vv, torch.from_numpy(local[f0:f1])).face_face.numpy()
        return args, local, ff, foff

    def test_guard_falls_back(self):
        def fail(*a, **kw):
            raise ValueError("injected failure")
        wrapped = B.guarded(fail, lambda x, y=0: x + y)
        self.assertEqual(wrapped(4, y=5), 9)

    def test_install_checked_atomic_and_idempotent(self):
        from unittest.mock import patch
        orig = S.cluster_charts_pec, S._merge_small_charts, M.chart_boundary_loops
        with patch.object(B, "cone_half", side_effect=RuntimeError("injected mismatch")):
            with self.assertRaises(RuntimeError):
                B.install()
        self.assertEqual((S.cluster_charts_pec, S._merge_small_charts, M.chart_boundary_loops), orig)  # nothing published
        try:
            self.assertEqual(B.install(), 2)
            self.assertTrue(getattr(S.cluster_charts_pec, "_lf_batch_fast", False))
            self.assertTrue(getattr(S.cluster_charts_pec, "_lf_unwrap_fast", False))  # unwrap_fast won't downgrade it
            self.assertEqual(B.install(), 0)
        finally:
            S.cluster_charts_pec, S._merge_small_charts, M.chart_boundary_loops = orig

    def test_install_order_keeps_upstream_reachable(self):
        """install_pec then batch_fast.install, as __init__ runs them; a forced fast failure falls back to upstream's real
        clustering (upstream merge included) on a real mesh."""
        up_cluster, up_merge, up_loops = S.cluster_charts_pec, S._merge_small_charts, M.chart_boundary_loops
        mesh = U.demo_mesh(12)
        ref = up_cluster(mesh, max_cost=1.)
        try:
            U.install_pec()
            self.assertEqual(B.install(), 2)
            self.assertIs(S.cluster_charts_pec.__wrapped__, up_cluster)
            self.assertIs(S._merge_small_charts.__wrapped__, up_merge)
            failing = U.pec_guard(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected")),
                                  S.cluster_charts_pec.__wrapped__, S._merge_small_charts.__wrapped__, S)
            self.assertTrue(torch.equal(failing(mesh, max_cost=1.), ref))
            self.assertIsNot(S._merge_small_charts, up_merge)  # the fast merge is back after the fallback
            self.assertEqual(U.install_pec(), True)  # no downgrade: batch_fast's entry stays published
            self.assertTrue(getattr(S.cluster_charts_pec, "_lf_batch_fast", False))
        finally:
            S.cluster_charts_pec, S._merge_small_charts, M.chart_boundary_loops = up_cluster, up_merge, up_loops

    def test_pec_fallback_runs_upstreams_whole_path(self):
        seen = []
        def up_merge(*a, **kw): return "up-merge"
        def fast_merge(*a, **kw): return "fast-merge"
        def up_cluster(*a, **kw):
            seen.append(S._merge_small_charts)  # what upstream's clustering would look up on the module
            return "upstream"
        def failing(*a, **kw): raise RuntimeError("GPU OOM")
        saved = S._merge_small_charts
        S._merge_small_charts = fast_merge
        try:
            pec = U.pec_guard(failing, up_cluster, up_merge, S)
            self.assertEqual(pec("mesh"), "upstream")
            self.assertIs(seen[0], up_merge)               # the retry used upstream's merge
            self.assertIs(S._merge_small_charts, fast_merge)  # and put the fast one back
            self.assertIs(pec.__wrapped__, up_cluster)
            self.assertEqual(U.pec_guard(lambda *a, **kw: "fast", up_cluster, up_merge, S)("m"), "fast")
        finally:
            S._merge_small_charts = saved


if __name__ == "__main__":
    unittest.main()
