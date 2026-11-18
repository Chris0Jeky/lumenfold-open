"""lf_qem_hip's low-peak QEM half-edge table on the CPU, against upstream's `_sorted_edge_halfedges` read from your
ComfyUI install (comfy_extras/mesh3d/postprocess/qem_decimate.py), and the source pin. No GPU; needs torch (CPU is
enough). ComfyUI itself is not imported: only that one function is compiled from its source.

  set COMFYUI_ROOT=<ComfyUI root>
  python test_edges_lowpeak.py
"""
import ast, hashlib, os, sys, unittest
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "lf_qem_hip"))
import edges_lowpeak as E  # noqa: E402

ROOT = os.environ.get("COMFYUI_ROOT")
SRC = Path(ROOT, "comfy_extras", "mesh3d", "postprocess", "qem_decimate.py") if ROOT else None


def load_upstream():
    """(function, its source segment) for upstream's _sorted_edge_halfedges, compiled alone from the installed file."""
    src = SRC.read_text(encoding="utf8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_sorted_edge_halfedges")
    seg = ast.get_source_segment(src, node)
    ns = {"torch": torch, "Tuple": Tuple}
    exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
    return ns["_sorted_edge_halfedges"], seg


@unittest.skipUnless(SRC and SRC.exists(), "set COMFYUI_ROOT to a ComfyUI v0.43.2 checkout")
class EdgesLowPeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up, cls.seg = load_upstream()

    def test_pin_matches_the_installed_comfyui(self):
        self.assertEqual(hashlib.sha256(self.seg.encode()).hexdigest(), E.UPSTREAM_SHA256)

    def test_selftest_passes_against_upstream_and_fails_a_wrong_key(self):
        self.assertTrue(E.selftest(self.up))
        self.assertFalse(E.selftest(self.up, lambda f, v: E.sorted_edge_halfedges(f, v + 1)))

    def test_a_larger_mesh_with_shared_edges_is_identical(self):
        g = torch.Generator().manual_seed(1)
        faces = torch.randint(0, 50_000, (120_000, 3), generator=g, dtype=torch.int32)
        faces[1::2, :2] = faces[::2, :2].flip(1)[: faces[1::2].shape[0]]  # every other face shares an edge with its neighbour
        for a, b in zip(self.up(faces, 50_000), E.sorted_edge_halfedges(faces, 50_000)):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.dtype, b.dtype)

    def test_ties_from_repeated_faces_are_identical(self):
        faces = torch.tensor([[0, 1, 2]] * 500 + [[2, 1, 0]] * 500 + [[3, 1, 2]] * 7, dtype=torch.int32)
        for a, b in zip(self.up(faces, 4), E.sorted_edge_halfedges(faces, 4)):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
