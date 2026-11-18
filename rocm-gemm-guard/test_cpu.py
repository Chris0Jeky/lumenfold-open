"""The guard's chunking logic on the CPU: needs torch (any build, CPU is enough), no GPU.

    python test_cpu.py

The limits are lowered to a few rows so small CPU tensors exercise every chunked path. On a machine without a ROCm GPU
importing the guard installs nothing, so these tests call its helpers directly.
"""
import importlib.util, unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

SRC = Path(__file__).resolve().parent / "hip_gemm_chunk" / "__init__.py"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


@unittest.skipIf(torch is None, "torch not installed")
class Rows(unittest.TestCase):
    def test_row_chunking_matches_one_call(self):
        m = load("hgc_rows"); m.ROWS = 7
        x, w, b = torch.randn(3, 10, 4), torch.randn(5, 4), torch.randn(5)
        lin = torch.nn.functional.linear
        torch.testing.assert_close(m._rows_chunked(lin, x, w, b), lin(x, w, b))

    def test_lowp_rows_pass_through_fp32_chunks(self):
        m = load("hgc_lowp"); m.ROWS = 7
        calls = []
        def lin(x, *rest): calls.append(x.shape[0]); return torch.nn.functional.linear(x, *rest)
        w = torch.randn(5, 4)
        for dt, expect in ((torch.bfloat16, [30]), (torch.float16, [30]), (torch.float32, [7, 7, 7, 7, 2])):
            calls.clear()
            m._rows_chunked(lin, torch.randn(30, 4).to(dt), w.to(dt))
            self.assertEqual(calls, expect, dt)
        calls.clear()  # a non-contiguous 3-D bf16 input is flattened to 2-D rows (never an internal big-batch bmm)
        x3 = torch.randn(4, 30, 3).bfloat16().transpose(0, 1)  # [30, 4, 3], non-contiguous
        y = m._rows_chunked(lin, x3, torch.randn(5, 3).bfloat16())
        self.assertEqual((calls, tuple(y.shape)), ([120], (30, 4, 5)))
        dims = []  # an N-D input under ROWS is still flattened to 2-D (never an internal bmm), in every dtype
        def lin2(x, *rest): dims.append(x.dim()); return torch.nn.functional.linear(x, *rest)
        m.ROWS = 1000
        for dt in (torch.float32, torch.bfloat16):
            x3 = torch.randn(2, 30, 3).to(dt).transpose(0, 1)  # [30, 2, 3], non-contiguous, 60 rows < ROWS
            w3 = torch.randn(5, 3).to(dt)
            y = m._rows_chunked(lin2, x3, w3)
            self.assertEqual(tuple(y.shape), (30, 2, 5))
            torch.testing.assert_close(y, torch.nn.functional.linear(x3, w3))
        self.assertEqual(dims, [2, 2])
        m.ROWS = 7
        m.LOWP_MAX_ROWS = 20  # beyond the measured range, low precision is chunked too
        calls.clear(); m._rows_chunked(lin, torch.randn(30, 4).bfloat16(), w.bfloat16())
        self.assertEqual(calls, [7, 7, 7, 7, 2])


@unittest.skipIf(torch is None, "torch not installed")
class Batches(unittest.TestCase):
    def test_batch_chunks_stay_under_the_limit_and_match(self):
        m = load("hgc_batch"); m.BATCH = 8
        sizes = []
        def bmm(a, b): sizes.append(a.shape[0]); return torch.bmm(a, b)
        a, b = torch.randn(20, 2, 3), torch.randn(20, 3, 2)
        y = m._batch_chunked(bmm, (a, b), (20,), [(0,), (0,)], (0,), {})
        torch.testing.assert_close(y, torch.bmm(a, b))
        self.assertTrue(sizes and all(s < 8 for s in sizes), sizes)

    def test_einsum_plan(self):
        m = load("hgc_einsum")
        a, b, v = torch.randn(20, 2, 2), torch.randn(20, 2, 2), torch.randn(20, 2)
        self.assertEqual(m._einsum_batch("bij,bjk->bik", (a, b)), ((20,), [(0,), (0,)], (0,)))
        self.assertEqual(m._einsum_batch("bi,bij->bj", (v, a)), ((20,), [(0,), (0,)], (0,)))
        self.assertIsNone(m._einsum_batch("bii,bij->bj", (a, b)))   # repeated label: native einsum
        self.assertIsNone(m._einsum_batch("bij,bjk", (a, b)))       # implicit output: native einsum


if __name__ == "__main__":
    unittest.main()
