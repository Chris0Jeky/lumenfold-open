"""CPU-only, offline comparison of upstream, the original patch, and the extended patch.

Run with ComfyUI's embedded python.exe -s, COMFYUI_ROOT set to the ComfyUI root. No ComfyUI server or model-management
module is imported. The node's two unwrap functions are compiled unchanged from
its installed source; only progress reporting and pack's device provider are stubbed.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench_results", "numba_cache")

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
import logging

import numpy as np
import torch

COMFY = Path(os.environ.get("COMFYUI_ROOT", ""))  # ComfyUI v0.43.2 source tree; its uv_unwrap is read from here
HERE = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / "_bench_results/numba_cache"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True


class Progress:
    def __init__(self, *a, **kw):
        self.n = 0

    def update_absolute(self, *a):
        pass

    def update(self, n):
        self.n += n

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def quiet_tqdm(it=None, **kw):
    return it if it is not None else Progress()


def load_upstream():
    """Load only uv_unwrap, never comfy's GPU-discovering imports."""
    root = COMFY / "comfy_extras/mesh3d/uv_unwrap"
    if not (root / "mesh.py").exists():
        raise ImportError(f"ComfyUI uv_unwrap source not found at {root}")
    # Torch can import Numba before this loader. Keep its cached configuration
    # from probing the read-only upstream source tree for writable cache paths.
    config = sys.modules.get("numba.core.config")
    if config is not None:
        config.CACHE_DIR = os.environ["NUMBA_CACHE_DIR"]
    for name, path in (("comfy_extras", root.parent.parent),
                       ("comfy_extras.mesh3d", root.parent),
                       ("comfy_extras.mesh3d.uv_unwrap", root)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod
    prefix = "comfy_extras.mesh3d.uv_unwrap."
    for leaf in ("mesh", "segment", "parameterize", "pack"):
        name = prefix + leaf
        spec = importlib.util.spec_from_file_location(name, root / (leaf + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        if leaf == "pack":
            tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
            tree.body = [n for n in tree.body if not (
                isinstance(n, ast.Import) and any(a.name == "comfy.model_management" for a in n.names))]
            mod.comfy = types.SimpleNamespace(model_management=types.SimpleNamespace(
                get_torch_device=lambda: torch.device("cpu"), get_free_memory=lambda device: 2 << 30))
            exec(compile(tree, spec.origin, "exec"), mod.__dict__)
        else:
            spec.loader.exec_module(mod)
        setattr(sys.modules[prefix[:-1]], leaf, mod)
    source = COMFY / "comfy_extras/nodes_mesh_postprocess.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    schema = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnwrapMesh")
    defaults = {}
    for n in ast.walk(schema):
        if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant):
            for k in n.keywords:
                if k.arg == "default":
                    defaults[n.args[0].value] = ast.literal_eval(k.value)
    node = types.ModuleType("lf_cpu_unwrap_node")
    node.__file__ = str(source)
    node.__dict__.update(torch=torch, np=np, math=math, time=time, logging=logging,
                         tqdm=quiet_tqdm, comfy=types.SimpleNamespace(utils=types.SimpleNamespace(ProgressBar=Progress)))
    for alias, leaf in (("_uv_mesh", "mesh"), ("_uv_seg", "segment"),
                        ("_uv_param", "parameterize"), ("_uv_pack", "pack")):
        node.__dict__[alias] = sys.modules[prefix + leaf]
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("_uv_weld_vertices", "_uv_unwrap")]
    exec(compile(tree, str(source), "exec"), node.__dict__)
    sys.modules[node.__name__] = node
    return node, defaults


def run(node, vertices, faces, defaults):
    s, p, pk = node._uv_seg, node._uv_param, node._uv_pack
    times, calls, captures = {}, {}, {}
    restore = []

    def wrap(mod, name, stage):
        orig = getattr(mod, name)
        restore.append((mod, name, orig))

        def measured(*a, **kw):
            if stage == "pack":
                captures["param_end"] = time.perf_counter()
            start = time.perf_counter()
            result = orig(*a, **kw)
            times[stage] = times.get(stage, 0.0) + time.perf_counter() - start
            calls[stage] = calls.get(stage, 0) + 1
            if stage == "segment":
                captures["face_chart"] = result.cpu().numpy().copy()
            elif stage == "pack":
                captures.update(uv=a[0].copy(), offsets=a[1].copy(), local_faces=a[2].copy(),
                                face_offsets=a[3].copy(), areas=a[5].copy(), layout=result)
            elif stage == "lscm_single":
                captures.setdefault("single_sizes", []).append(len(a[0]))
            return result
        setattr(mod, name, measured)

    class TimedProgress(Progress):
        def update_absolute(self, n, *a):
            if n == 400 and "param_start" not in captures:
                captures["param_start"] = time.perf_counter()

    progress = node.comfy.utils.ProgressBar
    node.comfy.utils.ProgressBar = TimedProgress
    for mod, name, stage in ((s, "cluster_charts_pec", "segment"),
                             (p, "lscm_charts_batch", "lscm_batch"), (p, "lscm_chart", "lscm_single"),
                             (p.spla, "spsolve", "sparse_solve"),
                             (pk, "pack_bitmap_concat", "pack")):
        wrap(mod, name, stage)
    extent = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    start = time.perf_counter()
    try:
        output = node._uv_unwrap(torch.from_numpy(vertices), torch.from_numpy(faces),
                                 defaults["segmenter"], defaults["resolution"], defaults["padding"],
                                 defaults["weld_distance"] * extent)
    finally:
        for mod, name, orig in reversed(restore):
            setattr(mod, name, orig)
        node.comfy.utils.ProgressBar = progress
    times["total"] = time.perf_counter() - start
    times["parameterize"] = captures["param_end"] - captures["param_start"]
    captures["output"] = output
    sizes = captures.get("single_sizes", [])
    return dict(seconds=times, calls=calls,
                single_sizes=dict(count=len(sizes), min=min(sizes, default=0), max=max(sizes, default=0))), captures


def compare(a, b):
    same_charts = np.array_equal(a["face_chart"], b["face_chart"])
    same_topology = all(np.array_equal(x, y) for x, y in zip(a["output"][:2], b["output"][:2]))
    if not same_charts or not same_topology:
        return dict(same_chart_assignment=same_charts, same_topology=same_topology, ok=False)
    errors = []
    for lo, hi in zip(a["offsets"][:-1], a["offsets"][1:]):
        span = max(float(np.ptp(a["uv"][lo:hi], axis=0).max()), 1e-20)
        errors.append(float(np.abs(a["uv"][lo:hi] - b["uv"][lo:hi]).max()) / span)
    # Float32 output: 1e-5 of chart span allows ~84 ulps, including
    # different sparse/dense elimination orders; pins and equations must match.
    area_rel = float(abs(b["areas"].sum() / a["areas"].sum() - 1))
    area_a = max(a["layout"][-2:]) ** 2
    area_b = max(b["layout"][-2:]) ** 2
    from comfy_extras.mesh3d.uv_unwrap import parameterize as p
    def stretch(c):
        vmap, indices, uv = c["output"]
        # Unpacked UVs remove packing scale/rotation from the comparison.
        gl = c["local_faces"] + np.repeat(c["offsets"][:-1], np.diff(c["face_offsets"]))[:, None]
        verts = compare.vertices[vmap].astype(np.float64)
        cof = np.repeat(np.arange(len(c["offsets"]) - 1), np.diff(c["face_offsets"]))
        rms, mx, flips, zero = p.stretch_metrics_concat(verts, c["uv"], gl, cof, len(c["offsets"]) - 1)
        return rms, mx, flips, zero
    sa, sb = stretch(a), stretch(b)
    finite = np.isfinite(sa[0]) & np.isfinite(sb[0])
    return dict(same_chart_assignment=True, same_topology=True, charts=len(errors),
                max_chart_uv_relative=max(errors), p99_chart_uv_relative=float(np.quantile(errors, .99)),
                same_layout=all(np.array_equal(x, y) for x, y in zip(a["layout"], b["layout"])),
                same_integer_layout=all(np.array_equal(a["layout"][i], b["layout"][i]) for i in (0, 1, 2, 5, 6, 7)),
                max_layout_rotation_delta=float(np.max(np.abs(a["layout"][3] - b["layout"][3]), initial=0.)),
                max_layout_scale_relative=float(np.max(np.abs(a["layout"][4] - b["layout"][4])
                                                       / np.maximum(np.abs(a["layout"][4]), 1e-20), initial=0.)),
                atlas_a=list(map(int, a["layout"][-2:])), atlas_b=list(map(int, b["layout"][-2:])),
                atlas_area_relative=area_b / area_a - 1, uv_area_relative=area_rel,
                max_stretch_rms_delta=float(np.max(np.abs(sa[0][finite] - sb[0][finite]), initial=0.)),
                same_finite_stretch=np.array_equal(np.isfinite(sa[0]), np.isfinite(sb[0])),
                same_flip_zero_counts=all(np.array_equal(x, y) for x, y in zip(sa[2:], sb[2:])),
                max_packed_uv_delta=float(np.abs(a["output"][2] - b["output"][2]).max()),
                ok=max(errors) < 1e-5 and area_rel < 1e-5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh", help="any mesh trimesh can load (.glb, .obj, .ply)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--modes", nargs="+", choices=("upstream", "current", "candidate"),
                    default=("upstream", "current", "candidate"))
    ap.add_argument("--profile", action="store_true", help="print CPU profiles (diagnostic timings)")
    args = ap.parse_args()
    canonical = ("upstream", "current", "candidate")
    if list(args.modes) != [m for m in canonical if m in args.modes]:
        ap.error("--modes must be a unique subsequence of upstream current candidate")
    os.environ["NUMBA_CACHE_DIR"] = str(HERE / "_bench_results/numba_cache")
    import trimesh
    node, defaults = load_upstream()
    torch.set_num_threads(args.threads)
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    # UVs/materials are deliberately unused. Node auto-welding joins seam vertices.
    vertices = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.int64)
    compare.vertices = vertices
    # Warm pack JIT before any timing; all three modes use the same backend.
    pk = node._uv_pack
    pk.pack_bitmap_concat(np.array([[0., 0.], [1., 0.], [0., 1.]], np.float32),
                          np.array([0, 3]), np.array([[0, 1, 2]]), np.array([0, 1]),
                          np.array([.5]), np.array([.5]), texels_per_unit=8, attempts=8)
    report = dict(mesh=args.mesh, faces=len(faces), vertices=len(vertices), defaults=defaults,
                  threads=args.threads, numba=pk._HAVE_NUMBA_PACK,
                  reference_mode=args.modes[0], runs={}, equivalence={})
    upstream = None
    for mode in args.modes:
        if mode == "current":
            import unwrap_fast
            unwrap_fast.install()
        elif mode == "candidate":
            if args.baseline_only:
                break
            import unwrap_fast
            unwrap_fast.install()
            import batch_fast
            batch_fast.install()
        print(f"Running {mode} ({len(faces)} faces)...", flush=True)
        if args.profile:
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            row, capture = profiler.runcall(run, node, vertices, faces, defaults)
            pstats.Stats(profiler).sort_stats("tottime").print_stats(30)
        else:
            row, capture = run(node, vertices, faces, defaults)
        report["runs"][mode] = row
        print(json.dumps({mode: row}), flush=True)
        if upstream is None:
            upstream = capture
        else:
            report["equivalence"][mode] = compare(upstream, capture)
            print(json.dumps({"equivalence": report["equivalence"][mode]}), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r["ok"] for r in report["equivalence"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
