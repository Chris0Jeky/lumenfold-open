"""Benchmark the original and every solution side by side.

    py -3 bench_all.py [ROUNDS] [--cxx cl|g++|clang++]        ->  bench_all.json

Each build is made once with kbuild.py (its tests and both exactness checks must pass). Then ROUNDS rounds, in which
every build's bench_formation runs once in turn (interleaved, so drift hits all builds alike). Each process prints the
median of its 21 samples per benchmark; the table takes the median over rounds. The total is the slots median plus the
assign median, and every speedup is the original's time over the build's. Run it on a quiet machine."""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

K = Path(__file__).resolve().parent
EXE = ".exe" if os.name == "nt" else ""
TREES = {"original": K / "original", **{a: K / "solutions" / a for a in ("V", "N", "L", "A")}}
NAMES = ("formation.slots.dragon.n100000", "formation.assign.n100000")

ap = argparse.ArgumentParser(); ap.add_argument("rounds", type=int, nargs="?", default=9)
ap.add_argument("--cxx", default="cl" if os.name == "nt" else "g++")
a = ap.parse_args()
res = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "rounds": a.rounds, "cxx": a.cxx, "runs": {t: {n: [] for n in NAMES} for t in TREES}}
for t, tree in TREES.items():
    r = subprocess.run([sys.executable, str(K / "kbuild.py"), str(tree), "--cxx", a.cxx], capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:]); sys.exit(f"{t}: build or check failed")
    print(t, "built and checked", flush=True)
for k in range(a.rounds):
    for t, tree in TREES.items():
        out = subprocess.run([str(tree / "build" / ("bench_formation" + EXE))], capture_output=True, text=True)
        if out.returncode: sys.exit(f"{t}: bench failed: {out.stderr[-500:]}")
        for line in out.stdout.splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                if row.get("bench") in NAMES:
                    res["runs"][t][row["bench"]].append({"median_ns": row["median_ns"], "checksum": row.get("checksum")})
    print("round", k + 1, "done", flush=True)
med = {t: {n: statistics.median(x["median_ns"] for x in res["runs"][t][n]) / 1e6 for n in NAMES} for t in TREES}
base = sum(med["original"].values())
res["table"] = {}
for t in TREES:
    row = {n: {"median_ms": round(med[t][n], 4), "speedup": round(med["original"][n] / med[t][n], 2),
               "checksums": sorted({x["checksum"] for x in res["runs"][t][n]})} for n in NAMES}
    total = sum(med[t].values())
    row["total"] = {"ms": round(total, 4), "speedup": round(base / total, 2)}
    res["table"][t] = row
(K / "bench_all.json").write_text(json.dumps(res, indent=1))
for t, row in res["table"].items():
    print(f"{t:9s} " + "  ".join(f"{n.split('.')[1]} {row[n]['median_ms']:9.4f} ms x{row[n]['speedup']:7.2f} {row[n]['checksums']}" for n in NAMES)
          + f"  total {row['total']['ms']:8.4f} ms x{row['total']['speedup']:6.2f}")
