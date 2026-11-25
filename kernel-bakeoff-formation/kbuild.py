"""Build, test, check exactness and optionally bench one formation.cpp.

    py -3 kbuild.py <dir> [--bench] [--runs 5] [--samples 21] [--cxx cl|g++|clang++]

<dir> holds a formation.cpp (original/, or solutions/V/ and the like). It may also hold its own formation.h; without one,
original/formation.h is used. Builds into <dir>/build/ and runs test_formation, exact_formation (438 fixed cases) and
exact_wide (1,518 random cases), each against reference/ renamed into namespace ref_formation, bit for bit. With --bench
it also runs bench_formation --runs times and prints the median of each benchmark's per-process medians.
Exit 0 only when every step passes.

The comparison was measured with MSVC (--cxx cl, the default on Windows): /O2 /fp:precise, C++20, x64, no /arch flag.
See README.md for tested toolchains and portability limits."""
import argparse, json, os, shutil, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = ".exe" if os.name == "nt" else ""
MSVC_FLAGS = ["/nologo", "/std:c++20", "/O2", "/fp:precise", "/EHsc", "/W4", "/permissive-", "/Zc:__cplusplus",
              "/Zc:preprocessor", "/utf-8", "/DNDEBUG", "/MP"]
GNU_FLAGS = ["-std=c++20", "-O2", "-DNDEBUG", "-Wall"]  # ISO mode: no FMA contraction, no fast-math
NAMES = ("formation.slots.dragon.n100000", "formation.assign.n100000")


def msvc_env():
    """The environment of vcvars64.bat from the newest Visual Studio with the x64 C++ tools."""
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists(): sys.exit("kbuild: vswhere.exe not found; install Visual Studio or Build Tools with the C++ x64 tools")
    r = subprocess.run([str(vswhere), "-products", "*", "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property", "installationPath"], capture_output=True, text=True)
    bat = Path(r.stdout.strip() or "-") / "VC/Auxiliary/Build/vcvars64.bat"
    r = subprocess.run(f'call "{bat}" >nul && set', shell=True, capture_output=True, text=True)
    env = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l and not l.startswith("="))
    if "VCToolsVersion" not in env: sys.exit("kbuild: vcvars64.bat failed")
    print("kbuild: MSVC tools", env["VCToolsVersion"])
    return env


def reference(build):
    """reference/formation.cpp and .h, renamed into namespace ref_formation, compiled beside the build under test."""
    ref = build / "ref"; ref.mkdir(parents=True, exist_ok=True)
    for name in ("formation.h", "formation.cpp"):
        s = (HERE / "reference" / name).read_text(encoding="utf-8").replace("namespace lumen::formation", "namespace ref_formation")
        (ref / ("ref_" + name)).write_text(s.replace('#include "formation.h"', '#include "ref_formation.h"'), encoding="utf-8")
    return ref / "ref_formation.cpp"


def link(cxx, env, srcs, exe, inc):
    if cxx == "cl":
        obj = exe.parent / "obj" / exe.stem; obj.mkdir(parents=True, exist_ok=True)
        cl = shutil.which("cl", path=env["PATH"])
        cmd = [cl, *MSVC_FLAGS, *[f"/I{p}" for p in inc], *map(str, srcs), f"/Fo{obj}\\", f"/Fe{exe}", "/link", "/INCREMENTAL:NO"]
    else:
        cmd = [cxx, *GNU_FLAGS, *[f"-I{p}" for p in inc], *map(str, srcs), "-o", str(exe)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode: print(r.stdout.strip(), r.stderr.strip()); sys.exit(f"kbuild: {exe.name} failed to build")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir", type=Path); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--samples", type=int)
    ap.add_argument("--cxx", default="cl" if os.name == "nt" else "g++")
    a = ap.parse_args(); d = a.dir.resolve(); build = d / "build"
    if not (d / "formation.cpp").exists(): sys.exit(f"kbuild: no formation.cpp in {d}")
    if build.exists(): shutil.rmtree(build)
    build.mkdir(parents=True)
    env = msvc_env() if a.cxx == "cl" else dict(os.environ)
    ref_cpp = reference(build)
    inc = [d, HERE / "original", HERE / "include", build / "ref"]  # a formation.h in <dir> wins over the original's
    mains = {"test_formation": HERE / "tests/test_formation.cpp", "bench_formation": HERE / "tests/bench_formation.cpp",
             "exact_formation": HERE / "check/exact_formation.cpp", "exact_wide": HERE / "check/exact_wide.cpp"}
    for name, src in mains.items():
        link(a.cxx, env, [d / "formation.cpp", src] + ([ref_cpp] if name.startswith("exact") else []), build / (name + EXE), inc)
    for name in ("test_formation", "exact_formation", "exact_wide"):
        r = subprocess.run([str(build / (name + EXE))], capture_output=True, text=True)
        print(r.stdout.strip()[-3000:])
        if r.returncode: print(r.stderr.strip()[-2000:]); sys.exit(f"kbuild: {name} FAILED (exit {r.returncode})")
    if not a.bench: print("kbuild: tests and exactness pass"); return 0
    got = {nm: ([], set()) for nm in NAMES}
    for _ in range(a.runs):  # one process per run
        cmd = [str(build / ("bench_formation" + EXE))] + (["--samples", str(a.samples)] if a.samples else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-1000:], r.stderr[-1000:]); sys.exit("kbuild: bench failed")
        for line in r.stdout.splitlines():
            if not line.startswith("{"): continue
            row = json.loads(line)
            if row.get("bench") in got: got[row["bench"]][0].append(row["median_ns"]); got[row["bench"]][1].add(str(row.get("checksum")))
    res = {}
    for name, (meds, chk) in got.items():
        if not meds: sys.exit(f"kbuild: no row for {name}")
        res[name] = {"median_ns": statistics.median(meds), "runs": meds, "checksums": sorted(chk)}
        print(f"  {name:34s} median of {len(meds)} runs {statistics.median(meds) / 1e6:9.3f} ms  checksum {sorted(chk)}")
    (build / "bench.json").write_text(json.dumps(res, indent=1))
    print("kbuild: tests, exactness and bench pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
