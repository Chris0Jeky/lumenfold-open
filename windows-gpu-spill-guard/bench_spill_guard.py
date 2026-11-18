"""lf_spill_guard on the GPU (stop ComfyUI first): fill VRAM in steps past the WDDM budget and check that the guard
trips on the process's shared GPU memory and turns the next allocation into a HIP OOM.

  python -s bench_spill_guard.py [--step-mb 256] [--pause-s 0.05] [--threshold-gb 1.0] [--host-floor-gb 6] [--hog-gb 0]

Needs torch (ROCm build) and psutil. The guard is polled every 1 s here (ComfyUI's node polls every 0.25 s).
The 7 Nov real spill ran at about 0.3 GB/s; `--step-mb 128 --pause-s 0.4` fills at that rate, so a 1 s reading
can land between the budget and the cap (at the default 2.7 GB/s the cap OOM comes first).
`--hog-gb N` first starts a helper process that holds N GB of VRAM, as other GPU users did during that spill
(WDDM gave the ComfyUI process 11.5 GB then; on a clean box the process reaches 15.88 GiB with 0.38 GB shared, so
nothing spills and there is nothing to trip on).

Prints `SPILL {json}`: allocated GiB and shared GiB at the trip, seconds from the start to the clamp, how the fill ended
(OOM after the trip, host floor, or the cap), and the host available low point. Pass: "end": "OOM after the trip".
Safety: stops filling below --host-floor-gb host available, whatever the guard does.
"""
import argparse, json, subprocess, sys, threading, time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "lf_spill_guard"))
import spill_guard as sg

GB = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-mb", type=int, default=256)
    ap.add_argument("--threshold-gb", type=float, default=1.0)
    ap.add_argument("--host-floor-gb", type=float, default=6.0)
    ap.add_argument("--pause-s", type=float, default=0.05)
    ap.add_argument("--hog-gb", type=float, default=0.0)
    a = ap.parse_args()
    dev = 0
    hog = None
    if a.hog_gb > 0:  # a second GPU process holding N GB: the budget shortfall the real spill had
        code = ("import sys, time, torch; x = torch.ones(int(%r * 2**30) // 2, dtype=torch.float16, device='cuda'); "
                "torch.cuda.synchronize(); print('HOG ready', flush=True); time.sleep(420)" % a.hog_gb)
        hog = subprocess.Popen([sys.executable, "-s", "-c", code], stdout=subprocess.PIPE, text=True)
        __import__("atexit").register(hog.kill)  # never leave the helper holding VRAM, whatever ends this process
        watchdog = threading.Timer(60, hog.kill); watchdog.start()  # a HIP-init hang must not hang the bench
        ready = hog.stdout.readline().strip(); watchdog.cancel()
        if ready != "HOG ready":
            hog.kill(); raise SystemExit("hog process failed to allocate")
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(1.0, dev)  # lf_vram_cap's setting at the time of the measurement
    mem = sg.ProcessGpuMemory()
    row = {"threshold_gb": a.threshold_gb, "step_mb": a.step_mb, "total_gib": round(total / GB, 2)}
    trip_info = {}
    def trip(ded, sh, ids):
        reserved = torch.cuda.memory_reserved(dev)
        torch.cuda.set_per_process_memory_fraction(max(reserved / total, 0.01), dev)
        trip_info.update(t=round(time.perf_counter() - t0, 1), shared_gb=round(sh / GB, 2), dedicated_gb=round(ded / GB, 2),
                         reserved_gib=round(reserved / GB, 2))
    g = sg.Guard(mem.read, trip, lambda: None, lambda: ["bench"], threshold=a.threshold_gb * GB)
    t0 = time.perf_counter()
    stop = threading.Event(); low = [psutil.virtual_memory().available]
    def loop():
        while not stop.is_set():
            try: g.step()
            except Exception as e: row.setdefault("read_errors", []).append(f"{type(e).__name__}: {e}"[:120])
            low[0] = min(low[0], psutil.virtual_memory().available); time.sleep(1.0)
    threading.Thread(target=loop, daemon=True).start()
    held, n, end = [], a.step_mb * 2 ** 20 // 2, "cap"
    try:
        while time.perf_counter() - t0 < 300:
            if psutil.virtual_memory().available < a.host_floor_gb * GB: end = "host floor"; break
            held.append(torch.ones(n, dtype=torch.float16, device="cuda")); torch.cuda.synchronize()
            time.sleep(a.pause_s)
    except torch.OutOfMemoryError as e:
        end = "OOM after the trip" if trip_info else "OOM at the cap (no trip)"
        row["reserved_at_oom_gib"] = round(torch.cuda.memory_reserved(dev) / GB, 2)
        row["oom"] = str(e)[:160]
    time.sleep(1.5)
    stop.set()
    row.update(end=end, allocated_gib=round(len(held) * n * 2 / GB, 2), trip=trip_info or None,
               peak_shared_gb=round(g.peak / GB, 2), host_low_gb=round(low[0] / GB, 2),
               seconds=round(time.perf_counter() - t0, 1))
    del held; torch.cuda.empty_cache(); torch.cuda.set_per_process_memory_fraction(1.0, dev)
    if hog is not None: hog.kill(); hog.wait(); row["hog_gb"] = a.hog_gb
    print("SPILL " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
