"""lf_spill_guard: stop a ComfyUI prompt the moment WDDM starts backing its GPU memory with host memory (7 Nov 2026).
Adds no nodes.

A real image-to-3D prompt (7 Nov) showed the spill path. WDDM gave the ComfyUI process a dGPU budget of about
11.5 GB, under lf_vram_cap's 15.92 GiB cap. Past it, the process's **shared** GPU memory (host RAM) rose 0.1 -> 6.3 GB
in 20 s; then host available memory collapsed 5.2 -> 0.1 GB in 4 s. The cap never fired before the host was gone.
Shared usage rising is the earliest signal, and it is per process: Windows' "GPU Process Memory(pid_<pid>_luid_..)"
performance counters, read here through PDH by ctypes (no GPU access).

A daemon thread reads this process's shared usage every 0.25 s (on the adapter where the process has the most
dedicated memory). While a prompt runs and shared usage is more than `LF_SPILL_GUARD_GB` (default 1.0) above what was
left in the last idle spell (its maximum, counting for at most the threshold), it trips:
  * torch's memory fraction is clamped to what torch has reserved now, so the next allocation is a HIP OOM instead of
    more spill (an interrupt alone waits for the next node boundary);
  * each prompt running at that moment is interrupted by id (the queue's atomic interrupt_if_running), and the queue's
    unload_models and free_memory flags are set (what POST /free does);
  * one JSON line goes to `LF_SPILL_GUARD_LOG` (default ComfyUI/user/lf_spill_guard.jsonl) and the log.
Once none of those prompts is still running, the previous fraction is restored when shared usage is under the idle
residue plus half the threshold, or at once when no prompt runs at all (the clamp then protects nothing). With no prompt
running it never trips (nothing to stop; a clamp to the idle reservation failed the next prompts at once, and held a
process clamped for 2 minutes); shared usage above the threshold while idle is recorded once per spell ("idle-shared").
A trip or a restore that raises is retried at the next reading; a reading that fails changes nothing (an unreadable
counter is unknown, never zero).
"""
import json, logging, os, threading, time

try:  # the package import inside ComfyUI, and a plain import from tests and the bench
    from .spill_core import GB, PDH_MORE_DATA, ProcessGpuMemory, Guard, busiest, setting, _Item
    from . import oom_drop, dxgi_budget
except ImportError:
    from spill_core import GB, PDH_MORE_DATA, ProcessGpuMemory, Guard, busiest, setting, _Item
    import oom_drop, dxgi_budget

log = logging.getLogger("lf_spill_guard")
GUARD = None  # the installed Guard, for inspection


def cap_fraction():
    """lf_vram_cap's own setting (its env var and its default, 0.92); 0 (cap off) or an odd value: the whole card."""
    try:
        f = float(os.environ.get("LF_VRAM_CAP_FRACTION", "0.92"))
    except ValueError:  # malformed: never let it stop a trip
        return 1.0
    return f if 0 < f <= 1 else 1.0


def _record(path, **kw):
    try:  # a log line must never fail an action
        line = json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), **kw))
        log.warning(f"lf_spill_guard: {line}"); print(f"[lf_spill_guard] {line}", flush=True)
        with open(path, "a", encoding="utf8") as f: f.write(line + "\n")
    except Exception: pass


def _budget_reader(luid=None):
    """A function returning this process's WDDM budget in GB (DXGI QueryVideoMemoryInfo), or None, on the adapter whose
    LUID low part is `luid` (where the process holds its memory, as the PDH counters name it), else the AMD adapter with
    the most VRAM. A ComfyUI process spilled at 10.8-11.5 GB dedicated with the card otherwise empty; the trip records
    the budget Windows gave the process then. Never raises; the other adapters' references are released."""
    adapter = None
    try:
        found = dxgi_budget.adapters()
        mine = [a for a in found if luid and a["luid"].lower() == luid.lower()]
        amd = [a for a in found if a["vendor"] == 0x1002]
        adapter = (mine or [max(amd, key=lambda a: a["dedicated_gb"])] if (mine or amd) else [None])[0]
        for a in found:
            if a is not adapter: dxgi_budget.release(a)
    except Exception:
        adapter = None

    def read():
        try:
            return dxgi_budget.gib(dxgi_budget.query(adapter)) if adapter else None
        except Exception:
            return None
    return read


def _bounded(fn, seconds):
    """fn() on a daemon thread, waited for at most `seconds`: None if it has not returned (a hung driver call must not
    hold the guard's startup)."""
    out = []
    t = threading.Thread(target=lambda: out.append(fn()), name="lf_spill_guard-budget", daemon=True)
    t.start(); t.join(seconds)
    return out[0] if out else None


def _luid_of(reading):
    """The LUID low part ("0x0000A1B2") of the adapter where this process has the most dedicated memory, from
    ProcessGpuMemory.read()'s keys (pid_<pid>_luid_0x<high>_0x<low>_phys_<n>), or None."""
    try:
        key = max(reading, key=lambda k: reading[k][0])
        return "0x" + key.split("_luid_")[1].split("_")[1][2:].upper()
    except Exception:
        return None


def install():
    import torch
    import comfy.model_management as mm
    import folder_paths
    from server import PromptServer
    threshold_gb, interval = setting("LF_SPILL_GUARD_GB", "1.0"), setting("LF_SPILL_GUARD_INTERVAL", "0.25")  # a 1 s poll let 2.86 GB spill before the trip (7 Nov)
    if not (torch.version.hip and torch.cuda.is_available()):
        raise RuntimeError("not a ROCm build with a GPU: nothing to guard")
    mem = ProcessGpuMemory()
    ded, sh = busiest(mem.read())  # fail here, at startup, if the counters are unreadable; no thread is running yet
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    state = {"fraction": None}
    path = os.environ.get("LF_SPILL_GUARD_LOG") or os.path.join(folder_paths.get_user_directory(), "lf_spill_guard.jsonl")
    budget = _bounded(lambda: _budget_reader(_luid_of(mem.read())), 5.0) or (lambda: None)

    def queue():
        return PromptServer.instance.prompt_queue

    def running():
        q = queue()
        live = q.get_current_queue_volatile() if hasattr(q, "get_current_queue_volatile") else q.get_current_queue()
        return [item[1] for item in live[0]]

    def trip(ded, sh, ids):
        reserved = torch.cuda.memory_reserved(dev)
        if state["fraction"] is None:  # validated before it is saved, so a restore never sets an odd fraction
            try:
                f = torch.cuda.get_per_process_memory_fraction(dev)
            except AttributeError:  # older torch: lf_vram_cap's own setting
                f = cap_fraction()
            state["fraction"] = f if 0 < f <= 1 else cap_fraction()
        clamp = min(max(reserved / total, 0.01), 1.0)
        clamp_error = None
        try:
            torch.cuda.set_per_process_memory_fraction(clamp, dev)
        except Exception as e:  # the interrupt and the free still run; the trip is retried next reading
            clamp_error = e
        try:
            q = queue()
            for pid in ids:
                if hasattr(q, "interrupt_if_running"): q.interrupt_if_running(pid)
                else: mm.interrupt_current_processing(True)
            oom_drop.arm(q)  # unload_models + free_memory, and the main loop drops models instead of offloading to host
        except Exception:  # undo the clamp: never leave it on while the guard is not tripped and watching
            if clamp_error is None: torch.cuda.set_per_process_memory_fraction(state["fraction"], dev)
            raise
        if clamp_error is not None:
            _record(path, event="clamp-failed", error=f"{type(clamp_error).__name__}: {clamp_error}"[:200])
            raise clamp_error
        _record(path, event="stopped", torch_reserved_gb=round(reserved / GB, 2), clamped_fraction=round(clamp, 3))
        # off the trip's path: a separate record, so a slow or hung DXGI call never holds the clamp, the interrupt or
        # the guard's next readings
        threading.Thread(target=lambda: _record(path, event="budget", budget_gb=budget()), name="lf_spill_guard-budget",
                         daemon=True).start()

    def clear():
        if state["fraction"] is not None:
            f = state["fraction"]
            torch.cuda.set_per_process_memory_fraction(f, dev)
            state["fraction"] = None
            _record(path, event="restored", fraction=f)

    global GUARD
    GUARD = guard = Guard(mem.read, trip, clear, running, threshold_gb * GB, record=lambda **kw: _record(path, **kw))

    def loop():
        failing = False
        while True:
            try:
                guard.step()
                if failing: log.warning("lf_spill_guard: readings back"); failing = False
            except Exception as e:  # log once per failure streak; the next step retries
                if not failing:
                    msg = f"lf_spill_guard: step failed (retrying every {interval:g} s): {type(e).__name__}: {e}"
                    log.warning(msg); print(f"[lf_spill_guard] {msg}", flush=True); failing = True
            time.sleep(interval)

    drop = oom_drop.install(mm, queue)
    threading.Thread(target=loop, name="lf_spill_guard", daemon=True).start()
    return (f"threshold {threshold_gb:.1f} GB shared, every {interval:g} s, {drop}; now {sh / GB:.2f} GB shared, "
            f"{ded / GB:.2f} GB dedicated; WDDM budget {_bounded(budget, 2.0)}; log {path}")
