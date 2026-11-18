"""lf_spill_guard core: per-process GPU memory from Windows performance counters, and the trip/restore state
machine with action callbacks. No ComfyUI and no torch (stdlib ctypes only): any host process can own it; the
ComfyUI node (spill_guard.py) supplies the actions (clamp, interrupt, free) and the running-prompt ids.
"""
import ctypes, math, os, time
from ctypes import wintypes

GB = 2 ** 30
PDH_FMT_LARGE, PDH_MORE_DATA = 0x400, 0x800007D2
PDH_CSTATUS_VALID, PDH_CSTATUS_NEW = 0, 1


def setting(name, default):
    v = float(os.environ.get(name, default))
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be a positive number, not {os.environ.get(name)!r}")
    return v


class _Value(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]  # union member at 8-byte alignment


class _Item(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _Value)]


class ProcessGpuMemory:
    """This process's dedicated and shared GPU memory per adapter, from PDH's GPU Process Memory counters."""

    def __init__(self, pid=None):
        self.prefix = f"pid_{pid or os.getpid()}_"
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        self._ok(self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query)), "PdhOpenQuery")
        try:
            self.counters = {}
            for key in ("Dedicated Usage", "Shared Usage"):
                h = ctypes.c_void_p()
                self._ok(self.pdh.PdhAddEnglishCounterW(self.query, f"\\GPU Process Memory(*)\\{key}", None, ctypes.byref(h)),
                         f"PdhAddEnglishCounter {key}")
                self.counters[key] = h
            self.pdh.PdhCollectQueryData(self.query)
        except Exception:
            self.close(); raise

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query); self.query = ctypes.c_void_p()

    @staticmethod
    def _ok(status, what):
        if status != 0: raise OSError(f"{what} failed: 0x{status & 0xFFFFFFFF:08X}")

    def _array(self, h, key):
        for _ in range(3):  # PDH_MORE_DATA on the second call: the instance set grew in between, size again
            size, count = wintypes.DWORD(0), wintypes.DWORD(0)
            st = self.pdh.PdhGetFormattedCounterArrayW(h, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
            if st & 0xFFFFFFFF != PDH_MORE_DATA:
                self._ok(st or 0xFFFFFFFF, f"PdhGetFormattedCounterArray {key} (sizing)")
            buf = (ctypes.c_byte * size.value)()
            st = self.pdh.PdhGetFormattedCounterArrayW(h, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf)
            if st & 0xFFFFFFFF == PDH_MORE_DATA: continue
            self._ok(st, f"PdhGetFormattedCounterArray {key}")
            items = ctypes.cast(buf, ctypes.POINTER(_Item))
            out = {}
            for i in range(count.value):
                it = items[i]
                if not it.szName.startswith(self.prefix): continue
                if it.FmtValue.CStatus not in (PDH_CSTATUS_VALID, PDH_CSTATUS_NEW):  # unknown, never zero
                    raise OSError(f"{key} for {it.szName}: counter status 0x{it.FmtValue.CStatus:08X}")
                out[it.szName] = it.FmtValue.largeValue
            return out
        raise OSError(f"{key}: the instance set kept changing")

    def read(self):
        """{instance: (dedicated_bytes, shared_bytes)} for this process. Raises when a value is unavailable."""
        self._ok(self.pdh.PdhCollectQueryData(self.query), "PdhCollectQueryData")
        ded = self._array(self.counters["Dedicated Usage"], "Dedicated Usage")
        sh = self._array(self.counters["Shared Usage"], "Shared Usage")
        if set(ded) != set(sh):
            raise OSError(f"dedicated and shared instances differ: {sorted(set(ded) ^ set(sh))}")
        return {k: (ded[k], sh[k]) for k in ded}


def busiest(per_adapter):
    """(dedicated, shared) bytes on the adapter where the process holds the most dedicated memory, or (0, 0)."""
    return max(per_adapter.values(), default=(0, 0))  # by dedicated, then shared


class Guard:
    """The trip logic, separate from Windows and ComfyUI so it can be tested. `running()` returns the ids of the
    prompts running now; `trip(ded, shared, ids)` and `clear()` act, and are retried next step if they raise."""

    SETTLE_S = 60  # after this, a reading under the threshold (not half) restores: WDDM may trim lazily
    # With no prompt running there is nothing to stop, and a clamp to torch's idle reservation (0.014 of VRAM,
    # measured 8 Nov) fails every next prompt at once. Shared memory left over after a trip (1.5-2.0 GB, slow to page back)
    # caused exactly that: a clear, then a re-trip with no prompt running. So the guard never trips while idle, and
    # measures a prompt's spill above the idle residue, which counts for at most one threshold: a prompt trips at
    # base + threshold, never above 2x the threshold in total.

    def __init__(self, read, trip, clear, running, threshold, record=lambda **kw: None, clock=time.monotonic):
        self.read, self.trip_fn, self.clear_fn, self.running = read, trip, clear, running
        self.threshold, self.record, self.clock = threshold, record, clock
        self.tripped, self.ids, self.since, self.noted = False, frozenset(), 0.0, 0.0
        self.pending = None  # prompt ids of a trip being retried: captured once, so a retry never hits a successor
        self.peak = 0
        self.base = 0  # the most shared memory seen in the last idle spell, capped at the threshold
        self.idle_max, self.idle_high = 0, False

    def step(self):
        ded, sh = busiest(self.read())  # a failed read raises here, before any state changes
        self.peak = max(self.peak, sh)
        run = set(self.running())
        if not run:  # a spell's maximum, not its last reading: one dip (0.08 between readings of 1.72) must not set the base
            self.idle_max = max(self.idle_max, sh)
        else:
            self.idle_max = 0
        if not self.tripped and not run:
            if self.pending is not None:  # a trip that raised may have left its clamp on: undo it before forgetting it
                self.clear_fn()            # (raises: pending stays, retried next reading)
            self.pending, self.base = None, min(self.idle_max, self.threshold)
            if sh > self.threshold and not self.idle_high:  # recorded once per idle spell above the threshold
                self.record(event="idle-shared", shared_gb=round(sh / GB, 2), dedicated_gb=round(ded / GB, 2))
            self.idle_high = sh > self.threshold
        elif not self.tripped and sh > self.base + self.threshold:
            if self.pending is None: self.pending = frozenset(run)
            ids = self.pending
            self.trip_fn(ded, sh, ids)
            self.tripped, self.ids, self.since, self.pending = True, ids, self.clock(), None
            self.noted = self.since
            self.record(event="trip", shared_gb=round(sh / GB, 2), dedicated_gb=round(ded / GB, 2), prompts=sorted(ids),
                        idle_base_gb=round(self.base / GB, 2))
        elif not self.tripped:
            self.pending = None
        else:
            waited = self.clock() - self.since
            low = sh < self.base + self.threshold / 2 or (waited > self.SETTLE_S and sh <= self.base + self.threshold)
            # with no prompt running the clamp protects nothing and would fail the next prompt (a process held a 0.787
            # clamp idle for 2 minutes): restore, and let the idle residue set the base for the next prompt's trip
            if (low or not run) and not (self.ids & run):
                self.clear_fn()
                self.tripped, self.ids = False, frozenset()
                if not run: self.base = min(self.idle_max, self.threshold)
                self.record(event="clear", shared_gb=round(sh / GB, 2), dedicated_gb=round(ded / GB, 2))
            elif self.clock() - self.noted > self.SETTLE_S:  # never clamped in silence
                self.noted = self.clock()
                self.record(event="still-clamped", seconds=round(waited), shared_gb=round(sh / GB, 2), prompts=sorted(self.ids))
        return ded, sh
