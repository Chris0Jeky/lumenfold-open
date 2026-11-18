"""lf_spill_guard without ComfyUI, torch or a GPU: the trip logic on readings from a measured 7 Nov spill, failure and
queue cases, install() with stand-in torch and ComfyUI modules, the PDH reader against fakes and (Windows only) against
typeperf, and the DXGI budget reader with fakes. Real torch is never imported.

  python test_spill_guard.py   (or pytest)
"""
import json, os, subprocess, sys, types, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lf_spill_guard"))
import spill_guard as sg

GB = 2 ** 30
# a ComfyUI process's shared usage on 7 Nov over 39 s (adapter-wide, sampled at 1 s): baseline, then the spill
SPILL_TRACE = [0.1] * 21 + [2.9, 2.9, 2.9, 1.8, 4.6, 6.3]


class Trip(unittest.TestCase):
    def make(self, shared_gb, running=("p1",)):
        self.events, self.run, self.now = [], set(running), 0.0
        self.fail_trip = self.fail_clear = 0
        self.readings = iter(shared_gb)
        def read():
            sh = next(self.readings)
            if sh is None: raise OSError("counter status 0xC0000BC6")
            return {"pid_1_luid_igpu": (0, int(0.01 * GB)), "pid_1_luid_dgpu": (int(9.5 * GB), int(sh * GB))}
        def trip(d, s, ids):
            if self.fail_trip: self.fail_trip -= 1; raise RuntimeError("set_per_process_memory_fraction failed")
            self.events.append(("trip", round(s / GB, 1), tuple(sorted(ids))))
        def clear():
            if self.fail_clear: self.fail_clear -= 1; raise RuntimeError("restore failed")
            self.events.append(("clear",))
        return sg.Guard(read, trip, clear, lambda: set(self.run), 1.0 * GB, clock=lambda: self.now)

    def steps(self, g, n):
        for _ in range(n):
            try: g.step()
            except Exception: pass
            self.now += 1

    def test_the_measured_spill_trips_once_at_the_first_reading_over_1_gb(self):
        g = self.make(SPILL_TRACE)
        for i in range(len(SPILL_TRACE)):
            g.step()
            if self.events: break
        self.assertEqual((i, self.events), (21, [("trip", 2.9, ("p1",))]))  # the first spill reading, about 20 s before host available hit 0.1 GB
        self.steps(g, len(SPILL_TRACE) - i - 1)
        self.assertEqual(len(self.events), 1)  # no second trip while tripped

    def test_baseline_never_trips(self):
        g = self.make([0.1, 0.5, 0.99, 1.0] * 5)
        self.steps(g, 20)
        self.assertEqual(self.events, [])

    def test_clears_when_under_half_and_the_tripped_prompt_is_gone(self):
        g = self.make([2.0, 0.8, 0.4, 0.4, 0.4])
        g.step()                               # trip on p1
        self.run = {"p2"}; g.step()            # p1 gone (p2 runs), but 0.8 is not under half the threshold
        self.run = {"p1"}; g.step()            # 0.4 but p1 still runs
        self.assertEqual(len(self.events), 1)
        self.run = {"p2"}; g.step()            # p1 gone and 0.4: clear
        g.step()
        self.assertEqual(self.events[1:], [("clear",)])

    def test_a_queue_backlog_does_not_keep_the_clamp(self):
        # with a backlog, some prompt is always running; only the prompts running at the trip hold the clamp
        g = self.make([2.0, 0.3])
        g.step()
        self.run = {"p2"}; g.step()
        self.assertEqual(self.events[1:], [("clear",)])

    def test_a_failed_trip_is_retried(self):
        g = self.make([2.0, 2.0])
        self.fail_trip = 1
        self.assertRaises(RuntimeError, g.step)
        self.assertFalse(g.tripped)
        g.step()
        self.assertEqual(self.events, [("trip", 2.0, ("p1",))])

    def test_a_failed_restore_is_retried(self):
        g = self.make([2.0, 0.2, 0.2])
        g.step(); self.run = set()
        self.fail_clear = 1
        self.assertRaises(RuntimeError, g.step)
        self.assertTrue(g.tripped)
        g.step()
        self.assertEqual(self.events[1:], [("clear",)])

    def test_a_retried_trip_keeps_the_prompts_of_the_first_attempt(self):
        seen = []
        g = self.make([2.0, 2.0])
        g.trip_fn = lambda d, s, ids: (seen.append(tuple(sorted(ids))), self.fail_trip and (_ for _ in ()).throw(RuntimeError()))
        self.fail_trip = 1
        self.assertRaises(RuntimeError, g.step)
        self.fail_trip = 0; self.run = {"p2"}  # p1 finished, its successor runs
        g.step()
        self.assertEqual(seen, [("p1",), ("p1",)])  # never p2

    def test_an_unreadable_counter_changes_nothing(self):
        g = self.make([2.0, None, None, 0.2])
        g.step(); self.run = set()
        self.assertRaises(OSError, g.step); self.assertRaises(OSError, g.step)
        self.assertTrue(g.tripped)            # unknown is not zero: no restore
        g.step()
        self.assertEqual(self.events[1:], [("clear",)])

    def test_settled_shared_between_half_and_threshold_restores_after_a_minute(self):
        records = []
        g = self.make([2.0] + [0.8] * 70)
        g.record = lambda **kw: records.append(kw["event"])
        g.step(); self.run = {"p2"}           # a backlog prompt runs: only settling restores
        self.steps(g, 30); self.assertEqual(len(self.events), 1)
        self.steps(g, 40); self.assertEqual(self.events[1:], [("clear",)])
        self.assertEqual(records, ["trip", "clear"])

    def test_idle_shared_never_trips_and_is_recorded_once(self):
        records = []
        g = self.make([1.72] * 6 + [0.3] + [1.9] * 3, running=())
        g.record = lambda **kw: records.append(kw["event"])
        self.steps(g, 10)
        self.assertEqual(self.events, [])                          # no prompt: nothing to stop, no clamp
        self.assertEqual(records, ["idle-shared", "idle-shared"])  # once per spell above the threshold

    def test_a_re_trip_after_a_clear_no_longer_clamps_the_next_prompt(self):
        # measured 8 Nov: a trip, a clear at a 0.08 reading, then 1.72 GB shared again with no prompt running:
        # the old guard re-tripped and clamped to the idle reservation (0.014), failing the next two props at once
        g = self.make([1.72, 0.08, 1.72, 1.72, 1.72, 1.95, 2.1])
        g.step()                                   # trip on p1
        self.run = set(); g.step()                 # p1 gone, 0.08: clear
        g.step(); g.step()                         # idle at 1.72: no trip
        self.run = {"p2"}; g.step(); g.step()      # p2 starts over the residue: under base (1.0) + 1.0
        self.assertEqual(self.events, [("trip", 1.7, ("p1",)), ("clear",)])
        g.step()                                   # p2 spills past 2.0: trips
        self.assertEqual(self.events[2:], [("trip", 2.1, ("p2",))])

    def test_the_idle_residue_counts_for_at_most_one_threshold(self):
        g = self.make([5.0, 5.0, 2.01], running=())
        g.step(); g.step()
        self.run = {"p1"}; g.step()
        self.assertEqual(self.events, [("trip", 2.0, ("p1",))])   # base capped at 1.0: trips above 2.0, never 6.0

    def test_a_clear_after_an_idle_residue_is_measured_from_it(self):
        g = self.make([0.8, 2.5, 1.4, 1.2, 1.2], running=())
        g.step()                                   # idle base 0.8
        self.run = {"p1"}; g.step()                # 2.5 > 1.8: trip
        self.run = {"p2"}; g.step()                # 1.4 is not under 0.8 + 0.5
        self.assertEqual(len(self.events), 1)
        g.step()                                   # 1.2 is: clear
        self.assertEqual(self.events[1:], [("clear",)])

    def test_a_trip_restores_once_no_prompt_runs_even_over_a_high_residue(self):
        # residue that pages back slowly kept the clamp on with nothing running (2 minutes, measured)
        g = self.make([3.0, 1.8, 1.8, 1.8, 2.9])
        g.step()                                   # trip on p1
        self.run = set(); g.step()                 # p1 ended, 1.8 left: restore at once
        self.assertEqual(self.events, [("trip", 3.0, ("p1",)), ("clear",)])
        g.step()                                   # idle at 1.8: no re-trip, base 1.0
        self.run = {"p2"}; g.step()                # p2 starts on the residue: under 2.0
        self.assertEqual(len(self.events), 2)
        g.step()                                   # p2 spills: trips
        self.assertEqual(self.events[2:], [("trip", 2.9, ("p2",))])

    def test_a_failed_trip_then_idle_undoes_any_clamp(self):
        # a trip that raised after its clamp landed, then its prompt ended: an idle reading must not
        # forget the retry and leave that clamp on
        g = self.make([2.0, 0.3, 0.3])
        self.fail_trip = 1
        self.assertRaises(RuntimeError, g.step)
        self.run = set(); self.fail_clear = 1
        self.assertRaises(RuntimeError, g.step)   # the undo failed: still pending, retried
        g.step()
        self.assertEqual(self.events, [("clear",)])
        self.assertFalse(g.tripped)
        self.assertIsNone(g.pending)

    def test_one_idle_dip_does_not_lower_the_base(self):
        g = self.make([1.72, 1.72, 0.08, 1.72, 2.6], running=())
        g.step(); g.step(); g.step()               # a dip at the last idle reading before p1
        self.run = {"p1"}; g.step()                # p1 starts on 1.72: the spell's max (1.0 capped) is the base
        self.assertEqual(self.events, [])
        g.step()
        self.assertEqual(self.events, [("trip", 2.6, ("p1",))])

    def test_a_long_clamp_is_logged(self):
        records = []
        g = self.make([2.0] + [3.0] * 130)
        g.record = lambda **kw: records.append(kw["event"])
        self.steps(g, 131)
        self.assertEqual(records, ["trip", "still-clamped", "still-clamped"])

    def test_reads_the_adapter_holding_the_dedicated_memory(self):
        self.assertEqual(sg.busiest({"igpu": (0, 5 * GB), "dgpu": (9 * GB, 2)}), (9 * GB, 2))
        self.assertEqual(sg.busiest({}), (0, 0))

    def test_settings_must_be_positive_numbers(self):
        for bad in ("nan", "-1", "0", "inf"):
            os.environ["LF_SPILL_GUARD_GB"] = bad
            try: self.assertRaises(ValueError, sg.setting, "LF_SPILL_GUARD_GB", "1.0")
            finally: os.environ.pop("LF_SPILL_GUARD_GB")


class Install(unittest.TestCase):
    """install() with stand-in torch, ComfyUI and counters: what a trip does to the allocator and the queue."""

    def setUp(self):
        self.calls = []
        self.saved = {k: sys.modules.get(k) for k in ("torch", "comfy", "comfy.model_management", "folder_paths", "server")}
        self.cap_env = os.environ.pop("LF_VRAM_CAP_FRACTION", None)  # the shell's value must not decide these tests
        self.mem, self.thread = sg.ProcessGpuMemory, sg.threading.Thread
        self.addCleanup(self.restore)
        t = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2"))
        self.frac = [1.0]
        t.cuda = types.SimpleNamespace(
            is_available=lambda: True, current_device=lambda: 0,
            get_device_properties=lambda d: types.SimpleNamespace(total_memory=16 * GB),
            memory_reserved=lambda d: 14 * GB,
            get_per_process_memory_fraction=lambda d: self.frac[0],
            set_per_process_memory_fraction=lambda f, d: (self.frac.__setitem__(0, f), self.calls.append(("fraction", round(f, 3)))))
        self.moved = []
        class Model:
            def __init__(m, name): m.name, m.model_loaded_weight_memory, m.model_lowvram = name, 5 * 2**30, True
            def to(m, dev): self.moved.append((m.name, dev)); return m
        self.loaded = [types.SimpleNamespace(model=types.SimpleNamespace(model=Model(k))) for k in ("Trellis2", "ShapeVae")]
        mm = types.SimpleNamespace(interrupt_current_processing=lambda v=True: self.calls.append(("global interrupt", v)),
                                   is_oom=lambda e: "out of memory" in str(e),
                                   unload_all_models=lambda: self.calls.append(("offload to host",)),
                                   current_loaded_models=self.loaded, soft_empty_cache=lambda: None)
        self.mm = mm
        sg.oom_drop._armed.clear()
        self.main_file = sg.oom_drop.MAIN_FILE; sg.oom_drop.MAIN_FILE = lambda: "ComfyUI/main.py"
        self.addCleanup(lambda: setattr(sg.oom_drop, "MAIN_FILE", self.main_file))
        self.resets = []
        self.running = [[(0, "p1", {})]]
        q = types.SimpleNamespace(get_current_queue_volatile=lambda: (self.running[0], []),
                                  interrupt_if_running=lambda pid: self.calls.append(("interrupt", pid)),
                                  set_flag=lambda k, v: self.calls.append(("flag", k, v)))
        server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(prompt_queue=q)))
        comfy = types.ModuleType("comfy"); comfy.model_management = mm
        fp = types.SimpleNamespace(get_user_directory=lambda: str(Path(__file__).resolve().parent))
        sys.modules.update({"torch": t, "comfy": comfy, "comfy.model_management": mm, "folder_paths": fp, "server": server})
        self.shared = [0.1]
        class FakeMem:
            def __init__(s, pid=None): pass
            def read(s): return {"pid_1_luid_dgpu": (int(11.5 * GB), int(self.shared[0] * GB))}
        sg.ProcessGpuMemory = FakeMem
        sg.threading.Thread = lambda **kw: types.SimpleNamespace(start=lambda: None)  # the test drives step()
        # no real DXGI in these tests: the budget reader is covered by class Budget
        real_reader, real_bounded = sg._budget_reader, sg._bounded
        sg._budget_reader, sg._bounded = (lambda luid=None: (lambda: None)), (lambda fn, seconds: fn())
        self.addCleanup(lambda: (setattr(sg, "_budget_reader", real_reader), setattr(sg, "_bounded", real_bounded)))
        self.log = Path(__file__).resolve().parent / "lf_spill_guard.test.jsonl"
        os.environ["LF_SPILL_GUARD_LOG"] = str(self.log)

    def restore(self):
        for k, v in self.saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
        sg.ProcessGpuMemory, sg.threading.Thread = self.mem, self.thread
        os.environ.pop("LF_SPILL_GUARD_LOG", None)
        os.environ.pop("LF_VRAM_CAP_FRACTION", None)
        if self.cap_env is not None: os.environ["LF_VRAM_CAP_FRACTION"] = self.cap_env
        if self.log.exists(): self.log.unlink()

    def test_trip_clamps_to_reserved_interrupts_by_id_and_frees_then_restores(self):
        self.assertIn("threshold 1.0 GB shared, every 0.25 s", sg.install())  # 0.25 s: a 1 s poll let 2.86 GB spill first
        g = sg.GUARD
        g.step(); self.assertEqual(self.calls, [])
        self.shared[0] = 2.9; g.step()
        self.assertEqual(self.calls, [("fraction", 0.875), ("interrupt", "p1"),  # 14 of 16 GB reserved: no more growth
                                      ("flag", "free_memory", True), ("flag", "unload_models", True)])
        self.shared[0] = 0.2; g.step()          # p1 still running: stays clamped
        self.assertEqual(len(self.calls), 4)
        self.running[0] = [(1, "p2", {})]; g.step()  # p1 gone (p2 from the backlog runs): lf_vram_cap's fraction back
        self.assertEqual(self.calls[-1], ("fraction", 1.0))
        events = [json.loads(l)["event"] for l in self.log.read_text().splitlines()]
        self.assertEqual(events, ["stopped", "trip", "restored", "clear"])

    def test_a_second_trip_keeps_the_original_fraction(self):
        sg.install(); g = sg.GUARD
        self.shared[0] = 2.9; g.step()
        self.running[0] = []; self.shared[0] = 0.2; g.step()
        self.running[0] = [(1, "p2", {})]; self.shared[0] = 2.9; g.step()
        self.running[0] = []; self.shared[0] = 0.2; g.step()
        self.assertEqual(self.calls[-1], ("fraction", 1.0))  # never ratchets down to a clamped value

    def test_a_retried_trip_restores_the_fraction_from_before_the_first_attempt(self):
        sg.install(); g = sg.GUARD
        q = sys.modules["server"].PromptServer.instance.prompt_queue
        ok = q.set_flag
        q.set_flag = lambda k, v: (_ for _ in ()).throw(RuntimeError("flag failed"))  # clamp done, then the trip fails
        self.shared[0] = 2.9
        self.assertRaises(RuntimeError, g.step)
        q.set_flag = ok; g.step()                   # retried while clamped at 0.875
        self.running[0] = []; self.shared[0] = 0.2; g.step()
        self.assertEqual(self.calls[-1], ("fraction", 1.0))  # not the clamped 0.875

    def test_a_failed_trip_undoes_its_clamp(self):
        # the clamp lands, set_flag raises, then the spill subsides; no clamp may be left behind
        sg.install(); g = sg.GUARD
        q = sys.modules["server"].PromptServer.instance.prompt_queue
        ok = q.set_flag
        q.set_flag = lambda k, v: (_ for _ in ()).throw(RuntimeError("flag failed"))
        self.shared[0] = 2.9
        self.assertRaises(RuntimeError, g.step)
        self.assertEqual(self.calls[-1], ("fraction", 1.0))  # undone at once
        q.set_flag = ok; self.shared[0] = 0.2; g.step()
        self.assertFalse(g.tripped)

    def test_a_failed_clamp_still_interrupts_and_frees(self):
        # the emergency interrupt must not depend on the clamp
        sg.install(); g = sg.GUARD
        cuda = sys.modules["torch"].cuda
        ok = cuda.set_per_process_memory_fraction
        cuda.set_per_process_memory_fraction = lambda f, d: (_ for _ in ()).throw(ValueError("fraction out of range"))
        self.shared[0] = 2.9
        self.assertRaises(ValueError, g.step)
        self.assertEqual(self.calls, [("interrupt", "p1"), ("flag", "free_memory", True), ("flag", "unload_models", True)])
        self.assertFalse(g.tripped)            # retried at the next reading
        cuda.set_per_process_memory_fraction = ok; g.step()
        self.assertTrue(g.tripped)

    def restored_after_one_trip(self):
        sg.install(); g = sg.GUARD
        self.running[0] = [(0, "p1", {})]  # a prompt runs at the trip (an idle guard never trips)
        self.shared[0] = 2.9; g.step()
        self.running[0] = []; self.shared[0] = 0.2; g.step()
        return self.calls[-1]

    def test_an_unset_fraction_restores_lf_vram_caps_setting(self):
        self.frac[0] = 0.0  # what a getter could report with no cap set
        self.assertEqual(self.restored_after_one_trip(), ("fraction", 0.92))  # lf_vram_cap's default

    def test_without_the_getter_lf_vram_caps_setting_is_restored(self):
        # the fallback must not restore 1.0 over lf_vram_cap's 0.92
        del sys.modules["torch"].cuda.get_per_process_memory_fraction
        self.assertEqual(self.restored_after_one_trip(), ("fraction", 0.92))
        os.environ["LF_VRAM_CAP_FRACTION"] = "0.8"
        try:
            self.calls.clear(); self.assertEqual(self.restored_after_one_trip(), ("fraction", 0.8))
            os.environ["LF_VRAM_CAP_FRACTION"] = "0"  # cap off: the whole card
            self.calls.clear(); self.assertEqual(self.restored_after_one_trip(), ("fraction", 1.0))
        finally:
            os.environ.pop("LF_VRAM_CAP_FRACTION")

    def test_a_malformed_cap_setting_never_stops_a_trip(self):
        del sys.modules["torch"].cuda.get_per_process_memory_fraction
        for bad in ("abc", "nan", "1.5", "-1"):
            os.environ["LF_VRAM_CAP_FRACTION"] = bad
            self.calls.clear(); self.running[0] = [(0, "p1", {})]
            self.assertEqual(self.restored_after_one_trip(), ("fraction", 1.0), bad)
            self.assertIn(("interrupt", "p1"), self.calls)

    def test_an_odd_getter_value_is_never_saved(self):
        self.frac[0] = 0.0
        os.environ["LF_VRAM_CAP_FRACTION"] = "abc"  # and a malformed fallback on top
        self.assertEqual(self.restored_after_one_trip(), ("fraction", 1.0))

    def test_the_default_matches_lf_vram_caps(self):
        import re
        cap = Path(__file__).resolve().parents[1] / "rocm-vram-cap" / "lf_vram_cap" / "__init__.py"
        if not cap.exists(): self.skipTest("rocm-vram-cap is not beside this folder")
        src = cap.read_text(encoding="utf8")
        theirs = re.search(r'environ\.get\("LF_VRAM_CAP_FRACTION", "([0-9.]+)"\)', src).group(1)
        self.assertEqual(sg.cap_fraction(), float(theirs))

    def call_from(self, filename, code):
        # ComfyUI's callers, as frames from execution.py, main.py and another file
        e = types.SimpleNamespace(reset=lambda: self.resets.append(filename))  # main.py's prompt worker holds its executor as `e`
        exec(compile(code, "ComfyUI/" + filename, "exec"), {"mm": self.mm, "err": RuntimeError("HIP out of memory"), "e": e})

    def test_oom_in_a_node_drops_models_after_the_prompt_instead_of_offloading(self):
        sg.install(); dropped = list(self.loaded)
        self.call_from("execution.py", "mm.is_oom(err) and mm.unload_all_models()")  # ComfyUI's OOM handler
        self.assertNotIn(("offload to host",), self.calls)            # no host copy mid-prompt
        self.assertIn(("flag", "free_memory", True), self.calls)      # main.py resets the cache with the drop
        self.call_from("main.py", "mm.unload_all_models()")           # main loop, processing those flags
        self.assertEqual(self.moved, [("ShapeVae", "meta"), ("Trellis2", "meta")])
        self.assertEqual(self.resets, ["main.py"])                  # the cache reset happens with the drop, not later
        self.assertEqual([(m.model.model.model_loaded_weight_memory, m.model.model.model_lowvram) for m in dropped], [(0, False), (0, False)])
        self.assertEqual(self.loaded, [])
        self.assertNotIn(("offload to host",), self.calls)
        self.call_from("main.py", "mm.unload_all_models()")           # next ordinary unload: upstream again
        self.assertEqual(self.calls[-1], ("offload to host",))

    def test_a_guard_trip_arms_the_drop(self):
        sg.install(); g = sg.GUARD
        self.shared[0] = 2.9; g.step()
        self.call_from("main.py", "mm.unload_all_models()")
        self.assertEqual(len(self.moved), 2)

    def test_other_callers_keep_upstreams_unload(self):
        sg.install()
        self.call_from("execution.py", "mm.is_oom(err)")              # armed
        self.call_from("server.py", "mm.unload_all_models()")         # any caller other than main.py's worker: never a drop
        self.assertEqual(self.calls[-1], ("offload to host",)); self.assertEqual(self.moved, [])
        self.call_from("model_management.py", "mm.is_oom(err)")       # raise_non_oom inside comfy: does not arm
        self.call_from("main.py", "mm.unload_all_models()")           # the armed drop from execution.py
        self.call_from("main.py", "mm.unload_all_models()")
        self.assertEqual(self.calls[-1], ("offload to host",))

    def test_an_oom_check_inside_comfy_does_not_arm(self):
        sg.install()
        self.call_from("model_management.py", "mm.is_oom(err)")       # raise_non_oom and friends: not the node handler
        self.assertNotIn(("flag", "free_memory", True), self.calls)
        self.call_from("main.py", "mm.unload_all_models()")
        self.assertEqual(self.calls[-1], ("offload to host",)); self.assertEqual(self.moved, [])

    def test_a_non_oom_error_never_arms(self):
        sg.install()
        exec(compile("mm.is_oom(RuntimeError('shape mismatch'))", "ComfyUI/execution.py", "exec"), {"mm": self.mm})
        self.call_from("main.py", "mm.unload_all_models()")
        self.assertEqual(self.calls[-1], ("offload to host",)); self.assertEqual(self.moved, [])

    def test_the_drop_needs_comfyuis_main_launcher(self):
        sg.oom_drop.MAIN_FILE = lambda: "somewhere/other_tool.py"
        self.assertIn("OOM drop NOT armed", sg.install())
        self.call_from("execution.py", "mm.is_oom(err) and mm.unload_all_models()")
        self.assertEqual(self.calls[-1], ("offload to host",))       # upstream path, unchanged

    def test_arming_sets_free_memory_first_under_the_queue_mutex(self):
        order, held = [], []
        lock = __import__("threading").RLock()
        q = types.SimpleNamespace(mutex=lock, set_flag=lambda k, v: (order.append(k), held.append(lock._is_owned())))
        sg.oom_drop.arm(q)
        self.assertEqual(order, ["free_memory", "unload_models"]); self.assertTrue(all(held))
        self.assertTrue(sg.oom_drop._armed.is_set())

    def test_not_rocm_is_not_installed(self):
        sys.modules["torch"].version.hip = None
        self.assertRaises(RuntimeError, sg.install)


class FakePdh:
    """PdhGetFormattedCounterArrayW's two-call protocol over given (name, status, value) items."""
    def __init__(self, items, more_data_once=False):
        self.names = [__import__("ctypes").c_wchar_p(n) for n, _, _ in items]  # kept alive for szName
        self.items, self.more = items, more_data_once
    def PdhCollectQueryData(self, q): return 0
    def PdhGetFormattedCounterArrayW(self, h, fmt, size_p, count_p, buf):
        import ctypes
        arr = (sg._Item * len(self.items))()
        for i, ((_, st, v), nm) in enumerate(zip(self.items, self.names)):  # shared (handle 2) reads v // 4
            arr[i].szName = nm.value; arr[i].FmtValue.CStatus = st; arr[i].FmtValue.largeValue = v if h == 1 else v // 4
        size_p._obj.value, count_p._obj.value = ctypes.sizeof(arr), len(self.items)
        if buf is None: return sg.PDH_MORE_DATA
        if self.more: self.more = False; return sg.PDH_MORE_DATA  # the instance set grew between the calls
        ctypes.memmove(buf, arr, ctypes.sizeof(arr)); self.keep = arr
        return 0


class Reader(unittest.TestCase):
    def reader(self, pdh):
        m = sg.ProcessGpuMemory.__new__(sg.ProcessGpuMemory)
        m.prefix, m.pdh, m.query, m.counters = "pid_7_", pdh, None, {"Dedicated Usage": 1, "Shared Usage": 2}
        return m

    def test_parses_this_pid_only(self):
        r = self.reader(FakePdh([("pid_7_luid_a", 0, 5 * GB), ("pid_70_luid_a", 0, 9), ("pid_8_luid_b", 0, 1)])).read()
        self.assertEqual(r, {"pid_7_luid_a": (5 * GB, 5 * GB // 4)})  # (dedicated, shared), not swapped

    def test_an_invalid_status_raises_never_zero(self):
        m = self.reader(FakePdh([("pid_7_luid_a", 0xC0000BC6, 0)]))
        self.assertRaises(OSError, m.read)

    def test_more_data_on_the_second_call_is_retried(self):
        self.assertIn("pid_7_luid_a", self.reader(FakePdh([("pid_7_luid_a", 0, 3)], more_data_once=True)).read())

    def test_a_failed_sizing_call_raises(self):
        pdh = FakePdh([]); pdh.PdhGetFormattedCounterArrayW = lambda *a: 0xC0000BB8
        self.assertRaises(OSError, self.reader(pdh).read)


@unittest.skipUnless(os.name == "nt", "Windows performance counters")
class Pdh(unittest.TestCase):
    def test_reads_only_this_pid(self):
        r = sg.ProcessGpuMemory().read()
        self.assertTrue(all(k.startswith(f"pid_{os.getpid()}_") for k in r))

    def test_bytes_match_typeperf(self):
        out = subprocess.run(["typeperf", "-qx", "GPU Process Memory"], capture_output=True, text=True).stdout
        names = [l.split("(")[1].split(")")[0] for l in out.splitlines() if l.endswith("\\Shared Usage")]
        if not names: self.skipTest("no GPU process instances")
        m = sg.ProcessGpuMemory(); m.prefix = "pid_"  # every process
        mine = m.read()
        tp = subprocess.run(["typeperf", "-sc", "1"] + [f"\\GPU Process Memory({n})\\Shared Usage" for n in names[:20]],
                            capture_output=True, text=True).stdout.splitlines()
        hdr = next(l for l in tp if l.startswith('"(PDH'))
        row = next(l for l in tp[tp.index(hdr) + 1:] if l.startswith('"'))
        cols = [c.strip('"') for c in hdr.split('","')][1:]
        vals = [c.strip('"') for c in row.split('","')][1:]
        checked = 0
        for col, v in zip(cols, vals):
            inst = col.split("(")[1].split(")")[0]
            if inst in mine and v.strip():
                self.assertAlmostEqual(mine[inst][1], float(v), delta=max(64 * 2 ** 20, 0.1 * float(v)), msg=inst)
                checked += 1
        self.assertGreater(checked, 0)



class Budget(unittest.TestCase):  # the trip record's WDDM budget never fails the guard
    def setUp(self):
        self.real = sg.dxgi_budget.adapters, sg.dxgi_budget.query

    def tearDown(self):
        sg.dxgi_budget.adapters, sg.dxgi_budget.query = self.real

    def test_reads_the_amd_adapter_with_the_most_vram(self):
        sg.dxgi_budget.adapters = lambda: [{"vendor": 0x8086, "dedicated_gb": 0.1, "n": "igpu"},
                                           {"vendor": 0x1002, "dedicated_gb": 15.9, "n": "dgpu"}]
        sg.dxgi_budget.query = lambda a: {"Budget": 15 * 2 ** 30} if a["n"] == "dgpu" else {"Budget": 0}
        self.assertEqual(sg._budget_reader()(), {"Budget": 15.0})

    def test_prefers_the_adapter_where_the_process_holds_memory_and_releases_the_rest(self):
        released = []
        sg.dxgi_budget.adapters = lambda: [{"vendor": 0x1002, "dedicated_gb": 15.9, "luid": "0x0000AAAA", "n": "big"},
                                           {"vendor": 0x1002, "dedicated_gb": 7.9, "luid": "0x0000A1B2", "n": "mine"}]
        sg.dxgi_budget.query = lambda a: {"Budget": 2 ** 30 if a["n"] == "mine" else 0}
        real_release = sg.dxgi_budget.release
        sg.dxgi_budget.release = lambda a: released.append(a["n"])
        try:
            self.assertEqual(sg._budget_reader("0x0000a1b2")(), {"Budget": 1.0})
            self.assertEqual(released, ["big"])
        finally:
            sg.dxgi_budget.release = real_release

    def test_luid_of_the_busiest_adapter(self):
        self.assertEqual(sg._luid_of({"pid_1_luid_0x00000000_0x0000A1B2_phys_0": (9, 0),
                                      "pid_1_luid_0x00000000_0x0000C3D4_phys_0": (0, 1)}), "0x0000A1B2")
        self.assertIsNone(sg._luid_of({}))

    def test_bounded_gives_up_on_a_hung_call(self):
        import threading
        hold = threading.Event()
        self.assertIsNone(sg._bounded(lambda: hold.wait(5), 0.2))
        hold.set()
        self.assertEqual(sg._bounded(lambda: 7, 1.0), 7)

    def test_struct_layouts(self):  # x64 sizes from the DXGI headers; pure ctypes, no DXGI call
        import ctypes, struct
        if struct.calcsize("P") == 8:
            self.assertEqual(ctypes.sizeof(sg.dxgi_budget.ADAPTER_DESC1), 312)
        self.assertEqual(ctypes.sizeof(sg.dxgi_budget.VIDEO_MEMORY_INFO), 32)
        self.assertEqual(sg.dxgi_budget.IID_IDXGIAdapter3.d1, 0x645967A4)

    def test_never_raises(self):
        def boom(*a): raise OSError("dxgi")
        sg.dxgi_budget.adapters = boom
        self.assertIsNone(sg._budget_reader()())  # no adapter list
        sg.dxgi_budget.adapters = lambda: [{"vendor": 0x1002, "dedicated_gb": 15.9}]
        sg.dxgi_budget.query = boom
        self.assertIsNone(sg._budget_reader()())  # a failing query
        sg.dxgi_budget.adapters = lambda: [{"vendor": 0x8086, "dedicated_gb": 0.1}]
        self.assertIsNone(sg._budget_reader()())  # no AMD adapter


if __name__ == "__main__":
    unittest.main()
