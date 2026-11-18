"""lf_spill_guard's OOM drop: free models instead of offloading them to host RAM after an OOM or a guard trip (7 Nov 2026).

After the guard turned a spill into a HIP OOM, ComfyUI's OOM handler ("Got an OOM, unloading all loaded models",
execution.py) ran `unload_all_models()`, which unpatches every model to its offload device: the CPU. The ComfyUI
process's working set went 2.7 -> 7.7 GB in 2 s and host free hit 1.9 GB. Host RAM is the resource that takes the box
down; a reload from disk costs seconds.

So, when armed (an OOM seen by execution.py's handler, or a spill-guard trip):
  * the OOM handler's own `unload_all_models()` does nothing (no host copy mid-prompt); unload_models and free_memory
    are flagged instead;
  * the main loop's `unload_all_models()` (main.py's prompt worker, while it processes those flags) moves each loaded
    model's weights to the `meta` device, forgets the model, and resets the executor cache itself (`e.reset()`, taken
    from the worker's frame) in the same step, so no cached model object with dropped weights can reach another prompt,
    whatever flags the worker read. Models reload from disk when next needed.
Unloads while not armed keep upstream's behaviour: a client's POST /free (its flags reach main.py's unload too), and
DISABLE_SMART_MEMORY's end-of-prompt unload in execution.py.
"""
import gc, logging, sys, threading

log = logging.getLogger("lf_spill_guard")
_armed = threading.Event()
_state = {"dropped": 0}
MAIN_FILE = lambda: str(getattr(sys.modules.get("__main__"), "__file__", "") or "")  # tests override this


def _caller_file(depth=2):
    f = sys._getframe(depth)
    return (f.f_code.co_filename or "").replace("\\", "/").lower()


def arm(queue=None):
    """Make the next main-loop unload a drop. Sets unload_models and free_memory so main.py resets the cache with it."""
    if queue is None:
        _armed.set(); return
    lock = getattr(queue, "mutex", None)
    if lock is None:
        queue.set_flag("free_memory", True); queue.set_flag("unload_models", True); _armed.set(); return
    with lock:  # the worker's get_flags takes this mutex: it sees both flags and the arm, or none of them
        queue.set_flag("free_memory", True); queue.set_flag("unload_models", True); _armed.set()


def drop_all(mm):
    """Free every loaded model's weights without a host copy; returns how many models were dropped."""
    n = 0
    for i in range(len(mm.current_loaded_models) - 1, -1, -1):
        lm = mm.current_loaded_models.pop(i)
        patcher = getattr(lm, "model", None)
        try:
            if patcher is not None and getattr(patcher, "model", None) is not None:
                if hasattr(patcher, "unpin_all_weights"):
                    try: patcher.unpin_all_weights()
                    except Exception: pass
                patcher.model.to("meta")  # storage released; the caller resets the executor cache in the same step
                module = patcher.model  # these live on the nn.Module (model_patcher.py sets them there)
                for attr, val in (("model_loaded_weight_memory", 0), ("model_lowvram", False), ("current_weight_patches_uuid", None),
                                  ("model_offload_buffer_memory", 0), ("lowvram_patch_counter", 0)):
                    if hasattr(module, attr): setattr(module, attr, val)
                if isinstance(getattr(patcher, "backup", None), dict): patcher.backup.clear()
                n += 1
        except Exception as e:  # a model that cannot move to meta is left to the garbage collector
            log.warning(f"lf_spill_guard: drop of {type(patcher).__name__} failed: {type(e).__name__}: {e}")
    gc.collect()
    try: mm.soft_empty_cache()
    except Exception: pass
    return n


def install(mm, queue_fn):
    """Wrap `mm.is_oom` and `mm.unload_all_models` (idempotent). `queue_fn()` returns the prompt queue."""
    if getattr(mm.unload_all_models, "lf_oom_drop", False):
        return "already installed"
    if not MAIN_FILE().replace("\\", "/").lower().endswith("/main.py"):
        return f"OOM drop NOT armed: the launcher is {MAIN_FILE()!r}, not ComfyUI's main.py"
    orig_is_oom, orig_unload = mm.is_oom, mm.unload_all_models

    def is_oom(e):
        r = orig_is_oom(e)
        if r and _caller_file().endswith("/execution.py"):  # ComfyUI's per-node exception handler
            try: arm(queue_fn())
            except Exception as ex: log.warning(f"lf_spill_guard: could not arm the OOM drop: {ex}")
        return r

    def unload_all_models():
        caller = _caller_file()
        if _armed.is_set() and caller.endswith("/execution.py"):
            log.warning("lf_spill_guard: OOM: models will be dropped after this prompt (no offload to host RAM)")
            return None
        if _armed.is_set() and caller.endswith("/main.py"):
            executor = sys._getframe(1).f_locals.get("e")  # main.py's prompt worker holds its PromptExecutor as `e`
            if executor is None or not hasattr(executor, "reset"):
                log.warning("lf_spill_guard: no executor in main.py's frame; offloading as upstream does")
                _armed.clear(); return orig_unload()
            _armed.clear()
            n = drop_all(mm); _state["dropped"] += n
            executor.reset()  # in the same step as the drop: no cached model with meta weights survives
            log.warning(f"lf_spill_guard: dropped {n} model(s) instead of offloading them to host RAM")
            print(f"[lf_spill_guard] dropped {n} model(s) instead of offloading them to host RAM", flush=True)
            return None
        return orig_unload()

    is_oom.lf_oom_drop = unload_all_models.lf_oom_drop = True
    mm.is_oom, mm.unload_all_models = is_oom, unload_all_models
    return "OOM drop armed on ComfyUI's OOM handler and guard trips"
