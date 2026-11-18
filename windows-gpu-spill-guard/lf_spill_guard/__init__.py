"""lf_spill_guard: stops a prompt when this process's shared GPU memory (host RAM backing VRAM) passes
LF_SPILL_GUARD_GB (default 1.0); see spill_guard.py. Adds no nodes. Install: copy this folder to
`ComfyUI/custom_nodes/lf_spill_guard/` and restart with an empty queue; the log shows "[lf_spill_guard] installed".
Undo: delete the folder and restart.
"""
try:
    from .spill_guard import install
    print(f"[lf_spill_guard] installed: {install()}", flush=True)
except Exception as e:  # never stop ComfyUI from starting; the line says the guard is off
    print(f"[lf_spill_guard] NOT installed (no spill protection): {type(e).__name__}: {e}", flush=True)

NODE_CLASS_MAPPINGS = {}
