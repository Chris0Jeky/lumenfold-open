"""lf_qem_hip: an AMD-native edge_collapse_checks for ComfyUI's QEM decimation. Adds no nodes.

On ROCm, ComfyUI's DecimateMesh always took the torch fallback for its collapse checks (comfy-kitchen ships that kernel
only for CUDA). Install: copy this folder to `<ComfyUI root>/custom_nodes/lf_qem_hip/` and restart ComfyUI with an
empty queue; the log shows "[lf_qem_hip] installed". Undo: delete the folder and restart. edges_lowpeak.py: QEM's sorted half-edge table
without its (3F, 2) temporaries ("[lf_qem_hip] edges installed").
"""
try:
    from .qem_hip import install
    result = install()
    print(f"[lf_qem_hip] installed: HIP edge_collapse_checks, self-test vs the torch fallback "
          f"(flip mismatches, max skinny diff, link mismatches) = {result}", flush=True)
except Exception as e:  # never stop ComfyUI from starting: without the patch DecimateMesh runs upstream's fallback
    print(f"[lf_qem_hip] not installed: {type(e).__name__}: {e}", flush=True)

try:
    from .edges_lowpeak import install as _install_edges
    print(f"[lf_qem_hip] edges installed: {_install_edges()}", flush=True)
except Exception as e:  # without it DecimateMesh runs upstream's half-edge sort
    print(f"[lf_qem_hip] edges not installed: {type(e).__name__}: {e}", flush=True)

NODE_CLASS_MAPPINGS = {}
