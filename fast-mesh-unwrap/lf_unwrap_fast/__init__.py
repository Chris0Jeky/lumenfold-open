"""lf_unwrap_fast: UnwrapMesh's batched LSCM solved sparse, and MeshSmoothNormals' crease path vectorised
(normals_fast.py, bit-identical). Adds no nodes.

Install: copy this folder to `<ComfyUI root>/custom_nodes/lf_unwrap_fast/` and restart ComfyUI with an empty queue; the
log shows "[lf_unwrap_fast] sparse batched LSCM installed" and the normals_fast status line. Undo: delete the folder and
restart.
"""
try:
    from .unwrap_fast import install
    err = install()
    print(f"[lf_unwrap_fast] sparse batched LSCM installed, self-test max rel diff {err:.2e}", flush=True)
except Exception as e:  # never stop ComfyUI from starting: without the patch UnwrapMesh runs upstream's code
    print(f"[lf_unwrap_fast] not installed: {type(e).__name__}: {e}", flush=True)

try:
    from .batch_fast import install as _install_batch
    n = _install_batch()
    print(f"[lf_unwrap_fast] PEC clustering and boundary-loop helpers installed ({n} entry points), self-tests passed", flush=True)
except Exception as e:
    print(f"[lf_unwrap_fast] batch extension not installed: {type(e).__name__}: {e}", flush=True)

try:  # MeshSmoothNormals' crease path (~5 s per job in upstream's Python loops), bit-identical, vectorised
    from .normals_fast import install as _install_normals
    n = _install_normals()
    print(f"[lf_unwrap_fast] vectorised MeshSmoothNormals crease path installed in {n} module(s), self-test exact", flush=True)
except Exception as e:
    print(f"[lf_unwrap_fast] normals not installed: {type(e).__name__}: {e}", flush=True)

NODE_CLASS_MAPPINGS = {}
