"""Compile and launch HIP kernels from Python on Windows ROCm, with no build system: hiprtc + the HIP module API.

PyTorch's ROCm wheels for Windows ship `hiprtc0702.dll` and `amdhip64_7.dll` (in `_rocm_sdk_core/bin`) but no Triton and
no working torch C++ extension build, so this is the shortest path to a native kernel inside ComfyUI's Python.
Kernels run on PyTorch's current stream and take tensors by pointer.

    k = Kernel(SRC, "my_kernel")                  # compiled once per process for the device's arch, cached on disk
    k((blocks,), (256,), x, y, n)                 # tensors pass their data_ptr; ints as int32, floats as float32
    k((blocks,), (256,), x, y, I64(n))            # wrap to choose another C type (I64, U64, F64)

Compile only (no GPU needed): `python hipkernel.py --compile <file.hip> [--arch gfx1201]`.
"""
import ctypes, hashlib, os, sys
from pathlib import Path

CACHE = Path(os.environ.get("LF_HIPRTC_CACHE", Path(os.environ.get("LOCALAPPDATA", ".")) / "lf-hiprtc"))
_libs = {}


def _sdk_bin():
    import _rocm_sdk_core  # ships with torch's ROCm wheels
    return Path(_rocm_sdk_core.__file__).parent / "bin"


def _lib(name):
    if name not in _libs:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetModuleHandleW.restype = ctypes.c_void_p
        h = k32.GetModuleHandleW(name)  # reuse the copy torch already loaded: one HIP runtime per process
        _libs[name] = ctypes.WinDLL(name, handle=h) if h else ctypes.WinDLL(str(_sdk_bin() / name))
    return _libs[name]


def hiprtc():
    lib = _lib("hiprtc0702.dll")
    lib.hiprtcGetErrorString.restype = ctypes.c_char_p
    return lib


def hip():
    lib = _lib("amdhip64_7.dll")
    lib.hipGetErrorString.restype = ctypes.c_char_p
    return lib


def _ok_rtc(rc, what, prog=None):
    if rc != 0:
        msg = hiprtc().hiprtcGetErrorString(rc).decode()
        log = ""
        if prog is not None:
            n = ctypes.c_size_t(); hiprtc().hiprtcGetProgramLogSize(prog, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value + 1); hiprtc().hiprtcGetProgramLog(prog, buf); log = buf.value.decode(errors="replace")
        raise RuntimeError(f"{what}: {msg}\n{log}")


def _ok(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: {hip().hipGetErrorString(rc).decode()} ({rc})")


def compile(src, arch, options=()):
    """HIP C++ source -> code object bytes for `arch` (e.g. gfx1201). Cached by source, arch and options."""
    opts = [f"--offload-arch={arch}", "-O3", "-std=c++17", *options]
    key = hashlib.sha256((src + "\0" + "\0".join(opts)).encode()).hexdigest()[:24]
    path = CACHE / f"{key}-{arch}.co"
    if path.exists():
        return path.read_bytes()
    rtc = hiprtc(); prog = ctypes.c_void_p()
    _ok_rtc(rtc.hiprtcCreateProgram(ctypes.byref(prog), src.encode(), b"lf.hip", 0, None, None), "hiprtcCreateProgram")
    try:
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        _ok_rtc(rtc.hiprtcCompileProgram(prog, len(opts), arr), "hiprtcCompileProgram", prog)
        n = ctypes.c_size_t(); _ok_rtc(rtc.hiprtcGetCodeSize(prog, ctypes.byref(n)), "hiprtcGetCodeSize")
        buf = ctypes.create_string_buffer(n.value); _ok_rtc(rtc.hiprtcGetCode(prog, buf), "hiprtcGetCode")
    finally:
        rtc.hiprtcDestroyProgram(ctypes.byref(prog))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp"); tmp.write_bytes(buf.raw); os.replace(tmp, path)  # never a torn .co
    return buf.raw


class I32(ctypes.c_int32): pass
class I64(ctypes.c_int64): pass
class U64(ctypes.c_uint64): pass
class F32(ctypes.c_float): pass
class F64(ctypes.c_double): pass


def device_arch(device=0):
    import torch
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


class Kernel:
    """One `extern "C" __global__` function from `src`, loaded into the current device's context."""

    def __init__(self, src, name, options=()):
        import torch
        torch.cuda.init()  # the module must load into the context torch uses
        self.code = compile(src, device_arch(torch.cuda.current_device()), options)
        self._img = ctypes.create_string_buffer(self.code, len(self.code))
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        _ok(hip().hipModuleLoadData(ctypes.byref(mod), self._img), "hipModuleLoadData")
        _ok(hip().hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()), f"hipModuleGetFunction({name})")
        self.mod, self.fn, self.name = mod, fn, name

    def __call__(self, grid, block, *args, shared=0):
        import torch
        vals = []
        for a in args:
            if isinstance(a, torch.Tensor):
                if not a.is_cuda:
                    raise ValueError(f"{self.name}: tensor argument on {a.device}")
                vals.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, ctypes._SimpleCData):
                vals.append(a)
            elif isinstance(a, bool) or isinstance(a, int):
                vals.append(ctypes.c_int32(int(a)))
            elif isinstance(a, float):
                vals.append(ctypes.c_float(a))
            else:
                raise TypeError(f"{self.name}: unsupported argument {type(a)}")
        params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        g = tuple(grid) + (1,) * (3 - len(grid)); b = tuple(block) + (1,) * (3 - len(block))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _ok(hip().hipModuleLaunchKernel(self.fn, *[ctypes.c_uint(x) for x in g], *[ctypes.c_uint(x) for x in b],
                                        ctypes.c_uint(shared), stream, params, None), f"launch {self.name}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--compile":
        arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "gfx1201"
        code = compile(Path(sys.argv[2]).read_text(), arch)
        print(f"compiled {sys.argv[2]} for {arch}: {len(code)} bytes")
