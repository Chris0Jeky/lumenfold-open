"""This process's WDDM video-memory budget per adapter, and the reservation it can ask for. Stdlib only (ctypes COM), no torch.

Windows decides how much of a GPU's VRAM each process may keep resident (its budget); allocations past it are demoted to
shared host memory, which is the spill lf_spill_guard trips on. The budget is per process and changes with what the OS
sees (measured: a fresh ComfyUI process filled 15.3 GB before spilling, the same process after a TRELLIS.2 job only
12.1 GB). DXGI reads it for the calling process (IDXGIAdapter3::QueryVideoMemoryInfo) and lets the process set a
reservation the OS keeps resident (SetVideoMemoryReservation). Neither creates a device or touches the GPU's queues.

    for a in adapters(): print(a["name"], a["luid"], query(a))   # Budget, CurrentUsage, AvailableForReservation, ...
    reserve(adapter, 12 << 30)                                     # ask the OS to keep 12 GiB resident for this process

    python dxgi_budget.py     # one JSON line per hardware adapter: this Python process's budget, in GiB
"""
import ctypes
from ctypes import POINTER, byref, c_uint, c_uint64, c_void_p, c_wchar, c_size_t, c_long, c_ulong

HRESULT = c_long  # not ctypes.HRESULT, which raises on every failure code (EnumAdapters1 ends the list with one)

LOCAL, NON_LOCAL = 0, 1  # DXGI_MEMORY_SEGMENT_GROUP


class GUID(ctypes.Structure):
    _fields_ = [("d1", c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

    def __init__(self, s):
        import uuid
        u = uuid.UUID(s)
        super().__init__(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))


IID_IDXGIFactory1 = GUID("770aae78-f26f-4dba-a829-253c83d1b387")
IID_IDXGIAdapter3 = GUID("645967a4-1392-4310-a798-8053ce3e93fd")


class ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", c_wchar * 128), ("VendorId", c_uint), ("DeviceId", c_uint), ("SubSysId", c_uint),
                ("Revision", c_uint), ("DedicatedVideoMemory", c_size_t), ("DedicatedSystemMemory", c_size_t),
                ("SharedSystemMemory", c_size_t), ("LuidLow", c_ulong), ("LuidHigh", c_long), ("Flags", c_uint)]


class VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [("Budget", c_uint64), ("CurrentUsage", c_uint64), ("AvailableForReservation", c_uint64),
                ("CurrentReservation", c_uint64)]


def _method(obj, index, restype, *argtypes):
    """COM vtable slot `index` of interface pointer `obj` as a callable taking (obj, *args)."""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[index])


def _check(hr, what):
    if hr < 0:
        raise OSError(f"{what} failed: HRESULT 0x{hr & 0xFFFFFFFF:08X}")


# vtable slots: IUnknown 0-2, IDXGIObject 3-6, IDXGIFactory 7-11 (EnumAdapters 7), IDXGIFactory1 12 EnumAdapters1;
# IDXGIAdapter 7-9, IDXGIAdapter1 10 GetDesc1, IDXGIAdapter2 11, IDXGIAdapter3 12-13, 14 QueryVideoMemoryInfo,
# 15 SetVideoMemoryReservation.
_RELEASE, _QI, _ENUM_ADAPTERS1, _GET_DESC1, _QUERY, _RESERVE = 2, 0, 12, 10, 14, 15
DXGI_ERROR_NOT_FOUND = -2005270526  # 0x887A0002


def adapters():
    """Hardware adapters as dicts: name, luid ("0x%08X" of the low part, as the PDH counters print it), dedicated GB, and
    the IDXGIAdapter3 pointer (keep the dict alive while you use it)."""
    factory = c_void_p()
    _check(ctypes.windll.dxgi.CreateDXGIFactory1(byref(IID_IDXGIFactory1), byref(factory)), "CreateDXGIFactory1")
    out, i = [], 0
    try:
        while True:
            a1 = c_void_p()
            hr = _method(factory, _ENUM_ADAPTERS1, HRESULT, c_uint, POINTER(c_void_p))(factory, i, byref(a1))
            if hr == DXGI_ERROR_NOT_FOUND:
                break
            _check(hr, "EnumAdapters1")
            desc, a3 = ADAPTER_DESC1(), c_void_p()
            try:
                _check(_method(a1, _GET_DESC1, HRESULT, POINTER(ADAPTER_DESC1))(a1, byref(desc)), "GetDesc1")
                hr = _method(a1, _QI, HRESULT, POINTER(GUID), POINTER(c_void_p))(a1, byref(IID_IDXGIAdapter3), byref(a3))
            finally:
                _method(a1, _RELEASE, c_ulong)(a1)
            i += 1
            if hr < 0 or desc.Flags & 2:  # no IDXGIAdapter3, or DXGI_ADAPTER_FLAG_SOFTWARE
                if hr >= 0 and a3.value: _method(a3, _RELEASE, c_ulong)(a3)
                continue
            out.append({"name": desc.Description, "luid": f"0x{desc.LuidLow:08X}", "vendor": desc.VendorId,
                        "dedicated_gb": round(desc.DedicatedVideoMemory / 2**30, 2), "_ptr": a3})
    finally:
        _method(factory, _RELEASE, c_ulong)(factory)
    return out


def release(adapter):
    """Drop the IDXGIAdapter3 reference adapters() took (once; later calls do nothing)."""
    p = adapter.pop("_ptr", None)
    if p is not None and p.value:
        _method(p, _RELEASE, c_ulong)(p)


def query(adapter, group=LOCAL):
    """This process's budget on `adapter`, in bytes: Budget, CurrentUsage, AvailableForReservation, CurrentReservation."""
    info = VIDEO_MEMORY_INFO()
    p = adapter["_ptr"]
    _check(_method(p, _QUERY, HRESULT, c_uint, c_uint, POINTER(VIDEO_MEMORY_INFO))(p, 0, group, byref(info)),
           "QueryVideoMemoryInfo")
    return {f: getattr(info, f) for f, _ in VIDEO_MEMORY_INFO._fields_}


def reserve(adapter, nbytes, group=LOCAL):
    """Ask the OS to keep `nbytes` of this process's memory resident on `adapter` (at most AvailableForReservation)."""
    p = adapter["_ptr"]
    _check(_method(p, _RESERVE, HRESULT, c_uint, c_uint, c_uint64)(p, 0, group, nbytes), "SetVideoMemoryReservation")


def gib(info):
    return {k: round(v / 2**30, 2) for k, v in info.items()}


if __name__ == "__main__":  # one read of this (Python) process's budget per adapter; no device is created
    import json
    for a in adapters():
        print(json.dumps({"name": a["name"], "luid": a["luid"], "dedicated_gb": a["dedicated_gb"], **gib(query(a))}))
