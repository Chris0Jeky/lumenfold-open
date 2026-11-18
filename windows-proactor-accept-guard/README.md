# Windows proactor accept guard

## Problem

An asyncio HTTP server on Windows (here ComfyUI's aiohttp server) stops accepting connections for good while its
process keeps running. Every new connection is refused; work already queued carries on. The server log shows
`Accept failed on a socket` with `OSError: [WinError 10055] ... the system lacked sufficient buffer space`.

Cause: the proactor event loop's accept loop, `BaseProactorEventLoop._start_serving` in `Lib/asyncio/proactor_events.py`,
closes the listening socket on any `OSError` from `accept()`. WinError 10055 (WSAENOBUFS) is transient: it appears for a
moment when the machine is short of memory. One such error ends the server.

## Who this concerns

- Windows, any CPython whose asyncio uses `ProactorEventLoop` (the default event loop on Windows).
- Any asyncio server: aiohttp, ComfyUI, and others. No GPU is involved; we hit it while a large GPU job squeezed host
  memory.
- Confirmed in CPython 3.12.10 (source and repro) and 3.14.3 (source still closes the socket; the stock repro case
  dies). Not fixed upstream as of 3.14.3; we did not check 3.13 or later 3.14 releases.
- Upstream: our comment on the CPython issue,
  https://github.com/python/cpython/issues/156365#issuecomment-6671393542

## What we changed

`lf_accept_guard/` is a ComfyUI custom node that adds no nodes. On import it replaces
`BaseProactorEventLoop._start_serving` with a modified copy. In the `OSError` branch the copy keeps the socket open,
logs a warning and retries `accept()` after a backoff (0.25 s, doubling to 10 s, reset on the next good accept). The
rest of the function is unchanged.

`proactor_events-3.12.10.diff` is the change as a unified diff against CPython v3.12.10
`Lib/asyncio/proactor_events.py`. The function in `lf_accept_guard/__init__.py` is that file's `_start_serving` with the
diff applied, character for character.

The node installs only on Windows with CPython 3.12, the version the copy was taken from.

## Result

`accept_repro.py` injects one WinError 10055 into `IocpProactor.accept`, then opens a client connection. Run on
18 November 2026:

| Python | Guard | Injected failures | Server alive after |
| --- | --- | ---: | --- |
| 3.12.10 | no | 1 | false |
| 3.12.10 | yes | 1 | true |
| 3.14.3 | no | 1 | false |

## How to run

Prerequisites: Windows and CPython 3.12 (for the guarded case). The repro is stdlib-only; no GPU, no ComfyUI, about
two seconds.

```
py -3.12 accept_repro.py
```

Expected output, exit code 0:

```
{"python": "3.12.10", "guard": false, "injected_failures": 1, "server_alive_after": false}
[lf_accept_guard] installed: transient accept errors no longer close the HTTP listener
[lf_accept_guard] accept failed ([WinError 10055] ...); retry 1 in 0.25 s, listener kept open
{"python": "3.12.10", "guard": true, "injected_failures": 1, "server_alive_after": true}
repro: stock asyncio closes the listener after one accept error; lf_accept_guard keeps it
```

On another Python version it runs the stock case only and says whether that version closes the listener.

Install in ComfyUI: copy `lf_accept_guard/` to `<ComfyUI root>/custom_nodes/` and restart. The log shows
`[lf_accept_guard] installed`. Undo: delete the folder and restart.

## Measured on

- Repro: Windows 11, CPython 3.12.10 (python.org build) and CPython 3.14.3, i5-13600K. No GPU used.
- Field failure: ComfyUI 0.43.2 on its embedded CPython 3.12.10, Windows 11, Radeon RX 9070 XT (gfx1201), Adrenalin
  32.0.31041.1004, torch 2.9.1+rocm7.2.1. Full stack: [../docs/windows-rocm-stack.md](../docs/windows-rocm-stack.md).

## Not verified

- CPython 3.12 patch releases other than 3.12.10, and 3.13: the guard installs on any 3.12.x but was copied from
  3.12.10 only.
- Real memory pressure: the repro injects the error; it does not exhaust buffer space.
- A permanent accept error: the guard retries every 10 s for as long as the socket stays open and never closes it.
  It also no longer calls the loop's exception handler for these errors; it logs a warning instead.
- SSL listeners: the code path is unchanged, but the repro uses plain TCP.

## Licence

`lf_accept_guard/__init__.py` contains a modified copy of a CPython function, used under the Python Software Foundation
License Version 2 (`lf_accept_guard/LICENSE-PSF.txt`); the copyright notice and summary of changes are at the top of the
file. The diff is under the same licence. The repro and the rest of the node are MIT.
