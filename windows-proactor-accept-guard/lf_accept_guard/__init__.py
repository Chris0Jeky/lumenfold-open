# _start_serving below is a modified copy of BaseProactorEventLoop._start_serving from CPython 3.12.10,
# Lib/asyncio/proactor_events.py.
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
# 2018, 2019, 2020, 2021, 2022, 2023 Python Software Foundation; All Rights Reserved.
# Used under the Python Software Foundation License Version 2; the full text is in LICENSE-PSF.txt beside this file.
# Summary of changes: in the accept loop's OSError branch the socket is no longer closed; the error is logged and accept
# is retried after a backoff (0.25 s doubling to 10 s, reset on the next good accept). The diff against 3.12.10 is
# ../proactor_events-3.12.10.diff. The rest of this file is MIT.
"""lf_accept_guard: keep ComfyUI's HTTP server listening through transient socket errors on Windows. Adds no nodes.

On Windows the asyncio proactor event loop's accept loop closes the listening socket on any OSError from accept()
(CPython 3.12.10, `asyncio/proactor_events.py`, `BaseProactorEventLoop._start_serving`, "Accept failed on a socket").
Under a memory squeeze accept() can fail with WinError 10055 (WSAENOBUFS); the listener is then gone for good while the
process keeps running, and every new connection is refused.

This replaces that loop with the same code except the OSError branch: log, keep the socket open, retry after a backoff.
Installed only on Windows with CPython 3.12, the version the copy was taken from.
Install: copy this folder to `<ComfyUI root>/custom_nodes/`, restart; the log shows "[lf_accept_guard] installed".
Undo: delete the folder and restart. Repro: `../accept_repro.py`.
"""
import sys


def install():
    import asyncio.proactor_events as P
    from asyncio import exceptions
    from asyncio.log import logger
    if getattr(P.BaseProactorEventLoop._start_serving, "_lf_accept_guard", False):
        return False

    def _start_serving(self, protocol_factory, sock,
                       sslcontext=None, server=None, backlog=100,
                       ssl_handshake_timeout=None,
                       ssl_shutdown_timeout=None):
        state = {"delay": 0.25, "fails": 0}

        def loop(f=None):
            try:
                if f is not None:
                    conn, addr = f.result()
                    if self._debug:
                        logger.debug("%r got a new connection from %r: %r",
                                     server, addr, conn)
                    protocol = protocol_factory()
                    if sslcontext is not None:
                        self._make_ssl_transport(
                            conn, protocol, sslcontext, server_side=True,
                            extra={'peername': addr}, server=server,
                            ssl_handshake_timeout=ssl_handshake_timeout,
                            ssl_shutdown_timeout=ssl_shutdown_timeout)
                    else:
                        self._make_socket_transport(
                            conn, protocol,
                            extra={'peername': addr}, server=server)
                    state["delay"], state["fails"] = 0.25, 0
                if self.is_closed():
                    return
                f = self._proactor.accept(sock)
            except OSError as exc:
                if sock.fileno() != -1:
                    # A transient error (for example WSAENOBUFS) must not end
                    # the server: keep the socket open and retry with backoff.
                    state["fails"] += 1
                    logger.warning(
                        "[lf_accept_guard] accept failed (%s); retry %d in "
                        "%.2f s, listener kept open",
                        exc, state["fails"], state["delay"])
                    self.call_later(state["delay"], loop)
                    state["delay"] = min(state["delay"] * 2, 10.0)
                elif self._debug:
                    logger.debug("Accept failed on socket %r",
                                 sock, exc_info=True)
            except exceptions.CancelledError:
                sock.close()
            else:
                self._accept_futures[sock.fileno()] = f
                f.add_done_callback(loop)

        self.call_soon(loop)

    _start_serving._lf_accept_guard = True
    P.BaseProactorEventLoop._start_serving = _start_serving
    return True


if sys.platform == "win32" and sys.version_info[:2] == (3, 12):
    try:
        install()
        print("[lf_accept_guard] installed: transient accept errors no longer close the HTTP listener", flush=True)
    except Exception as e:  # never stop ComfyUI from starting
        print(f"[lf_accept_guard] not installed: {type(e).__name__}: {e}", flush=True)
else:
    print(f"[lf_accept_guard] not installed: needs Windows and CPython 3.12 (this is {sys.platform} {sys.version.split()[0]})",
          flush=True)

NODE_CLASS_MAPPINGS = {}
