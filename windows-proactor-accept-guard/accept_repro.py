"""Minimal repro: one transient accept() failure closes an asyncio server's listener on Windows.

No GPU, no ComfyUI, about two seconds. Pure Python (stdlib only):

  py -3.12 accept_repro.py     # exit 0 when stock asyncio dies and the guard survives (the expected result)

It starts a TCP server on the proactor loop, makes `IocpProactor.accept` raise WinError 10055 (WSAENOBUFS) exactly once,
then tries a client connection. The guard installs only on CPython 3.12; on another Python this runs the stock case only
and exits 0, so it shows whether that version still closes the listener.
"""
import asyncio, json, socket, sys
from pathlib import Path


async def scenario(guard):
    loop = asyncio.get_running_loop()
    proactor = loop._proactor
    orig, fired = proactor.accept, {"n": 0}

    def flaky(sock):
        if not fired["n"]:
            fired["n"] += 1
            raise OSError(10055, "An operation on a socket could not be performed because the system lacked "
                                 "sufficient buffer space or because a queue was full", None, 10055)
        return orig(sock)
    proactor.accept = flaky
    loop.set_exception_handler(lambda l, ctx: None)  # stock asyncio logs "Accept failed on a socket" here

    async def handle(reader, writer):
        writer.write(b"ok"); await writer.drain(); writer.close()
    lsock = socket.socket(); lsock.bind(("127.0.0.1", 0)); port = lsock.getsockname()[1]
    server = await asyncio.start_server(handle, sock=lsock)
    await asyncio.sleep(1.0 if guard else 0.2)  # let the failed accept happen (and, with the guard, its retry)
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 3)
        data = await asyncio.wait_for(r.read(2), 3); w.close()
        alive = data == b"ok"
    except (OSError, asyncio.TimeoutError):
        alive = False
    server.close()
    return {"python": sys.version.split()[0], "guard": guard, "injected_failures": fired["n"], "server_alive_after": alive}


def main():
    if sys.platform != "win32":
        print("SKIP: the proactor event loop is Windows-only"); return 0
    stock = asyncio.run(scenario(False))
    print(json.dumps(stock))
    if sys.version_info[:2] != (3, 12):
        print("stock only: lf_accept_guard installs on CPython 3.12 only;",
              "this Python closes the listener" if not stock["server_alive_after"] else "this Python keeps the listener")
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lf_accept_guard  # noqa: F401  (installs on import, on Windows with CPython 3.12)
    guarded = asyncio.run(scenario(True))
    print(json.dumps(guarded))
    ok = stock["server_alive_after"] is False and guarded["server_alive_after"] is True
    print("repro: stock asyncio closes the listener after one accept error; lf_accept_guard keeps it" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
