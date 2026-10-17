"""
FastAPI app factory for the trendify dashboard. See `trendify.cli`'s `serve` command for how
this gets launched.
"""

from __future__ import annotations

import datetime
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trendify.__about__ import __version__
from trendify.store.record_store import RecordStore
from trendify.viewer.hydration import HydrationRunner
from trendify.viewer.routes import api, pages

if TYPE_CHECKING:
    import socket

    import uvicorn

__all__ = ["close_runners", "create_app", "create_app_from_env", "make_server"]

_DB_PATH_ENV_VAR = "TRENDIFY_DB_PATH"

logger = logging.getLogger(__name__)

_VIEWER_DIR = Path(__file__).parent
_TEMPLATES_DIR = _VIEWER_DIR / "templates"
_STATIC_DIR = _TEMPLATES_DIR / "static"


def create_app(
    db_path: Path | str,
    *,
    show_db_path: bool = True,
    cors_origins: Sequence[str] = (),
) -> FastAPI:
    """
    Builds the dashboard FastAPI app for `db_path`.

    The app runs on its own or mounted inside another FastAPI/Starlette app, either at a path
    prefix (`host.mount("/viz", create_app(db_path))`) or on its own hostname, and needs nothing
    from the host: no lifespan, no shared state. Pages and scripts build their URLs from the
    prefix the app is mounted under, so they work wherever it is served.

    All database and plot-building work runs on two dedicated worker threads
    (`app.state.runner` for real requests and `app.state.hydration_runner` for background
    prefetch, see `routes.api`), each with its own read-only `RecordStore` connection opened on
    first use. A slow request therefore never blocks the event loop, which matters when that loop
    is shared with a host site's other endpoints, and a prefetch never queues ahead of a click.
    A SQLite connection belongs to the thread that opens it, which is why each one lives
    entirely on its worker thread. The lifespan only closes those connections at shutdown; a
    mounted app doesn't get one, and its connections then close with the interpreter.

    This app is meant to run as a single process (each process would keep its own caches).

    Args:
        db_path (Path | str): path to the `.db` file to serve. Must already exist.
        show_db_path (bool): show the server-side path of `db_path` in the page header. Turn
            off when the viewer is exposed to people who shouldn't see the server's file layout.
        cors_origins (Sequence[str]): origins whose browser JavaScript may call this app's API
            from another origin (an exact list; no wildcard is added for you). Empty by
            default, which adds no CORS headers. Not needed to show the viewer in an `<iframe>`:
            the page then calls its own origin. Only needed if the host page's own scripts fetch
            from the viewer's API.

    Returns:
        (FastAPI): the configured app, ready for `uvicorn.run`, `host.mount`, or a `TestClient`.

    """
    db_path = Path(db_path)
    RecordStore.open(
        db_path, readonly=True
    ).close()  # fail now, not on the first request

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            close_runners(app)

    app = FastAPI(title="trendify", version=__version__, lifespan=lifespan)
    app.state.runner = HydrationRunner(db_path)
    app.state.hydration_runner = HydrationRunner(db_path)
    app.state.response_cache = OrderedDict()
    app.state.db_path = db_path
    app.state.db_mtime = None

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET"],
            allow_headers=["x-trendify-hydrate"],
        )

    templates = Jinja2Templates(directory=_TEMPLATES_DIR)
    # Jinja's `Environment.globals` is inferred from its default namespace (range, dict, ...),
    # so it isn't typed as the open `dict[str, Any]` it behaves as.
    env_globals = cast("dict[str, Any]", templates.env.globals)
    env_globals["current_year"] = datetime.datetime.now().year
    env_globals["app_version"] = __version__
    env_globals["db_path"] = str(db_path) if show_db_path else ""
    app.state.templates = templates

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    app.include_router(pages.router)
    app.include_router(api.router, prefix="/api")

    return app


def create_app_from_env() -> FastAPI:
    """
    `create_app`, reading `db_path` from the `TRENDIFY_DB_PATH` environment variable instead of
    a direct argument.

    This exists solely as a uvicorn ASGI factory target (`uvicorn.run(..., factory=True)`) for
    `trendify serve --reload`: uvicorn's reload mode re-imports the app in a fresh subprocess on
    every file change, so it needs an importable `module:attribute` string rather than a live app
    instance; there is no other way to hand it a runtime value like `db_path` across that
    re-import boundary.
    """
    db_path = os.environ[_DB_PATH_ENV_VAR]
    return create_app(Path(db_path))


def close_runners(app: FastAPI) -> None:
    """
    Stops a viewer app's worker threads and releases every request still waiting on them (those
    answer 503), without waiting for a plot that is mid-build. The app calls this itself at
    shutdown; a server that wants to stop quickly calls it as soon as shutdown begins (see
    `make_server`). Safe to call more than once.

    Args:
        app (FastAPI): an app made by `create_app`.

    """
    app.state.runner.close()
    app.state.hydration_runner.close()


def make_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    log_level: str = "critical",
    viewers: Sequence[FastAPI] | None = None,
) -> uvicorn.Server:
    """
    Builds a uvicorn server for `app` that stops promptly and quietly on Ctrl+C.

    The page prefetches every tag in the background, so when Ctrl+C arrives there can be many
    requests in flight, and plain uvicorn waits for all of them before it exits (or, after a
    timeout, cancels them with a traceback each). This server releases the viewer's waiting
    requests the moment shutdown begins, so they finish at once with a 503.

    Args:
        app (FastAPI): the app to serve.
        host (str): interface to bind.
        port (int): port to bind.
        log_level (str): uvicorn log level.
        viewers (Sequence[FastAPI] | None): viewer apps (from `create_app`) to release at
            shutdown. Defaults to `app` itself; pass the mounted viewers when `app` is a host
            site that mounts them.

    Returns:
        (uvicorn.Server): call `.run()` on it.

    """
    import uvicorn

    targets = (app,) if viewers is None else tuple(viewers)

    class ViewerServer(uvicorn.Server):
        async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
            for viewer in targets:
                close_runners(viewer)
            await super().shutdown(sockets)

        def run(self, sockets: list[socket.socket] | None = None) -> None:
            try:
                super().run(sockets)
            except KeyboardInterrupt:
                pass  # uvicorn re-raises Ctrl+C once it has shut down cleanly; that's the end

    return ViewerServer(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level,
            timeout_graceful_shutdown=3,  # a safety net; requests are already released
        )
    )
