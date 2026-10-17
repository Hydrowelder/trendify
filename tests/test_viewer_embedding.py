"""Tests for embedding the viewer in another FastAPI app: mounting, URL prefixes, limits, CORS."""

import asyncio
import json
import re
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from urllib.parse import quote

import httpx2
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Host

from trendify.formats.table import TableEntry
from trendify.plotting.point import Point2D
from trendify.plotting.trace import Trace2D
from trendify.store.record_store import RecordStore
from trendify.viewer.app import close_runners, create_app, make_server
from trendify.viewer.routes import api, pages

TAG = quote(json.dumps("scatter"))


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "trendify.db"
    with RecordStore.open(path) as store:
        store.write_run(
            tmp_path / "run1",
            [
                Point2D(tags=["scatter"], x=1.0, y=2.0),
                Trace2D(tags=[("group", "trace")], x=[0, 1], y=[0, 1]),
                TableEntry(tags=["table"], row="r1", col="c1", value=1.0),
            ],
        )
    return path


@pytest.fixture
def make_viewer(db_path: Path) -> Iterator[Callable[..., FastAPI]]:
    """Builds viewer apps and closes their worker threads afterward."""
    apps: list[FastAPI] = []

    def make(*, show_db_path: bool = True, cors_origins: Sequence[str] = ()) -> FastAPI:
        app = create_app(db_path, show_db_path=show_db_path, cors_origins=cors_origins)
        apps.append(app)
        return app

    yield make
    for app in apps:
        app.state.runner.close()
        app.state.hydration_runner.close()


def _root_absolute_urls(html: str) -> list[str]:
    return sorted(set(re.findall(r'(?:src|href)="(/[^/"][^"]*)"', html)))


def _base_path(html: str) -> str | None:
    match = re.search(r'<meta name="trendify-base"\s+content="([^"]*)"', html)
    return match.group(1) if match else None


class TestMounting:
    def test_works_without_the_host_running_any_lifespan(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        host = FastAPI()
        host.mount("/viz", make_viewer())

        with TestClient(host) as client:
            assert client.get("/viz/").status_code == 200
            assert client.get("/viz/api/tags").status_code == 200
            assert client.get(f"/viz/api/plot?tag={TAG}").status_code == 200

    def test_page_urls_follow_the_mount_prefix(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        host = FastAPI()
        host.mount("/viz", make_viewer())

        with TestClient(host) as client:
            html = client.get("/viz/").text
            urls = _root_absolute_urls(html)

            assert _base_path(html) == "/viz"
            assert urls
            assert all(url.startswith("/viz/static/") for url in urls)
            assert all(client.get(url).status_code == 200 for url in urls)

    def test_page_urls_are_unprefixed_at_the_root(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        with TestClient(make_viewer()) as client:
            html = client.get("/").text

            assert _base_path(html) == ""
            assert all(url.startswith("/static/") for url in _root_absolute_urls(html))

    def test_works_on_its_own_hostname(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        host = FastAPI()
        host.router.routes.append(Host("viewer.example.test", app=make_viewer()))

        with TestClient(host, base_url="http://viewer.example.test") as client:
            html = client.get("/").text

            assert client.get("/api/tags").status_code == 200
            assert all(
                client.get(u).status_code == 200 for u in _root_absolute_urls(html)
            )

    def test_script_bundle_reads_the_base_path_from_the_page(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        # Fails if the TypeScript was changed without rebuilding static/js/main.js.
        with TestClient(make_viewer()) as client:
            assert "trendify-base" in client.get("/static/js/main.js").text

    def test_accepts_a_plain_string_path(self, db_path: Path) -> None:
        app = create_app(str(db_path))
        try:
            with TestClient(app) as client:
                assert client.get("/api/ping").status_code == 200
        finally:
            app.state.runner.close()
            app.state.hydration_runner.close()

    def test_missing_database_fails_when_the_app_is_created(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(sqlite3.OperationalError):
            create_app(tmp_path / "missing.db")


class TestEventLoop:
    def test_a_slow_request_does_not_block_the_hosts_other_endpoints(
        self, make_viewer: Callable[..., FastAPI], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slow_tag_tree(store: RecordStore) -> list:
            time.sleep(0.4)
            return []

        monkeypatch.setattr(pages, "build_tag_tree", slow_tag_tree)
        host = FastAPI()
        host.mount("/viz", make_viewer())

        @host.get("/health")
        async def health() -> dict[str, bool]:
            return {"ok": True}

        async def scenario() -> tuple[int, float, int]:
            transport = httpx2.ASGITransport(app=host)
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://t"
            ) as c:
                slow = asyncio.create_task(c.get("/viz/"))
                await asyncio.sleep(0.05)  # let the slow request start
                latencies: list[float] = []
                while not slow.done():
                    start = time.perf_counter()
                    await c.get("/health")
                    latencies.append(time.perf_counter() - start)
                    await asyncio.sleep(0.01)
                return (await slow).status_code, max(latencies), len(latencies)

        status, worst, polls = asyncio.run(scenario())

        assert status == 200
        assert polls >= 5  # the loop stayed free for the whole 0.4 s
        assert worst < 0.15


class TestInputLimits:
    def test_malformed_tag_is_a_client_error(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        with TestClient(make_viewer()) as client:
            assert client.get("/api/plot?tag=notjson").status_code == 422
            assert client.get("/api/table?tag=notjson&view=melted").status_code == 422

    def test_max_points_has_an_upper_bound(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        with TestClient(make_viewer()) as client:
            limit = api._MAX_POINTS_LIMIT
            assert (
                client.get(f"/api/plot?tag={TAG}&max_points={limit}").status_code == 200
            )
            assert (
                client.get(f"/api/plot?tag={TAG}&max_points={limit + 1}").status_code
                == 422
            )

    def test_response_cache_is_bounded(
        self, make_viewer: Callable[..., FastAPI], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api, "_MAX_CACHED_RESPONSES", 3)
        app = make_viewer()

        with TestClient(app) as client:
            for max_points in range(1, 9):
                client.get(f"/api/plot?tag={TAG}&max_points={max_points}")

        assert len(app.state.response_cache) == 3


class TestShowDbPath:
    def test_path_is_shown_by_default(
        self, make_viewer: Callable[..., FastAPI], db_path: Path
    ) -> None:
        with TestClient(make_viewer()) as client:
            assert str(db_path) in client.get("/").text

    def test_path_can_be_hidden(
        self, make_viewer: Callable[..., FastAPI], db_path: Path
    ) -> None:
        with TestClient(make_viewer(show_db_path=False)) as client:
            assert str(db_path) not in client.get("/").text


class TestCors:
    HOST_ORIGIN = "https://host.example"

    def test_no_cors_headers_by_default(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        with TestClient(make_viewer()) as client:
            response = client.get("/api/ping", headers={"Origin": self.HOST_ORIGIN})

            assert "access-control-allow-origin" not in response.headers

    def test_listed_origin_is_allowed_and_others_are_not(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        with TestClient(make_viewer(cors_origins=[self.HOST_ORIGIN])) as client:
            allowed = client.get("/api/ping", headers={"Origin": self.HOST_ORIGIN})
            other = client.get("/api/ping", headers={"Origin": "https://evil.example"})

            assert allowed.headers["access-control-allow-origin"] == self.HOST_ORIGIN
            assert "access-control-allow-origin" not in other.headers

    def test_preflight_for_the_prefetch_header_is_answered(
        self, make_viewer: Callable[..., FastAPI]
    ) -> None:
        with TestClient(make_viewer(cors_origins=[self.HOST_ORIGIN])) as client:
            response = client.options(
                "/api/plot",
                headers={
                    "Origin": self.HOST_ORIGIN,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "x-trendify-hydrate",
                },
            )

            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == self.HOST_ORIGIN


def _slow_tag_tree(seconds: float) -> Callable[[RecordStore], list]:
    def slow(store: RecordStore) -> list:
        time.sleep(seconds)
        return []

    return slow


class TestShutdown:
    def test_requests_in_flight_get_503_when_the_runners_close(
        self, make_viewer: Callable[..., FastAPI], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pages, "build_tag_tree", _slow_tag_tree(3.0))
        app = make_viewer()

        async def scenario() -> tuple[int, float]:
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://t"
            ) as c:
                request = asyncio.create_task(c.get("/"))
                await asyncio.sleep(0.2)  # the slow build is running
                begin = time.perf_counter()
                close_runners(app)
                response = await request
                return response.status_code, time.perf_counter() - begin

        status, elapsed = asyncio.run(scenario())

        assert status == 503
        assert elapsed < 0.5  # not held up by the 3 s build

    def test_server_stops_promptly_and_quietly_with_requests_in_flight(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pages, "build_tag_tree", _slow_tag_tree(5.0))
        app = create_app(db_path)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = make_server(app, host="127.0.0.1", port=port)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.02)
        result: dict[str, int] = {}

        def request() -> None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10)
            except urllib.error.HTTPError as error:
                result["status"] = error.code

        client = threading.Thread(target=request)
        client.start()
        time.sleep(0.3)  # the slow build is running

        begin = time.perf_counter()
        server.should_exit = True  # what Ctrl+C does
        thread.join(timeout=3)
        client.join(timeout=3)

        assert not thread.is_alive()
        assert time.perf_counter() - begin < 2.5
        assert result == {"status": 503}

    def test_ctrl_c_after_a_clean_shutdown_is_not_an_error(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # uvicorn re-raises KeyboardInterrupt after shutting down; its own `uvicorn.run`
        # swallows that, so `make_server(...).run()` must too or Ctrl+C prints a traceback.
        def interrupted(self: uvicorn.Server, sockets: object = None) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(uvicorn.Server, "run", interrupted)
        app = create_app(db_path)
        try:
            make_server(app, host="127.0.0.1", port=0).run()
        finally:
            close_runners(app)
