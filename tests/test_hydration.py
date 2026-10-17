"""Tests for the viewer's background-hydration worker thread."""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from trendify.store.record_store import RecordStore
from trendify.viewer.hydration import HydrationRunner, RunnerClosed


def _make_db(tmp_path: Path) -> Path:
    # `HydrationRunner` opens its store readonly (`mode=ro`), which requires the file to
    # already exist -- unlike a normal RecordStore.open(), readonly mode can't create it.
    db_path = tmp_path / "trendify.db"
    RecordStore.open(db_path).close()
    return db_path


class TestHydrationRunner:
    def test_run_executes_on_a_different_thread(self, tmp_path: Path) -> None:
        runner = HydrationRunner(_make_db(tmp_path))
        try:
            worker_thread_id = asyncio.run(
                runner.run(lambda store: threading.get_ident())
            )
            assert worker_thread_id != threading.get_ident()
        finally:
            runner.close()

    def test_reuses_the_same_thread_and_store_across_calls(
        self, tmp_path: Path
    ) -> None:
        runner = HydrationRunner(_make_db(tmp_path))
        try:

            async def scenario() -> tuple[int, int]:
                first = await runner.run(lambda store: threading.get_ident())
                second = await runner.run(lambda store: threading.get_ident())
                return first, second

            first, second = asyncio.run(scenario())
            assert first == second
        finally:
            runner.close()

    def test_does_not_block_the_calling_event_loop(self, tmp_path: Path) -> None:
        # If `run()` blocked the event loop, `other_task` (a short sleep) couldn't make
        # progress while the (longer) hydration call is in flight -- it would only start
        # after `run()` already returned, and `other_progressed` would still be False here.
        runner = HydrationRunner(_make_db(tmp_path))
        try:

            async def scenario() -> bool:
                other_progressed = False

                async def other_task() -> None:
                    nonlocal other_progressed
                    await asyncio.sleep(0.05)
                    other_progressed = True

                task = asyncio.create_task(other_task())
                await runner.run(lambda store: time.sleep(0.2))
                await task
                return other_progressed

            assert asyncio.run(scenario()) is True
        finally:
            runner.close()

    def test_close_can_be_called_more_than_once(self, tmp_path: Path) -> None:
        # A host embedding the viewer may close it explicitly on top of the app's own
        # lifespan shutdown, so a second close must be a no-op rather than an error.
        runner = HydrationRunner(_make_db(tmp_path))
        asyncio.run(runner.run(lambda store: None))

        runner.close()
        runner.close()

    def test_worker_is_a_daemon_so_exit_never_waits_on_queued_work(
        self, tmp_path: Path
    ) -> None:
        runner = HydrationRunner(_make_db(tmp_path))
        try:
            worker = asyncio.run(runner.run(lambda store: threading.current_thread()))

            assert worker.daemon is True
            assert worker.name == "trendify-store"
        finally:
            runner.close()

    def test_close_does_not_wait_for_the_running_job(self, tmp_path: Path) -> None:
        runner = HydrationRunner(_make_db(tmp_path))

        async def scenario() -> float:
            started = threading.Event()

            def slow(store: RecordStore) -> None:
                started.set()
                time.sleep(1.0)

            running = asyncio.create_task(runner.run(slow))
            await asyncio.get_running_loop().run_in_executor(None, started.wait)
            begin = time.perf_counter()
            runner.close()
            elapsed = time.perf_counter() - begin
            # the caller waiting on the running job is released at once, with a clear error,
            # even though the job itself is still going on its thread
            with pytest.raises(RunnerClosed):
                await asyncio.wait_for(running, timeout=0.3)
            return elapsed

        assert asyncio.run(scenario()) < 0.3

    def test_close_drops_jobs_that_have_not_started(self, tmp_path: Path) -> None:
        runner = HydrationRunner(_make_db(tmp_path))
        executed: list[int] = []

        async def scenario() -> list[object]:
            started = threading.Event()

            def slow(store: RecordStore) -> None:
                started.set()
                time.sleep(0.3)

            first = asyncio.create_task(runner.run(slow))
            await asyncio.get_running_loop().run_in_executor(None, started.wait)
            queued = [
                asyncio.create_task(runner.run(lambda store, i=i: executed.append(i)))
                for i in range(5)
            ]
            await asyncio.sleep(0.05)  # all five are queued behind the running job
            runner.close(wait=True)
            return await asyncio.gather(first, *queued, return_exceptions=True)

        results = asyncio.run(scenario())

        assert executed == []
        assert all(isinstance(r, RunnerClosed) for r in results)  # running job included

    def test_abandoned_request_is_skipped_before_it_runs(self, tmp_path: Path) -> None:
        # A client that disconnects cancels its awaiting task; its queued job must not still
        # burn CPU building a plot nobody will read.
        runner = HydrationRunner(_make_db(tmp_path))
        executed: list[str] = []
        try:

            async def scenario() -> None:
                started = threading.Event()

                def slow(store: RecordStore) -> None:
                    started.set()
                    time.sleep(0.3)

                first = asyncio.create_task(runner.run(slow))
                await asyncio.get_running_loop().run_in_executor(None, started.wait)
                abandoned = asyncio.create_task(
                    runner.run(lambda store: executed.append("abandoned"))
                )
                await asyncio.sleep(0.05)
                abandoned.cancel()
                await first
                await runner.run(lambda store: executed.append("after"))

            asyncio.run(scenario())

            assert executed == ["after"]
        finally:
            runner.close()

    def test_run_after_close_is_an_error(self, tmp_path: Path) -> None:
        runner = HydrationRunner(_make_db(tmp_path))
        runner.close()

        with pytest.raises(RunnerClosed, match="closed"):
            asyncio.run(runner.run(lambda store: None))
