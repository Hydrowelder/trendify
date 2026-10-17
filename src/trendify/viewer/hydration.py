"""
Runs viewer requests on a dedicated worker thread with its own read-only `RecordStore`
connection, so SQLite and plot-building work never block the event loop (which a host app
embedding the viewer shares with its own endpoints). The app keeps two of these: one for real
user clicks and one for background-hydration prefetch (see `routes.api`), so a slow prefetch can
never queue ahead of a click. Several independent readers are safe because each store is opened
read-only and this app's `.db` file is in WAL mode, which supports multiple concurrent readers
with no coordination needed.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any, TypeVar

from trendify.store.record_store import RecordStore

__all__ = ["HydrationRunner", "RunnerClosed"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

_Job = tuple["Future[Any]", Callable[[RecordStore], Any]]


class RunnerClosed(RuntimeError):
    """
    Raised to the caller of `HydrationRunner.run` when the runner is closed before its job
    finishes (including when it was queued or already running), or is closed already.
    """


def _settle(outcome: asyncio.Future[Any], done: Future[Any]) -> None:
    """Copies a finished job onto the awaiting task's future (runs on the event loop)."""
    if outcome.done():
        return
    if done.cancelled():
        outcome.set_exception(RunnerClosed("The runner closed before this job ran."))
    elif (error := done.exception()) is not None:
        outcome.set_exception(error)
    else:
        outcome.set_result(done.result())


def _release(outcome: asyncio.Future[Any]) -> None:
    """Ends a still-waiting caller because the runner closed (runs on the event loop)."""
    if not outcome.done():
        outcome.set_exception(
            RunnerClosed("The runner closed while this job was waiting.")
        )


class HydrationRunner:
    """
    A single dedicated worker thread plus a `RecordStore` connection opened lazily on it the
    first time it's needed. The connection is only ever touched from that one thread, which
    satisfies sqlite3's thread-affinity requirement without `check_same_thread=False` or locks.

    The thread is a daemon and its queue is cancellable, deliberately not a
    `ThreadPoolExecutor`: an executor runs every queued job before the interpreter may exit, so
    a backlog of prefetch requests (the viewer queues one per tag) would keep the process alive
    for minutes after Ctrl+C. Here, closing the runner or exiting the interpreter abandons
    whatever hasn't started, and a request whose client gave up is skipped before it runs.
    Closing also releases every caller still waiting, even one whose job is mid-run (a running
    job can't be interrupted; its result is discarded), so a shutdown never waits on it.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        # Guards `_thread`, `_closed` and `_waiting`, and enqueueing versus closing.
        self._lock = threading.Lock()
        self._waiting: set[tuple[asyncio.AbstractEventLoop, asyncio.Future[Any]]] = (
            set()
        )
        self._closed = False

    def _work(self) -> None:
        store: RecordStore | None = None
        try:
            while (job := self._jobs.get()) is not None:
                future, fn = job
                if not future.set_running_or_notify_cancel():
                    continue  # cancelled while it was still queued
                try:
                    if store is None:
                        store = RecordStore.open(self._db_path, readonly=True)
                    future.set_result(fn(store))
                except (
                    BaseException
                ) as error:  # reported to whoever is awaiting this job
                    future.set_exception(error)
        finally:
            if store is not None:
                store.close()  # on the thread that opened it

    async def run(self, fn: Callable[[RecordStore], T]) -> T:
        """
        Runs `fn` against this runner's own store on its dedicated thread, off the calling
        event loop, and awaits the result without blocking that loop in the meantime.
        Cancelling the awaiting task cancels the job too, if it hasn't started yet.

        Raises:
            RunnerClosed: if the runner is closed, or closes before the job finishes.

        """
        loop = asyncio.get_running_loop()
        outcome: asyncio.Future[T] = loop.create_future()
        job: Future[T] = Future()

        def relay(
            done: Future[T],
        ) -> None:  # runs on the worker thread, or wherever it's cancelled
            try:
                loop.call_soon_threadsafe(_settle, outcome, done)
            except RuntimeError:
                pass  # the awaiting loop has already shut down

        job.add_done_callback(relay)
        waiter = (loop, outcome)
        with self._lock:
            if self._closed:
                raise RunnerClosed("This runner has been closed.")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._work, name="trendify-store", daemon=True
                )
                self._thread.start()
            self._waiting.add(waiter)
            self._jobs.put((job, fn))
        try:
            return await outcome
        except asyncio.CancelledError:
            job.cancel()  # skipped if it hasn't started yet
            raise
        finally:
            with self._lock:
                self._waiting.discard(waiter)

    def close(self, wait: bool = False) -> None:
        """
        Abandons jobs that haven't started, releases everyone still waiting on this runner
        (they get `RunnerClosed`), and tells the worker thread to close its store and stop. A
        job already running can't be interrupted, so by default this does not wait for it: a
        single heavy plot can take over a minute, and shutdown must not hang on that (the thread
        is a daemon, so the process can exit while it finishes). Safe to call more than once.

        Args:
            wait (bool): block until the worker thread has finished, including any job that
                was already running.

        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            waiting = list(self._waiting)
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job[0].cancel()
        for loop, outcome in waiting:
            try:
                loop.call_soon_threadsafe(_release, outcome)
            except RuntimeError:
                pass  # that loop has already shut down
        self._jobs.put(None)
        if wait and thread is not None:
            thread.join()
