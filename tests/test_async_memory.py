"""异步写入的并发顺序、真实落库状态与重启状态测试。"""

import copy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextvars import ContextVar

from local_memos.async_memory import AsyncMemoryWriter


class Request(SimpleNamespace):
    def model_copy(self, *, deep=False, update=None):
        result = copy.deepcopy(self) if deep else copy.copy(self)
        result.__dict__.update(update or {})
        return result


def request(ticket="one", session="task_001", cube="cube-a"):
    return Request(task_id=ticket, user_id="owner", session_id=session,
                   writable_cube_ids=[cube], async_mode="async", mode=None,
                   messages=[{"role": "user", "content": ticket}])


class AsyncWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.journal = Path(self.temp.name) / "async.jsonl"

    def test_submit_returns_before_processing_and_preserves_session_order(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def process(req):
            calls.append(req.task_id)
            self.assertEqual((req.async_mode, req.mode), ("sync", "fine"))
            if req.task_id == "one":
                entered.set()
                self.assertTrue(release.wait(3))
            return SimpleNamespace(data=[{"memory_id": req.task_id}])

        writer = AsyncMemoryWriter(process, lambda cube, ids: ids, self.journal, workers=2)
        try:
            receipt = writer.submit(request())
            self.assertEqual(receipt["status"], "submitted")
            self.assertTrue(entered.wait(2))
            writer.submit(request("two"))
            self.assertEqual(calls, ["one"])
            self.assertEqual(writer.status("cube-a", "task_001", "two", "owner")["status"], "submitted")
        finally:
            release.set()
            writer.close()
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual(writer.status("cube-a", "task_001", "two", "owner")["status"], "completed")

    def test_distinct_sessions_process_concurrently(self):
        barrier = threading.Barrier(2)

        def process(req):
            barrier.wait(timeout=3)
            return SimpleNamespace(data=[])

        writer = AsyncMemoryWriter(process, lambda *args: [], self.journal, workers=2)
        writer.submit(request("one", "task_001"))
        writer.submit(request("two", "task_002"))
        writer.close()
        self.assertEqual(writer.status("cube-a", "task_001", "one", "owner")["status"], "completed")
        self.assertEqual(writer.status("cube-a", "task_002", "two", "owner")["status"], "completed")

    def test_missing_returned_id_is_failed_instead_of_completed(self):
        writer = AsyncMemoryWriter(
            lambda req: SimpleNamespace(data=[{"memory_id": "not-stored"}]),
            lambda cube, ids: [], self.journal)
        writer.submit(request())
        writer.close()
        status = writer.status("cube-a", "task_001", "one", "owner")
        self.assertEqual(status["status"], "failed")
        self.assertIn("无法读回", status["error"])

    def test_restart_marks_unfinished_ticket_unknown_without_resubmitting(self):
        self.journal.write_text('{"task_id":"one","status":"submitted","memory_ids":[],"cube_id":"cube-a","session_id":"task_001","user_id":"owner"}\n')
        calls = []
        writer = AsyncMemoryWriter(lambda req: calls.append(req), lambda *args: [], self.journal)
        self.assertEqual(writer.status("cube-a", "task_001", "one", "owner")["status"], "unknown")
        self.assertEqual(writer.submit(request())["status"], "unknown")
        writer.close()
        self.assertEqual(calls, [])

    def test_ticket_scope_does_not_leak_to_other_task_or_cube(self):
        writer = AsyncMemoryWriter(lambda req: SimpleNamespace(data=[]), lambda *args: [], self.journal)
        writer.submit(request())
        writer.close()
        self.assertEqual(writer.status("other", "task_001", "one", "owner")["status"], "unknown")
        self.assertEqual(writer.status("cube-a", "task_002", "one", "owner")["status"], "unknown")

    def test_background_job_keeps_request_trace_context(self):
        trace = ContextVar("test_trace", default=None)
        seen = []
        def process(req):
            seen.append(trace.get())
            return SimpleNamespace(data=[])
        writer = AsyncMemoryWriter(process, lambda *args: [], self.journal)
        token = trace.set("trace-123")
        try:
            receipt = writer.submit(request(), trace_id=trace.get())
        finally:
            trace.reset(token)
        writer.close()
        self.assertEqual(seen, ["trace-123"])
        self.assertEqual(receipt["trace_id"], "trace-123")


if __name__ == "__main__":
    unittest.main()
