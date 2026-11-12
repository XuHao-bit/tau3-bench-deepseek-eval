"""服务退出前完成后台写入，再关闭数据库。"""

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


class LocalMemOSStopTests(unittest.TestCase):
    def test_database_stays_available_until_server_finishes_pending_write(self):
        script = Path(__file__).resolve().parents[1] / "local_memos/stop.sh"
        child = """
import signal, sys, time
from pathlib import Path
root, service = Path(sys.argv[1]), sys.argv[2]
def finish(*args):
    if service == 'server':
        time.sleep(0.15)
        (root / 'write_completed').touch()
    else:
        (root / 'database_stop_order').write_text(str((root / 'write_completed').exists()))
    sys.exit(0)
signal.signal(signal.SIGTERM, finish)
(root / (service + '.ready')).touch()
while True:
    time.sleep(0.05)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            shutil.copyfile(script, root / "stop.sh")
            processes = []
            reapers = []
            try:
                for service in ["server", "neo4j"]:
                    process = subprocess.Popen(
                        [sys.executable, "-c", child, str(root), service],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    processes.append(process)
                    (runtime / f"{service}.pid").write_text(str(process.pid))
                    reaper = threading.Thread(target=process.wait)
                    reaper.start()
                    reapers.append(reaper)
                deadline = time.monotonic() + 5
                while not all((root / f"{service}.ready").exists() for service in ["server", "neo4j"]):
                    self.assertLess(time.monotonic(), deadline, "替身服务未启动")
                    time.sleep(0.01)
                result = subprocess.run(["bash", str(root / "stop.sh")],
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((root / "database_stop_order").read_text(), "True")
                self.assertFalse((runtime / "server.pid").exists())
                self.assertFalse((runtime / "neo4j.pid").exists())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                for reaper in reapers:
                    reaper.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
