"""Background training and inference subprocess managers."""
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import psutil

from latentsync.finetune import FINETUNE_BASE_DIR, REPO_ROOT, logger

def _prune_debug_files(directory: Path, pattern: str, keep: int = 10) -> None:
    """Keep only the N most-recently-modified files matching `pattern` in `directory`.

    Tab 3 / 3.5 write a fresh tmp yaml per invocation; over a long session this
    leaks disk. We cap it at `keep` files (oldest deleted first).
    """
    if not directory.exists():
        return
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in matches[keep:]:
        try:
            old.unlink()
        except OSError:
            pass

class InferenceManager:
    """Track a single background inference subprocess (used by Tab 3 / Tab 3.5).

    Tab 3 inference runs are 5-10 minutes, so blocking the Gradio event
    loop is unacceptable. We spawn via Popen in a daemon thread, return
    immediately from the click handler, and update the UI by polling
    `status` from a Timer. Only one inference at a time (mirrors
    TrainingProcess).
    """

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_f = None
        self.log_path: Optional[Path] = None
        self.status: str = self.IDLE
        self.exit_code: Optional[int] = None
        self.result_video: Optional[Path] = None
        self.result_json: Optional[Path] = None
        self.result_warning: str = ""
        self.kind: str = ""  # "compare" / "validate"
        self.label: str = ""  # human-readable description (e.g. "compare base vs ft")
        self._lock = threading.Lock()

    def is_busy(self) -> bool:
        return self.status in (self.STARTING, self.RUNNING, self.CANCELLING)

    def is_running(self) -> bool:
        return self.status == self.RUNNING and self.proc is not None and self.proc.poll() is None

    def start(
        self,
        cmd: List[str],
        log_path: Path,
        kind: str,
        label: str,
        result_video: Path,
        result_json: Optional[Path] = None,
    ) -> bool:
        """Spawn the subprocess in a daemon thread.

        Returns True on accept, False if another inference is already busy.
        """
        with self._lock:
            if self.is_busy():
                return False
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = open(log_path, "w")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # so SIGINT kills the whole process group
                )
            except FileNotFoundError:
                log_f.close()
                self.status = self.FAILED
                self.exit_code = -1
                return False

            self.proc = proc
            self.log_f = log_f
            self.log_path = log_path
            self.status = self.STARTING
            self.exit_code = None
            self.result_video = result_video
            self.result_json = result_json
            self.result_warning = ""
            self.kind = kind
            self.label = label

        threading.Thread(
            target=self._monitor,
            args=(proc, log_f, log_path),
            daemon=True,
        ).start()
        return True

    def _monitor(self, proc: subprocess.Popen, log_f, log_path: Path) -> None:
        rc = proc.wait()
        log_f.close()
        with self._lock:
            self.exit_code = rc
            was_cancelling = self.status == self.CANCELLING
            if was_cancelling:
                self.status = self.CANCELLED
            elif rc == 0:
                self.status = self.DONE
            else:
                self.status = self.FAILED

    def stop(self) -> str:
        """Send SIGINT to the inference subprocess group. Non-blocking."""
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return "(没有运行的推理)"
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            except (ProcessLookupError, OSError) as exc:
                logger.warning("stop() failed: %s", exc)
            self.status = self.CANCELLING
        return "⏹ 停止信号已发出,等待子进程退出…"


_INFERENCE = InferenceManager()

class TrainingProcess:
    """Track a single background training subprocess.

    State is persisted to disk so that if the Gradio service restarts,
    we can reattach to a training subprocess that survived the restart
    (it runs in its own session via os.setsid).
    """

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self.started_at: Optional[str] = None
        self.cmd: List[str] = []
        self._pid: Optional[int] = None

    @staticmethod
    def _state_path() -> Path:
        path = FINETUNE_BASE_DIR / "training_logs" / "active_trainer.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_state(self) -> None:
        """Write active trainer metadata to disk."""
        pid = self.proc.pid if self.proc else self._pid
        if pid is None:
            return
        data = {
            "pid": pid,
            "log_path": str(self.log_path) if self.log_path else None,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "started_at": self.started_at,
            "cmd": self.cmd,
        }
        try:
            self._state_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("[TrainingProcess] failed to save state: %s", e)

    def clear_state(self) -> None:
        """Remove persisted state."""
        path = self._state_path()
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning("[TrainingProcess] failed to clear state: %s", e)

    def reattach(self) -> bool:
        """On startup, try to reattach to a training subprocess that is still alive."""
        path = self._state_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[TrainingProcess] corrupt state file %s: %s", path, e)
            self.clear_state()
            return False

        pid = data.get("pid")
        if not pid or not isinstance(pid, int):
            self.clear_state()
            return False

        if not self._pid_alive(pid):
            logger.info("[TrainingProcess] previously tracked pid=%s is no longer alive; clearing state", pid)
            self.clear_state()
            return False

        # Sanity check: the process should look like a LatentSync training job.
        try:
            cmdline = " ".join(psutil.Process(pid).cmdline() or [])
        except Exception:
            cmdline = ""
        if not any(token in cmdline for token in ("torchrun", "train_unet", "train_syncnet")):
            logger.warning(
                "[TrainingProcess] pid=%s does not look like a LatentSync training process (cmdline=%r); clearing state",
                pid, cmdline,
            )
            self.clear_state()
            return False

        self._pid = pid
        self.proc = None  # we don't have the Popen object, but we know the PID
        self.log_path = Path(data["log_path"]) if data.get("log_path") else None
        self.run_dir = Path(data["run_dir"]) if data.get("run_dir") else None
        self.started_at = data.get("started_at")
        self.cmd = data.get("cmd", [])
        logger.info(
            "[TrainingProcess] reattached to surviving training subprocess pid=%s, log=%s, run_dir=%s",
            pid, self.log_path, self.run_dir,
        )
        return True

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def is_running(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        if self._pid is not None:
            alive = self._pid_alive(self._pid)
            if not alive:
                self._pid = None
                self.clear_state()
            return alive
        return False

    @property
    def pid(self) -> Optional[int]:
        if self.proc is not None:
            return self.proc.pid
        return self._pid

    def stop(self) -> None:
        pid = self.pid
        if self.proc is not None and self.proc.poll() is None:
            try:
                # The training subprocess runs in its own session
                # (os.setsid), so signal the whole group — SIGINT to just
                # the torchrun pid can leave its workers behind.
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            except (ProcessLookupError, OSError) as e:
                logger.warning("[TrainingProcess] failed to signal pid=%s: %s", pid, e)
        elif pid is not None:
            # Reattached process: we only have the PID, send SIGINT to its group.
            try:
                os.killpg(os.getpgid(pid), signal.SIGINT)
                # Wait briefly for it to terminate.
                for _ in range(30):
                    if not self._pid_alive(pid):
                        break
                    time.sleep(0.5)
            except (ProcessLookupError, OSError) as e:
                logger.warning("[TrainingProcess] failed to signal pid=%s: %s", pid, e)
        self.proc = None
        self._pid = None
        self.clear_state()


_TRAINER = TrainingProcess()
# On module load, attempt to reattach to a training subprocess that survived a service restart.
_TRAINER.reattach()

