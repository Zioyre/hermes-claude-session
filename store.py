import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .sync import SessionRecord
except ImportError:
    from sync import SessionRecord


class RecordStore:
    """Per-Hermes-session records live beside the plugin state dir and must never be world-readable."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, session_id: str) -> Path:
        return self._root / (hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32] + ".json")

    def _ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._root, 0o700)

    def acquire(self, session_id: str) -> int:
        """Hold this session's native transcript exclusively until the process is retired.

        The OS releases the lock if Hermes exits, so owner_pid is diagnostic rather than a lease.
        Lock files stay in place: unlinking one could let another process lock a different inode.
        """
        import fcntl
        self._ensure_root()
        fd = os.open(self._path(session_id).with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        return fd

    @staticmethod
    def release(fd: int) -> None:
        os.close(fd)

    def load(self, session_id: str) -> SessionRecord | None:
        try:
            data = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            return SessionRecord.from_dict(data)
        except (AttributeError, TypeError, ValueError, KeyError):
            return None

    def save(self, session_id: str, record: SessionRecord) -> None:
        self._ensure_root()
        record.owner_pid = os.getpid()
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        handle, temp_path = tempfile.mkstemp(dir=self._root, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self._path(session_id))
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def log_edit(self, entry: dict) -> None:
        self._ensure_root()
        payload = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        descriptor = os.open(self._root / "edits.jsonl", os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
