"""
background_io.py — Non-blocking file persistence helpers
=========================================================
Moves CSV append and JSON snapshot writes onto a dedicated background thread so
the asyncio event loop is not blocked by filesystem latency during trading.
"""

from __future__ import annotations

import csv
import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class _AppendCsvTask:
    path: Path
    row: list[Any]
    header: Optional[list[str]] = None


@dataclass
class _WriteJsonTask:
    path: Path
    payload: Any
    indent: Optional[int] = None


_STOP = object()


class BackgroundIOManager:
    """Single-writer background thread for filesystem operations."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._run,
            name="background-io",
            daemon=True,
        )
        self._worker.start()

    def append_csv(
        self,
        path: Path | str,
        row: Iterable[Any],
        *,
        header: Optional[Iterable[str]] = None,
    ) -> None:
        self._queue.put(
            _AppendCsvTask(
                path=Path(path),
                row=list(row),
                header=list(header) if header is not None else None,
            )
        )

    def write_json_atomic(
        self,
        path: Path | str,
        payload: Any,
        *,
        indent: Optional[int] = None,
    ) -> None:
        self._queue.put(
            _WriteJsonTask(
                path=Path(path),
                payload=payload,
                indent=indent,
            )
        )

    def flush(self) -> None:
        """Block until all pending writes have been processed."""
        self._queue.join()

    def close(self) -> None:
        self.flush()
        self._queue.put(_STOP)
        self._worker.join(timeout=2)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is _STOP:
                    return
                if isinstance(task, _AppendCsvTask):
                    self._handle_append_csv(task)
                elif isinstance(task, _WriteJsonTask):
                    self._handle_write_json(task)
            except Exception as exc:
                logger.warning(f"Background I/O task failed: {exc}")
            finally:
                self._queue.task_done()

    @staticmethod
    def _handle_append_csv(task: _AppendCsvTask) -> None:
        task.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = bool(task.header and not task.path.exists())
        with open(task.path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(task.header)
            writer.writerow(task.row)

    @staticmethod
    def _handle_write_json(task: _WriteJsonTask) -> None:
        task.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = task.path.with_suffix(task.path.suffix + ".tmp")
        with open(tmp_path, "w") as handle:
            json.dump(task.payload, handle, indent=task.indent)
        tmp_path.replace(task.path)


default_io_manager = BackgroundIOManager()
