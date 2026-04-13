"""
runtime_logging.py — Queue-backed logging setup
===============================================
Routes logging through a queue so stdout/file handlers do not block the event
loop directly during trading cycles.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import SimpleQueue
from typing import Any


def configure_logging(log_path: str | Path) -> QueueListener:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    queue: SimpleQueue[Any] = SimpleQueue()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    queue_handler = QueueHandler(queue)
    root.addHandler(queue_handler)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)

    listener = QueueListener(queue, stream_handler, file_handler, respect_handler_level=True)
    listener.start()
    return listener
