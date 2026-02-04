"""Structured logging for the pipeline."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


class StageFormatter(logging.Formatter):
    """Formatter that includes elapsed time since stage start."""

    def __init__(self):
        super().__init__(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        self._stage_start = time.time()

    def reset_timer(self):
        self._stage_start = time.time()

    def format(self, record):
        elapsed = time.time() - self._stage_start
        record.msg = f"[{elapsed:7.1f}s] {record.msg}"
        return super().format(record)


_formatter = StageFormatter()


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure logging for the pipeline.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to file in addition to stderr.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_formatter)
    root.addHandler(stderr_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(_formatter)
        root.addHandler(file_handler)


def reset_stage_timer():
    """Reset the elapsed timer (call at the start of each stage)."""
    _formatter.reset_timer()
