"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger with console (and optional file) handlers.

    Args:
        level: Logging level (``logging.INFO`` or ``logging.ERROR`` etc.).
        log_file: Optional path to append log output.
    """
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
