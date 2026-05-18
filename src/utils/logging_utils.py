"""Logging setup utility."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a console handler.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
