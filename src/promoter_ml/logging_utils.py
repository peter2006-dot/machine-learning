"""Consistent console and file logging for all project scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    name: str,
    log_directory: str | Path,
    filename: str = "pipeline.log",
    level: str = "INFO",
) -> logging.Logger:
    """Create an idempotent logger with console and UTF-8 file handlers."""
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
