"""Logging configuration with file + console + dashboard buffer."""

import logging
import sys
import time
from collections import deque
from pathlib import Path


class DashboardLogHandler(logging.Handler):
    """Stores log entries in a deque for the dashboard API."""

    def __init__(self, max_entries: int = 500):
        super().__init__()
        self.entries: deque[dict] = deque(maxlen=max_entries)

    def emit(self, record: logging.LogRecord) -> None:
        self.entries.append({
            "time": record.created,
            "level": record.levelname,
            "name": record.name,
            "message": self.format(record),
        })

    def get_entries(self, limit: int = 100) -> list[dict]:
        entries = list(self.entries)
        return entries[-limit:]


# Global instance for the dashboard to read
dashboard_handler = DashboardLogHandler()


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure logging for the application."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"bot_{time.strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    brief_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler — DEBUG level
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler — configured level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(brief_formatter)

    # Dashboard handler
    dashboard_handler.setLevel(logging.INFO)
    dashboard_handler.setFormatter(brief_formatter)

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.addHandler(dashboard_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
