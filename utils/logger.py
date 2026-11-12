"""Logging setup driven by config/logging.yaml."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml

from constants import LOGS_DIR


def setup_logging(config_path: Optional[Path] = None, level: Optional[str] = None) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        # Every file handler writes into logs/ regardless of cwd.
        for handler in (cfg.get("handlers") or {}).values():
            if "filename" in handler:
                handler["filename"] = str(LOGS_DIR / Path(handler["filename"]).name)
        if level:
            cfg.setdefault("root", {})["level"] = level
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=level or "INFO",
            format="%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
        )
    # Quieten chatty third-party loggers.
    for noisy in ("coremltools", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
