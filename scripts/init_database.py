#!/usr/bin/env python3
"""Create the SQLite database and register cameras from config/cameras.yaml."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import load_settings  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import Database  # noqa: E402
from database.event_repository import EventRepository  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def main() -> int:
    settings = load_settings()
    setup_logging(settings.logging_config, settings.log_level)
    db = Database(settings.storage.db_path)
    init_schema(db)
    n = CameraRepository(db).sync(settings.cameras)
    print(f"database : {settings.storage.db_path}")
    print(f"cameras  : {n} registered")
    print(f"persons  : {len(PersonRepository(db).list())}")
    print(f"events   : {EventRepository(db).count()}")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
