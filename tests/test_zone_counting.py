import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import CameraConfig  # noqa: E402
from database.connection import Database  # noqa: E402
from database.person_count_repository import PersonCountRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from person_detection.person_counter import PersonCounter  # noqa: E402
from person_detection.zone_counter import ZoneCounter, points_in_polygon  # noqa: E402

# Zone = right half of a 1280x720 frame.
ZONE = [[640, 0], [1280, 0], [1280, 720], [640, 720]]


def _box(cx, bottom=600, w=100, h=300):
    return [cx - w / 2, bottom - h, cx + w / 2, bottom]


def test_points_in_polygon_concave():
    # L-shape: the notch (900, 100) is outside.
    poly = np.array([[0, 0], [800, 0], [800, 400], [1200, 400], [1200, 700], [0, 700]], np.float32)
    pts = np.array([[100, 100], [900, 100], [1000, 500], [1300, 500]], np.float32)
    assert points_in_polygon(pts, poly).tolist() == [True, False, True, False]


def test_walk_in_then_out_counts_once_each():
    z = ZoneCounter("cam01", ZONE, min_frames=2)
    ins = outs = 0
    xs = [300, 400, 500, 700, 800, 900, 800, 500, 400, 300]   # in at frame 3, out at frame 7
    for i, x in enumerate(xs):
        for c in z.update([1], np.array([_box(x)], np.float32), 1000.0 + i, alive_ids=[1]):
            ins += c.direction == "in"
            outs += c.direction == "out"
    assert (ins, outs) == (1, 1)
    assert (z.total_in, z.total_out, z.inside_now) == (1, 1, 0)


def test_jitter_on_edge_is_ignored():
    z = ZoneCounter("cam01", ZONE, min_frames=2)
    for i, x in enumerate([600, 680, 600, 680, 600, 680, 600]):  # never 2 frames in a row inside
        assert z.update([1], np.array([_box(x)], np.float32), float(i), alive_ids=[1]) == []


def test_first_seen_inside_is_not_an_in_and_feet_anchor_is_used():
    z = ZoneCounter("cam01", [[0, 500], [1280, 500], [1280, 720], [0, 720]], anchor="bottom", min_frames=1)
    # Box centre (y=450) is above the zone line, but the feet (y=600) are in it.
    assert z.update([7], np.array([_box(300)], np.float32), 0.0, alive_ids=[7]) == []
    assert z.inside_now == 1 and z.total_in == 0
    z.update([], np.zeros((0, 4), np.float32), 1.0, alive_ids=[])   # track dropped -> state dropped
    assert z.inside_now == 0


def test_counter_writes_in_out_into_interval_rows_and_crossings():
    c = PersonCounter(["cam01", "cam02"], interval_s=60, track_min_hits=1, zones={"cam01": ZONE},
                      zone_min_frames=2)
    for i, x in enumerate([300, 450, 600, 750, 900, 1000]):  # walks steadily into the zone
        c.update("cam01", np.array([_box(x)], np.float32), 1000.0 + i, i * 0.25)
        c.update("cam02", np.array([_box(x)], np.float32), 1000.0 + i, i * 0.25)
    rows = {r.camera_id: r for r in c.flush()}
    assert (rows["cam01"].in_count, rows["cam01"].out_count, rows["cam01"].zone_count) == (1, 0, 1)
    assert (rows["cam02"].in_count, rows["cam02"].out_count) == (0, 0)   # no zone on cam02
    crossings = c.take_crossings()
    assert [(x.camera_id, x.direction) for x in crossings] == [("cam01", "in")]
    assert c.take_crossings() == []
    assert c.stats()["in_total"] == 1

    db = Database(":memory:")
    init_schema(db)
    repo = PersonCountRepository(db)
    repo.insert_many(r.as_row() for r in rows.values())
    repo.insert_crossings(x.as_row() for x in crossings)
    assert repo.crossings()[0]["direction"] == "in"
    assert {t["camera_id"]: t["in_count"] for t in repo.totals()} == {"cam01": 1, "cam02": 0}


def test_old_database_gets_new_columns(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE person_counts (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL,
        period_start REAL NOT NULL, period_end REAL NOT NULL, frames INTEGER NOT NULL,
        current_count INTEGER NOT NULL, min_count INTEGER NOT NULL, max_count INTEGER NOT NULL,
        avg_count REAL NOT NULL, new_persons INTEGER NOT NULL, ts REAL NOT NULL)""")
    con.execute("INSERT INTO person_counts VALUES (1, 'cam01', 0, 60, 10, 1, 0, 2, 1.0, 3, 60)")
    con.commit()
    con.close()
    db = Database(path)
    init_schema(db)
    init_schema(db)  # idempotent
    row = db.fetchone("SELECT in_count, out_count, zone_count FROM person_counts")
    assert (row["in_count"], row["out_count"], row["zone_count"]) == (0, 0, 0)


def test_count_zone_parsing():
    base = {"camera_id": "a", "url": "x", "width": 1000, "height": 500}
    c = CameraConfig.from_dict({**base, "count_zone": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]})
    assert c.count_zone == [[500, 0], [1000, 0], [1000, 500], [500, 500]]       # fractions -> pixels
    c = CameraConfig.from_dict({**base, "count_zone": "[[10, 10], [200, 10], [200, 300]]"})  # JSON text
    assert c.count_zone[2] == [200, 300]
    assert CameraConfig.from_dict(base).count_zone is None
    with pytest.raises(ValueError):
        CameraConfig.from_dict({**base, "count_zone": [[0, 0], [10, 10]]})
