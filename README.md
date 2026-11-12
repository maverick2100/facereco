# Face Recognition: 20 cameras on a Mac mini M4

This system recognizes faces in real time from 20 RTSP cameras. It runs SCRFD (face detection) and ArcFace (face recognition) as Core ML models on the **Apple Neural Engine (ANE)**, and keeps CPU use low.

## How CPU use is kept low

| Stage | Where it runs | Why it's cheap |
|---|---|---|
| H.264/H.265 decode | **Media engine** (`vtdec` / VideoToolbox) | Hardware decode, zero CPU |
| Frame-rate reduction | GStreamer `videorate` | Only `fps` frames/s per camera reach Python |
| Colour conversion | `videoconvert` (1 thread) | Only on the kept frames |
| Scheduling | latest-frame-per-camera, fair round robin | No backlog, no unbounded memory |
| SCRFD detection | **ANE** (`CPU_AND_NE`), batched | Decode touches only anchors above threshold |
| Alignment + quality | OpenCV (~0.1 ms/face) | Blurry, profile and tiny faces never reach ArcFace |
| Track cache | Python | A face that is still in view reuses its identity; ArcFace re-checks it every 2 s |
| ArcFace embedding | **ANE**, batched up to 16 | |
| Matching | One matrix multiply (Accelerate BLAS) | 10k-embedding gallery takes under 1 ms |
| DB writes | Background thread, batched, SQLite WAL | |

Thread pools in numpy and OpenCV are pinned to 1 thread (see `main.py`), so CPU time isn't lost to many small threads.

```
cameras ─► GStreamer (vtdec, videorate, appsink) ─► CameraQueue (1 slot / cam)
        ─► FrameScheduler ─► [detector thread] SCRFD on ANE (batch 4)
        ─► filter ─► track-cache ─► align ─► quality ─► InferenceQueue
        ─► [recognizer thread] ArcFace on ANE (batch 16) ─► matcher ─► de-dup ─► EventService ─► SQLite / logs / listeners
```

## Install

Use a **native arm64** Python 3.10–3.12. Don't use the system Python 3.9.

```bash
brew install python@3.12 gstreamer pygobject3

cd face_recognition
# --system-site-packages lets the venv see Homebrew's PyGObject (gi)
/opt/homebrew/bin/python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

python -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"
gst-inspect-1.0 vtdec          # must exist (VideoToolbox hardware decoder)
```

Then put your models in place:

```
models/scrfd.mlpackage
models/arcface.mlpackage
```

## First run checklist

```bash
cp .env.example .env                        # then edit .env
# edit config/cameras.yaml (URLs, sub-streams, fps)

python scripts/test_coreml.py               # prints model I/O spec + test inference
python scripts/test_coreml.py --image group.jpg   # check detections on a real photo
python scripts/benchmark_scrfd.py           # confirms the ANE is being used (see below)
python scripts/benchmark_arcface.py
python scripts/test_gstreamer.py --cameras cam01,cam02   # stream check, no models
python scripts/init_database.py

# enrol people: data/enrollment/<Person Name>/*.jpg  (3-10 photos each)
python main.py enroll
python main.py persons

python scripts/benchmark_pipeline.py --image group.jpg --cameras 20 --fps 4   # full load test
python main.py run
```

Optional: `sudo python main.py run` also reports **ANE power in mW** from `powermetrics`. A non-zero value confirms the ANE is in use.

## Commands

```bash
python main.py run [--cameras cam01,cam05]
python main.py enroll [--dir data/enrollment] [--replace]
python main.py enroll --name "Alice" --images a.jpg b.jpg [--external-id E123]
python main.py persons [--all]
python main.py delete-person 3
python main.py export --since 2026-11-13 --until 2026-11-14 [--camera cam01] [--known-only]
python main.py info
pytest -q
```

Health endpoint (localhost only by default):
`curl localhost:8085/health`, `/metrics` and `/cameras`. A summary line is also logged every 30 s, and full metrics go to `logs/health.json`.

Events are stored in SQLite (`data/face_recognition.db`), in `logs/events.log` (JSON lines) and, if `FR_SAVE_SNAPSHOTS=true`, as JPEG crops in `data/exports/snapshots/`. To push events to other systems (webhook, MQTT, door relay), use `EventService.add_listener(fn)`.

Enrolment can run while `main.py run` is running. The gallery reloads itself within `FR_GALLERY_RELOAD_S`.

## Checking that the models really run on the ANE

`scripts/benchmark_scrfd.py` prints **CPU ms/item** for each compute unit. When the ANE is doing the work, `CPU_AND_NE` shows much lower CPU ms/item than `CPU_ONLY`, often 5–20× lower. If the two are close, Core ML is running most layers on the CPU. The usual causes:

1. **Flexible input shapes.** The ANE needs fixed shapes. Convert with fixed shapes, e.g. SCRFD `(1,3,640,640)` and ArcFace `(1,3,112,112)` or a fixed batch such as `(16,3,112,112)`.
2. **FP32 model.** Convert as an `mlprogram` with `compute_precision=ct.precision.FLOAT16` (the default).
3. **MultiArray input.** Use an `ImageType` input so normalization runs inside the model on the ANE. The code detects either kind automatically:
   - SCRFD: `ct.ImageType(shape=(1,3,640,640), scale=1/128, bias=[-127.5/128]*3, color_layout=ct.colorlayout.RGB)`
   - ArcFace: `ct.ImageType(shape=(1,3,112,112), scale=1/127.5, bias=[-1,-1,-1], color_layout=ct.colorlayout.RGB)`
4. **Unsupported ops.** Older SCRFD exports can include dynamic reshapes. Exporting with the reshapes removed (raw NCHW heads) is fine, because `scrfd_engine.py` decodes both layouts.
5. `minimum_deployment_target=ct.target.macOS14`.

**Fixed batch for ArcFace.** If the ArcFace model is converted with a fixed batch dimension, e.g. `(16,3,112,112)`, `arcface_engine.py` detects it and sends real tensor batches, padding the last one. That is the most ANE-efficient setup. Otherwise it uses Core ML's batch API (`predict([...])`).

## Capacity planning (20 cameras)

- Load = cameras × fps: 20 × 4 = **80 detections/s**. On the ANE, SCRFD-2.5G/10G at 640² usually takes a few ms per frame, so this leaves plenty of room. Check with `benchmark_scrfd.py`, then raise or lower `fps` per camera.
- Use camera **sub-streams** (720p or lower). Decoding 20 × 4K main streams wastes the media engine and memory bandwidth.
- If `/metrics` shows `inference_busy_pct` near 100% or rising `scheduler.stale_dropped`, lower `fps`, set an `roi`, or raise `FR_MIN_FACE_SIZE`.
- Memory: about 2.7 MB per 720p frame × 1 slot per camera, plus models (about 100–200 MB). Expect under 1.5 GB RSS.

## Tuning

| Variable | Effect |
|---|---|
| `FR_MATCH_THRESHOLD` (0.45) | Higher means fewer false matches but more "unknown" results. Calibrate on your own data (0.35–0.55 is typical). |
| `FR_MATCH_MARGIN` | Required gap between the best and second-best person |
| `FR_TRACK_REVERIFY_S` | Higher means fewer ArcFace calls; 0 disables the track cache |
| `FR_EVENT_COOLDOWN_S` | A person must be out of view this long before a new event is logged |
| `FR_DET_BATCH` / `FR_REC_BATCH` | Larger batches give better ANE efficiency but slightly more latency |
| `fps`, `roi`, `priority` (cameras.yaml) | Per-camera load control |
| `FR_COMPUTE_UNITS` | Try `ALL` if your benchmark shows it's faster; `CPU_AND_NE` avoids the GPU |

## Troubleshooting

- **`Unrecognised SCRFD outputs`**: run `scripts/test_coreml.py` to see the output shapes. The engine expects 3 strides × (score, bbox, kps). Set `FR_SCRFD_SIGMOID=true` if the scores are logits.
- **Every face is "unknown" even after enrolment**: check the ArcFace normalization (`FR_ARCFACE_MEAN/STD`, MultiArray models only) and compare two photos of the same person with `test_coreml.py --image`.
- **Camera keeps reconnecting**: test the URL with `scripts/test_gstreamer.py --url rtsp://...`. Set `codec: h265` for HEVC cameras, or `codec: auto`.
- **No GStreamer**: the app falls back to OpenCV/FFmpeg (`FR_INGESTION_BACKEND=opencv`). This works, but software decoding costs far more CPU.
