#!/usr/bin/env python3
"""Dummy RTSP cameras from YouTube videos (or local files), for testing without real cameras.

    YouTube / file --(yt-dlp download, once)--> data/test_media/youtube/<id>.mp4
        --(ffmpeg, real time, looped)--> MediaMTX --> rtsp://127.0.0.1:8555/<name>

Each feed behaves like an IP camera's H.264 sub-stream (<= 720p, no audio), so facereco reads
it exactly like a real camera.

    python scripts/dummy_rtsp.py cam1="https://www.youtube.com/watch?v=VIDEO_ID"
    python scripts/dummy_rtsp.py cam1="ytsearch1:people walking city street"   # first search hit
    python scripts/dummy_rtsp.py cam1=URL1 cam2=URL2 cam3=data/test_media/people_walking.mp4
    python scripts/dummy_rtsp.py --host 0.0.0.0 cam1=URL      # reachable from other machines too

Videos are downloaded once and then looped from disk (YouTube's direct links expire after a few
hours). YouTube live streams are relayed live instead. Ctrl+C stops everything.

Needs: brew install mediamtx ffmpeg; yt-dlp (in requirements.txt).
Only use videos you're allowed to use, and don't enrol people from public videos for face recognition.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "test_media" / "youtube"
MAX_HEIGHT = 720


def port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0


def start_mediamtx(host: str, port: int) -> subprocess.Popen:
    exe = shutil.which("mediamtx") or "/opt/homebrew/opt/mediamtx/bin/mediamtx"
    if not Path(exe).exists():
        sys.exit("mediamtx not found: brew install mediamtx")
    if port_open(host, port):
        sys.exit(f"port {port} is already in use - pick another with --port")
    # RTSP over TCP only; every other protocol off so nothing clashes with other servers.
    cfg = Path(tempfile.mkdtemp()) / "mediamtx.yml"
    cfg.write_text(f"""logLevel: warn
rtsp: yes
rtspAddress: {host}:{port}
rtspTransports: [tcp]
rtmp: no
hls: no
webrtc: no
srt: no
api: no
metrics: no
pprof: no
playback: no
moq: no
paths:
  all_others:
""")
    proc = subprocess.Popen([exe, str(cfg)])
    for _ in range(50):
        if port_open(host, port):
            return proc
        if proc.poll() is not None:
            sys.exit("mediamtx failed to start (see its output above)")
        time.sleep(0.1)
    sys.exit("mediamtx did not open its port")


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,r_frame_rate", "-of", "json", path],
                         capture_output=True, text=True).stdout
    streams = json.loads(out or "{}").get("streams") or [{}]
    return streams[0]


def resolve(source: str, minutes: float) -> tuple[str, bool]:
    """-> (local file path or live URL, is_live). Downloads only the first `minutes` of a video."""
    if Path(source).expanduser().exists():
        return str(Path(source).expanduser().resolve()), False
    import yt_dlp

    CACHE.mkdir(parents=True, exist_ok=True)
    # H.264 (avc1), <= 720p, <= 30 fps, video only: what a camera sub-stream looks like (no re-encoding).
    fmt = (f"bv*[vcodec^=avc1][height<={MAX_HEIGHT}][fps<=30][ext=mp4]/"
           f"bv*[vcodec^=avc1][height<={MAX_HEIGHT}][ext=mp4]/bv*[height<={MAX_HEIGHT}]/b[height<={MAX_HEIGHT}]/b")
    quiet = {"quiet": True, "no_warnings": True, "noprogress": True}
    with yt_dlp.YoutubeDL({**quiet, "format": fmt}) as ydl:
        info = ydl.extract_info(source, download=False)
        if info.get("_type") == "playlist":            # ytsearchN: / playlists -> first entry
            info = info["entries"][0]
        title = info.get("title", "?")
        if info.get("is_live"):
            print(f"  live stream: {title}")
            return info["url"], True
        duration = float(info.get("duration") or 0)
        cut = minutes > 0 and duration > minutes * 60
        stem = f"{info['id']}_{minutes:g}min" if cut else info["id"]
        target = CACHE / f"{stem}.mp4"
        if not target.exists():
            print(f"  downloading: {title} (first {minutes:g} of {duration / 60:.0f} min) -> "
                  f"{target.relative_to(ROOT)}" if cut else
                  f"  downloading: {title} ({duration / 60:.0f} min) -> {target.relative_to(ROOT)}")
            opts = {**quiet, "format": fmt, "outtmpl": str(CACHE / f"{stem}.%(ext)s")}
            if cut:
                from yt_dlp.utils import download_range_func

                opts["download_ranges"] = download_range_func(None, [(0, minutes * 60)])
            with yt_dlp.YoutubeDL(opts) as dl:
                dl.download([info["webpage_url"]])
            got = next((p for p in CACHE.glob(f"{stem}.*") if p.suffix not in (".part", ".ytdl")), None)
            if got is None:
                sys.exit(f"download failed for {source}")
            if got != target:
                got.rename(target)
        else:
            print(f"  cached: {title} -> {target.relative_to(ROOT)}")
    return str(target), False


def ensure_h264(path: str) -> str:
    """Camera-like H.264 <= 720p; re-encode once (cached) when the source isn't."""
    s = probe(path)
    if s.get("codec_name") == "h264" and int(s.get("height") or 0) <= MAX_HEIGHT:
        return path
    out = Path(path).with_name(Path(path).stem + "_h264_720p.mp4")
    if not out.exists():
        print(f"  re-encoding {s.get('codec_name')} {s.get('width')}x{s.get('height')} -> H.264 720p (once)")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-an",
                        "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'", "-c:v", "libx264", "-preset", "veryfast",
                        "-g", "50", "-pix_fmt", "yuv420p", str(out)], check=True)
    return str(out)


def ffmpeg_cmd(src: str, live: bool, url: str) -> list:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if not live:
        cmd += ["-re", "-stream_loop", "-1"]          # real time, loop forever
    cmd += ["-i", src, "-an"]
    if live:   # live HLS may be any codec/size: make it camera-like
        cmd += ["-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'", "-c:v", "libx264", "-preset", "veryfast",
                "-tune", "zerolatency", "-g", "50", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]                         # already H.264: no CPU spent re-encoding
    return cmd + ["-f", "rtsp", "-rtsp_transport", "tcp", url]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feeds", nargs="+", metavar="NAME=SOURCE",
                    help="camera name = YouTube URL, 'ytsearch1:<query>', or a local video file")
    ap.add_argument("--host", default="127.0.0.1", help="interface to serve on (0.0.0.0 = whole network)")
    ap.add_argument("--port", type=int, default=8555, help="RTSP port (8554 is often taken)")
    ap.add_argument("--minutes", type=float, default=10,
                    help="download only the first N minutes of long videos (0 = whole video)")
    args = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found: brew install ffmpeg")

    feeds = []
    for spec in args.feeds:
        name, sep, source = spec.partition("=")
        if not sep or not name or not source:
            sys.exit(f"expected NAME=SOURCE, got {spec!r}")
        print(f"[{name}]")
        src, live = resolve(source, args.minutes)
        if not live:
            src = ensure_h264(src)
            s = probe(src)
            print(f"  serving {s.get('codec_name')} {s.get('width')}x{s.get('height')} @ {s.get('r_frame_rate')}, looped")
        feeds.append((name, src, live))

    server = start_mediamtx(args.host, args.port)
    shown = socket.gethostbyname(socket.gethostname()) if args.host == "0.0.0.0" else args.host
    procs = {}
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    try:
        for name, src, live in feeds:
            procs[name] = subprocess.Popen(ffmpeg_cmd(src, live, f"rtsp://127.0.0.1:{args.port}/{name}"))
        time.sleep(2)
        print("\nDummy cameras running (Ctrl+C to stop):")
        for name, _, _ in feeds:
            print(f"  {name}: rtsp://{shown}:{args.port}/{name}")
        n0 = feeds[0][0]
        print(f"\nTest:      python scripts/test_gstreamer.py --url rtsp://{shown}:{args.port}/{n0}"
              f"\nAs camera: SELECT sp_upsert_camera('{n0}', '{n0}', 'rtsp://{shown}:{args.port}/{n0}', false, true,"
              f" '[[0.5,0],[1,0],[1,1],[0.5,1]]');")
        while not stop["now"]:
            for name, src, live in feeds:              # restart a feed if ffmpeg exits
                if procs[name].poll() is not None and not stop["now"]:
                    print(f"[{name}] ffmpeg exited ({procs[name].returncode}) - restarting")
                    time.sleep(1)
                    procs[name] = subprocess.Popen(ffmpeg_cmd(src, live, f"rtsp://127.0.0.1:{args.port}/{name}"))
            if server.poll() is not None:
                print("mediamtx stopped")
                break
            time.sleep(1)
    finally:
        for p in procs.values():
            p.terminate()
        server.terminate()
        for p in [*procs.values(), server]:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
