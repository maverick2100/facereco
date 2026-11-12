"""Aggregated health: periodic log line, logs/health.json and a small HTTP endpoint.

    GET /health   -> short status
    GET /metrics  -> everything (cpu, memory, ane, performance, cameras, pipeline)
    GET /cameras  -> per-camera health
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional

from config.settings import MonitoringSettings
from constants import LOGS_DIR
from ingestion.camera_health import CameraHealthRegistry
from monitoring.ane_monitor import ANEMonitor
from monitoring.cpu_monitor import CPUMonitor
from monitoring.memory_monitor import MemoryMonitor
from monitoring.performance import PerformanceTracker

log = logging.getLogger(__name__)


class _FastHTTPServer(ThreadingHTTPServer):
    """HTTPServer.server_bind() calls socket.getfqdn(), which can block for ~30 s on
    macOS while reverse DNS times out. Skip it - the name is never used."""

    daemon_threads = True

    def server_bind(self) -> None:
        import socketserver

        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), port


class SystemHealth:
    def __init__(self, settings: MonitoringSettings, perf: PerformanceTracker, cameras: CameraHealthRegistry,
                 extra: Optional[Dict[str, Callable[[], dict]]] = None) -> None:
        self.settings = settings
        self.perf = perf
        self.cameras = cameras
        self.extra = extra or {}
        self.cpu = CPUMonitor()
        self.mem = MemoryMonitor()
        self.ane = ANEMonitor(perf, settings.enable_powermetrics)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ data
    def metrics(self) -> dict:
        with self._lock:  # psutil cpu_percent is interval-based; serialise samplers
            cpu = self.cpu.sample()
            mem = self.mem.sample()
        data = {
            "time": time.time(),
            "cpu": cpu,
            "memory": mem,
            "ane": self.ane.sample(),
            "performance": self.perf.snapshot(),
            "cameras": self.cameras.summary(),
        }
        for k, fn in self.extra.items():
            try:
                data[k] = fn()
            except Exception as exc:
                data[k] = {"error": str(exc)}
        return data

    def health(self) -> dict:
        cams = self.cameras.summary()
        status = "ok" if cams["total"] and cams["streaming"] == cams["total"] else "degraded"
        if cams["total"] and cams["streaming"] == 0:
            status = "down"
        return {"status": status, "cameras": cams, "uptime_s": self.perf.snapshot()["uptime_s"]}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.ane.start()
        self._thread = threading.Thread(target=self._run, name="health", daemon=True)
        self._thread.start()
        if self.settings.health_port:
            self._start_http()

    def stop(self) -> None:
        self._stop.set()
        self.ane.stop()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def _run(self) -> None:
        while not self._stop.wait(self.settings.health_interval_s):
            try:
                m = self.metrics()
                self._log_summary(m)
                tmp = LOGS_DIR / "health.json.tmp"
                tmp.write_text(json.dumps(m, indent=2, default=str))
                tmp.replace(LOGS_DIR / "health.json")
            except Exception:
                log.exception("health sampling failed")

    def _log_summary(self, m: dict) -> None:
        c = m["performance"]["counters"]
        lat = m["performance"]["latency"]

        def rate(name: str) -> float:
            return c.get(name, {}).get("per_s", 0.0)

        log.info(
            "health | cams %d/%d up %.0f fps | det %.1f fr/s (%.1f ms/fr) | faces %.1f/s | emb %.1f/s (%.1f ms/face) "
            "| infer busy %.0f%% | CPU proc %.0f%% of 1 core, sys %.0f%% | RSS %.0f MB, avail %.0f MB",
            m["cameras"]["streaming"], m["cameras"]["total"], m["cameras"]["total_fps"],
            rate("frames_processed"), lat.get("scrfd_per_frame_ms", {}).get("mean_ms", 0.0),
            rate("faces_detected"), rate("faces_embedded"), lat.get("arcface_per_face_ms", {}).get("mean_ms", 0.0),
            m["ane"]["inference_busy_pct"], m["cpu"]["process_pct_of_core"], m["cpu"]["system_pct"],
            m["memory"]["process_rss_mb"], m["memory"]["system_available_mb"],
        )
        if m["cameras"]["down"]:
            log.warning("cameras not streaming: %s", ", ".join(m["cameras"]["down"]))
        if m["memory"]["low_memory"]:
            log.warning("low system memory: %s MB available", m["memory"]["system_available_mb"])

    def _start_http(self) -> None:
        health = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                routes = {"/health": health.health, "/metrics": health.metrics,
                          "/cameras": lambda: {"cameras": health.cameras.snapshot()}}
                fn = routes.get(self.path.split("?")[0])
                if fn is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(fn(), indent=2, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        try:
            self._server = _FastHTTPServer((self.settings.health_host, self.settings.health_port), Handler)
        except OSError as exc:
            log.warning("Health endpoint disabled: %s", exc)
            return
        threading.Thread(target=self._server.serve_forever, name="health-http", daemon=True).start()
        log.info("Health endpoint on http://%s:%d/health", self.settings.health_host, self.settings.health_port)
