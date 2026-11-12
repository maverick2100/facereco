"""Apple Neural Engine (ANE) configuration helpers.

Compute unit choice:
  CPU_AND_NE  (default) - Core ML schedules layers on the ANE and falls back to
                          CPU only for unsupported ops. GPU is never used, so
                          there is no GPU<->ANE ping-pong.
  ALL                   - ANE + GPU + CPU. Sometimes faster, but Core ML may put
                          parts on the GPU.
  CPU_ONLY / CPU_AND_GPU - for benchmarking/comparison only.

Getting a model to run (almost) fully on the ANE:
  * fixed input shapes (no flexible/range shapes)
  * FP16 compute precision (mlprogram default)
  * ImageType inputs so scale/bias normalisation happens inside the model
  * minimum_deployment_target >= macOS 13
Run scripts/benchmark_scrfd.py: if CPU_AND_NE is much faster AND uses far less
CPU time than CPU_ONLY, the model is running on the ANE.
"""
from __future__ import annotations

import logging
import platform
from typing import List

log = logging.getLogger(__name__)

RECOMMENDED_COMPUTE_UNITS = "CPU_AND_NE"
VALID_COMPUTE_UNITS = ("CPU_AND_NE", "ALL", "CPU_ONLY", "CPU_AND_GPU")


def resolve_compute_units(name: str):
    import coremltools as ct

    key = (name or RECOMMENDED_COMPUTE_UNITS).strip().upper().replace("-", "_")
    if key not in VALID_COMPUTE_UNITS:
        raise ValueError(f"compute units must be one of {VALID_COMPUTE_UNITS}, got {name!r}")
    return getattr(ct.ComputeUnit, key)


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def macos_version() -> tuple:
    ver = platform.mac_ver()[0]
    try:
        return tuple(int(p) for p in ver.split("."))
    except ValueError:
        return ()


def check_environment() -> List[str]:
    """Return a list of warnings that would prevent / degrade ANE usage."""
    warnings: List[str] = []
    if not is_apple_silicon():
        warnings.append("Not running on Apple Silicon (arm64) - no Neural Engine available.")
    mv = macos_version()
    if mv and mv < (13,):
        warnings.append(f"macOS {'.'.join(map(str, mv))} is old; macOS 14+ recommended for Core ML on ANE.")
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        warnings.append("Python is running under Rosetta (x86_64). Install a native arm64 Python.")
    try:
        import coremltools as ct

        major = int(ct.__version__.split(".")[0])
        if major < 7:
            warnings.append(f"coremltools {ct.__version__} is old; install >= 8.")
    except Exception as exc:  # pragma: no cover
        warnings.append(f"coremltools not importable: {exc}")
    return warnings


def log_environment() -> None:
    log.info("Platform: %s %s, macOS %s", platform.system(), platform.machine(), platform.mac_ver()[0] or "?")
    for w in check_environment():
        log.warning(w)
