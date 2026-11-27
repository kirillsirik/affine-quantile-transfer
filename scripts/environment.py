"""Record the execution environment of a run: platform, emulation, Python, packages and BLAS.

Usage, from the repository root:
    python -m scripts.environment --out FILE [--workers N]

Emulation is detected from sysctl.proc_translated on macOS and from /proc/cpuinfo on Linux
(Rosetta reports a "VirtualApple" CPU, QEMU a "QEMU" CPU). File paths are left out.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "threadpoolctl")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "OPENBLAS_CORETYPE")


def _run(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def cpu_and_emulation():
    system = platform.system()
    if system == "Darwin":
        cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        emulation = "Rosetta 2" if _run(["sysctl", "-n", "sysctl.proc_translated"]) == "1" else "none"
        return cpu, emulation
    cpuinfo = Path("/proc/cpuinfo")
    text = cpuinfo.read_text(errors="replace") if cpuinfo.exists() else ""
    names = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("model name")]
    parts = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("CPU part")]
    cpu = names[0] if names else (f"ARM CPU part {parts[0]}" if parts else platform.processor())
    if "VirtualApple" in text:
        emulation = "Rosetta (x86-64 translated on Apple silicon)"
    elif "QEMU" in text:
        emulation = "QEMU"
    else:
        emulation = "none detected"
    return cpu, emulation


def blas_info():
    import numpy
    import scipy
    from threadpoolctl import threadpool_info

    def build(module):
        config = module.show_config(mode="dicts")["Build Dependencies"]
        return {name: {k: v for k, v in config[name].items() if k in ("name", "version", "found")}
                for name in ("blas", "lapack") if name in config}

    loaded = [{k: v for k, v in entry.items() if k != "filepath"} for entry in threadpool_info()]
    return {"numpy_build": build(numpy), "scipy_build": build(scipy), "loaded": loaded}


def git_state():
    commit = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])) if commit else None
    return {"commit": commit or None, "uncommitted_changes": dirty}


def collect(workers=None):
    cpu, emulation = cpu_and_emulation()
    return {
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "emulation": emulation,
        "container": Path("/.dockerenv").exists(),
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "blas": blas_info(),
        "thread_settings": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "workers": workers,
        "git": git_state(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    info = collect(args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2) + "\n")
    loaded = ", ".join(f"{e.get('internal_api')} {e.get('version')} {e.get('architecture') or ''}".strip()
                       for e in info["blas"]["loaded"]) or "no runtime BLAS reported"
    print(f"{info['system']} {info['machine']} ({info['cpu']}), emulation: {info['emulation']}, "
          f"Python {info['python']}, numpy {info['packages']['numpy']}, scipy {info['packages']['scipy']}, "
          f"BLAS: {info['blas']['numpy_build'].get('blas', {}).get('name')} build, {loaded}")


if __name__ == "__main__":
    sys.exit(main())
