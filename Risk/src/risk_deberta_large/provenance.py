from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_text, utc_now, write_json


def _run(command: list[str], cwd: Path) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.returncode, completed.stdout
    except OSError as exc:
        return 127, str(exc)


def capture_provenance(
    run_dir: Path,
    project_root: Path,
    command: list[str],
    data_manifest: dict[str, Any],
    split_metadata: dict[str, Any],
) -> None:
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / "command.txt", " ".join(command) + "\n")

    git_code, git_commit = _run(["git", "rev-parse", "HEAD"], project_root)
    git_commit_text = git_commit.strip() if git_code == 0 else f"UNAVAILABLE: {git_commit.strip()}"
    atomic_write_text(provenance_dir / "git_commit.txt", git_commit_text + "\n")

    _, git_status = _run(["git", "status", "--short"], project_root)
    atomic_write_text(provenance_dir / "git_status.txt", git_status)
    _, git_diff = _run(["git", "diff", "--binary", "HEAD"], project_root)
    atomic_write_text(provenance_dir / "git_diff.patch", git_diff)

    _, pip_freeze = _run([sys.executable, "-m", "pip", "freeze"], project_root)
    atomic_write_text(provenance_dir / "environment.txt", pip_freeze)

    hardware: dict[str, Any] = {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        hardware["torch_version"] = torch.__version__
        hardware["cuda_available"] = bool(torch.cuda.is_available())
        hardware["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            hardware["visible_gpu_count"] = torch.cuda.device_count()
            hardware["visible_gpus"] = []
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                hardware["visible_gpus"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": int(properties.total_memory),
                        "compute_capability": [properties.major, properties.minor],
                    }
                )
    except Exception as exc:  # pragma: no cover - provenance must not hide primary error
        hardware["torch_probe_error"] = repr(exc)

    _, nvidia_smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        project_root,
    )
    hardware["nvidia_smi"] = nvidia_smi.strip()
    write_json(provenance_dir / "hardware.json", hardware)
    write_json(provenance_dir / "data_manifest.json", data_manifest)
    write_json(provenance_dir / "split_manifest.json", split_metadata)
