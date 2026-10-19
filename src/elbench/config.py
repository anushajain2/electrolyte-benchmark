"""Configuration loading, hashing and per-command run manifests."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    path: Path
    data: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def p(self, key: str) -> Path:
        """Resolve a configured path relative to the project root."""
        return (self.root / self.data["paths"][key]).resolve()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def load_config(path: str | os.PathLike, root: str | os.PathLike | None = None) -> Config:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    # Project root: the directory that holds configs/ unless overridden.
    root_path = Path(root).resolve() if root else path.parent.parent
    return Config(path=path, data=data, root=root_path)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def md5_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def peak_rss_mb() -> float:
    """Process peak resident set size in MB (macOS reports bytes, Linux KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1 << 20) if sys.platform == "darwin" else rss / 1024


def _package_versions() -> dict[str, str]:
    out = {}
    for name in ["numpy", "pandas", "sklearn", "scipy", "rdkit", "pyarrow"]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional dependency missing
            out[name] = "missing"
    return out


@dataclass
class RunManifest:
    """Machine-readable record written by every command."""

    command: str
    config: Config
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6])
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.time)
    _c0: float = field(default_factory=time.process_time)

    def add_input(self, path: Path) -> None:
        self.inputs[str(Path(path).resolve().relative_to(self.config.root))] = sha256_file(path)

    def add_output(self, path: Path) -> None:
        self.outputs[str(Path(path).resolve().relative_to(self.config.root))] = sha256_file(path)

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        from . import __version__

        record = {
            "run_id": self.run_id,
            "command": self.command,
            "package_version": __version__,
            "benchmark_version": self.config["benchmark_version"],
            "config_path": str(self.config.path.relative_to(self.config.root)),
            "config_sha256": self.config.sha256,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._t0)),
            "wall_seconds": round(time.time() - self._t0, 3),
            "cpu_seconds": round(time.process_time() - self._c0, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "notes": self.notes,
        }
        out = directory / f"run_{self.command}_{self.run_id}.json"
        out.write_text(json.dumps(record, indent=2, default=str))
        return out
