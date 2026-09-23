"""Manifest, hashing, preflight, and paper-metric verification helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "reproducibility" / "paper_manifest.yaml"
LOCK_PATH = ROOT / "requirements-lock.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_requirements(path: Path = LOCK_PATH) -> dict[str, str]:
    """Read the exact, marker-free Python 3.10 release lock."""
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"Unpinned requirement in {path}: {line}")
        name, version = line.split("==", 1)
        requirements[name] = version
    return requirements


def installed_dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in locked_requirements():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def dependency_version_mismatches() -> list[str]:
    expected = locked_requirements()
    installed = installed_dependency_versions()
    return [
        f"{name}: expected {version}, installed {installed[name] or 'missing'}"
        for name, version in expected.items()
        if installed[name] != version
    ]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if manifest.get("schema_version") != 1 or len(manifest.get("experiments", {})) != 11:
        raise ValueError(f"Invalid paper manifest: {path}")
    return manifest


def metrics_file(path: Path) -> Path:
    if path.is_file():
        return path
    matches = list(path.rglob("metrics_summary.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one metrics_summary.json below {path}, found {len(matches)}")
    return matches[0]


def compare_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("ids", "frag", "tp", "fp"):
        if int(actual[key]) != int(expected[key]):
            failures.append(f"{key}: expected {int(expected[key])}, got {int(actual[key])}")
    for key in ("amota", "amotp"):
        if f"{float(actual[key]):.3f}" != f"{float(expected[key]):.3f}":
            failures.append(
                f"{key}: expected {float(expected[key]):.3f}, got {float(actual[key]):.3f}"
            )
    return failures


def compare_calibration(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Compare the class-wise Table II quantities at displayed precision."""
    failures: list[str] = []
    actual_labels = actual.get("label_metrics", {})
    expected_labels = expected.get("label_metrics", {})
    for metric in ("nees_mean", "pct_inside", "pct_outside"):
        for class_name, wanted in expected_labels.get(metric, {}).items():
            got = actual_labels.get(metric, {}).get(class_name)
            if got is None or f"{float(got):.3f}" != f"{float(wanted):.3f}":
                failures.append(f"{metric}.{class_name}: expected {float(wanted):.3f}, got {got}")
    for class_name, wanted in expected_labels.get("chi2_significant", {}).items():
        got = actual_labels.get("chi2_significant", {}).get(class_name)
        if bool(got) != bool(wanted):
            failures.append(f"chi2_significant.{class_name}: expected {wanted}, got {got}")
    return failures


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
