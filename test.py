"""Hydra entry point for the CANMOT paper experiments."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from canmot.config_resolvers import register_resolvers
from canmot.dataloader.cache import CacheLoader
from canmot.dataloader.nusc_loader import NuScenesloader
from canmot.reproducibility import (
    dependency_version_mismatches,
    installed_dependency_versions,
    sha256_file,
)
from canmot.tracking.nusc_tracker import Tracker
from nuscenes.eval.common.config import config_factory as tracking_config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval


register_resolvers()

PAPER_INPUT_HASHES = {
    "train": "547fdb7dd6c401749f619a504fa37ee70dbad676d3f400b28c5176bff3327145",
    "val": "4186f748f91e071c1276287244e23a01da5908a373af85ee2dfb08ea935b903e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_config(config: dict[str, Any]) -> None:
    if platform.python_version_tuple()[:2] != ("3", "10"):
        raise RuntimeError(
            f"CANMOT requires Python 3.10; active version is {platform.python_version()}"
        )
    dependency_problems = dependency_version_mismatches()
    if dependency_problems:
        raise RuntimeError(
            "Locked dependency versions are not installed:\n- " + "\n- ".join(dependency_problems)
        )
    environment = config["environment_cfg"]
    required = (
        "dataset_base_path", "detector_base_path", "logging_base_path",
        "cache_path", "n_processes",
    )
    missing = [name for name in required if name not in environment]
    if missing:
        raise ValueError(f"environment_cfg is missing required fields: {', '.join(missing)}")

    dataset_path = Path(config["dataset_cfg"]["dataset_path"])
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"nuScenes dataset directory not found: {dataset_path}")
    split = config["dataset_cfg"]["split"]
    detection_path = Path(config["detector_cfg"]["ordered_detections_path"][split])
    if not detection_path.is_file():
        raise FileNotFoundError(f"CenterPoint detections not found: {detection_path}")
    expected_hash = PAPER_INPUT_HASHES.get(split)
    if expected_hash and _sha256(detection_path) != expected_hash:
        raise ValueError(
            f"CenterPoint checksum mismatch for {split}: {detection_path}. "
            "Refusing to run a paper configuration with a different detector file."
        )
    first_tokens = Path(config["dataset_cfg"]["first_token_path"])
    if not first_tokens.is_file():
        raise FileNotFoundError(f"first-token table not found: {first_tokens}")


def track_sequence(
    result_path: str,
    sequence_id: int,
    process_count: int,
    nusc_loader: NuScenesloader,
    classes_to_evaluate: list[int] | None = None,
) -> None:
    """Run tracking for all frames, or one scene when multiprocessing."""
    result: dict[str, Any] = {
        "results": {},
        "meta": {
            "use_camera": False, "use_lidar": True, "use_radar": False,
            "use_map": False, "use_external": False,
        },
    }
    eval_config = tracking_config_factory("tracking_nips_2019")
    tracker = Tracker(config=nusc_loader.config)

    for frame_data in tqdm(nusc_loader, desc="Running", total=len(nusc_loader)):
        if process_count > 0 and frame_data["seq_id"] != sequence_id:
            continue
        if classes_to_evaluate is not None:
            selected = [
                index for index, box in enumerate(frame_data["box_dets"])
                if eval_config.class_names.index(box.name) in classes_to_evaluate
            ]
            frame_data["box_dets"] = [frame_data["box_dets"][index] for index in selected]
            frame_data["np_dets"] = np.asarray([frame_data["np_dets"][index] for index in selected])
            frame_data["np_dets_bottom_corners"] = np.asarray(
                [frame_data["np_dets_bottom_corners"][index] for index in selected]
            )
            frame_data["det_num"] = len(selected)
            frame_data["no_dets"] = not selected

        # Fail fast: partial output must never look like a valid paper run.
        tracker.tracking(frame_data)
        sample_results = []
        if "no_val_track_result" not in frame_data:
            for predicted_box in frame_data["box_track_res"]:
                box_result = {
                    "sample_token": frame_data["sample_token"],
                    "translation": [float(value) for value in predicted_box.center],
                    "size": [float(value) for value in predicted_box.wlh],
                    "rotation": [float(value) for value in predicted_box.orientation],
                    "velocity": [float(predicted_box.velocity[0]), float(predicted_box.velocity[1])],
                    "tracking_id": str(predicted_box.tracking_id),
                    "tracking_name": predicted_box.name,
                    "tracking_score": predicted_box.score,
                }
                if getattr(predicted_box, "covariance", None) is not None:
                    box_result["covariance"] = predicted_box.covariance
                    box_result["state_dim"] = len(predicted_box.covariance)
                sample_results.append(box_result)
        result["results"].setdefault(frame_data["sample_token"], []).extend(sample_results)

    for sample_token, boxes in result["results"].items():
        ranked = sorted(((-box["tracking_score"], index) for index, box in enumerate(boxes)))
        result["results"][sample_token] = [boxes[index] for _, index in ranked[:500]]

    output = Path(f"{result_path}{sequence_id}.json") if process_count > 0 else Path(result_path) / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream)


def _track_sequence_star(arguments: tuple[Any, ...]) -> None:
    track_sequence(*arguments)


def evaluate(result_path: Path, eval_path: Path, dataset_path: Path, split: str) -> dict[str, Any]:
    evaluator = TrackingEval(
        config=tracking_config_factory("tracking_nips_2019"),
        result_path=str(result_path), eval_set=split, output_dir=str(eval_path),
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=str(dataset_path),
    )
    return evaluator.main()


def run(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Resolve, validate, track, and evaluate one Hydra experiment."""
    resolved = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    assert isinstance(resolved, dict)
    validate_config(resolved)

    split = resolved["dataset_cfg"]["split"]
    detection_path = Path(resolved["detector_cfg"]["ordered_detections_path"][split])
    result_path = Path(resolved["result_path"]) / split
    eval_path = Path(resolved["eval_path"]) / split
    eval_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)

    with (eval_path / "resolved_config.json").open("w", encoding="utf-8") as stream:
        json.dump(resolved, stream, indent=2)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": _git_revision(Path(__file__).resolve().parent),
        "nuscenes_devkit_revision": "656cf5636f5afbe0a1eaa425b3d2dde664b171c5",
        "detection_path": str(detection_path),
        "detection_sha256": _sha256(detection_path),
        "python": platform.python_version(),
        "requirements_lock_sha256": sha256_file(
            Path(__file__).resolve().parent / "requirements-lock.txt"
        ),
        "dependencies": installed_dependency_versions(),
    }
    with (eval_path / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    if not resolved.get("eval_only", False):
        loader = CacheLoader(
            NuScenesloader(
                str(detection_path), resolved["dataset_cfg"]["first_token_path"], resolved,
            ),
            cache_path=resolved["environment_cfg"]["cache_path"],
        )
        process_count = int(resolved["environment_cfg"]["n_processes"])
        if process_count > 0:
            temporary = result_path / "temp_result"
            temporary.mkdir(parents=True, exist_ok=True)
            loader.build_cache(n_processes=process_count)
            arguments = [
                (f"{temporary}{os.sep}", sequence_id, process_count, loader, resolved.get("limit_classes"))
                for sequence_id in range(loader.max_seq_id + 1)
            ]
            with multiprocessing.Pool(process_count) as pool:
                pool.map(_track_sequence_star, arguments)
            combined: dict[str, Any] = {"results": {}, "meta": {}}
            for sequence_id in range(loader.max_seq_id + 1):
                with (temporary / f"{sequence_id}.json").open(encoding="utf-8") as stream:
                    partial = json.load(stream)
                combined["results"].update(partial["results"])
                combined["meta"].update(partial["meta"])
            with (result_path / "results.json").open("w", encoding="utf-8") as stream:
                json.dump(combined, stream)
        else:
            loader.build_cache(n_processes=0)
            track_sequence(str(result_path), 0, 0, loader, resolved.get("limit_classes"))

    return evaluate(
        result_path / "results.json", eval_path,
        Path(resolved["dataset_cfg"]["dataset_path"]), split,
    )


@hydra.main(config_path="config", config_name="config", version_base=None)
def hydra_entry(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    hydra_entry()
