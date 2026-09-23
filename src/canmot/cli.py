"""Small convenience CLI over the canonical Hydra interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from canmot.config_resolvers import register_resolvers
from canmot.reproducibility import (
    ROOT,
    compare_calibration,
    compare_metrics,
    dependency_version_mismatches,
    load_json,
    load_manifest,
    metrics_file,
    sha256_file,
)


def _compose(environment: str, experiment: str | None = None):
    register_resolvers()
    overrides = [f"environment_cfg={environment}"]
    if experiment:
        overrides.append(f"+experiment={experiment}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return compose(config_name="config", overrides=overrides)


def _run(experiment: str, environment: str, extra: list[str] | None = None) -> None:
    command = [sys.executable, str(ROOT / "test.py"), f"environment_cfg={environment}", f"+experiment={experiment}"]
    command.extend(extra or [])
    subprocess.run(command, cwd=ROOT, check=True)


def preflight(environment: str) -> None:
    cfg = OmegaConf.to_container(_compose(environment), resolve=True)
    env = cfg["environment_cfg"]
    dataset = Path(env["dataset_base_path"]) / "nuscenes"
    detector_root = Path(env["detector_base_path"])
    logging_root = Path(env["logging_base_path"])
    cache = Path(env["cache_path"])
    problems: list[str] = []
    if sys.version_info[:2] != (3, 10):
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is active; Python 3.10 is required"
        )
    problems.extend(
        f"dependency version mismatch: {item}"
        for item in dependency_version_mismatches()
    )
    devkit = ROOT / "dependencies_repos" / "nuscenes-devkit"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=devkit, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        expected_revision = load_manifest()["nuscenes_devkit_revision"]
        if revision != expected_revision:
            problems.append(f"nuScenes devkit revision is {revision}; expected {expected_revision}")
    except (OSError, subprocess.CalledProcessError):
        problems.append("pinned nuScenes devkit submodule is missing; run git submodule update --init")
    if not dataset.is_dir():
        problems.append(f"nuScenes directory is missing: {dataset}")
    manifest = load_manifest()
    for item in manifest["inputs"].values():
        path = detector_root / item["filename"]
        if not path.is_file():
            problems.append(f"detector input is missing: {path}")
        elif sha256_file(path) != item["sha256"]:
            problems.append(f"detector checksum mismatch: {path}")
    for path in (logging_root, cache):
        parent = path if path.exists() else path.parent
        if not parent.exists():
            problems.append(f"parent directory is missing: {parent}")
        elif not parent.is_dir():
            problems.append(f"output location is not a directory: {parent}")
        elif not os.access(parent, os.W_OK):
            problems.append(f"output location is not writable: {parent}")
        elif shutil.disk_usage(parent).free < 10 * 1024**3:
            problems.append(f"less than 10 GiB free at {parent}")
    if problems:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(problems))
    print("Preflight passed: paths, detector hashes, and disk space are valid.")


def verify(path: Path) -> None:
    actual_path = metrics_file(path)
    actual = load_json(actual_path)
    config_candidates = list(actual_path.parent.rglob("resolved_config.json")) + list(actual_path.parent.parent.rglob("resolved_config.json"))
    experiment = None
    for candidate in config_candidates:
        experiment = load_json(candidate).get("experiment_id")
        if experiment:
            break
    if not experiment:
        raise SystemExit("Cannot identify experiment: resolved_config.json with experiment_id is missing")
    entry = load_manifest()["experiments"].get(experiment)
    if not entry:
        raise SystemExit(f"Experiment is not in paper manifest: {experiment}")
    reference = load_json(ROOT / "reproducibility" / entry["reference"])
    failures = compare_metrics(actual, entry["expected"])
    failures.extend(compare_calibration(actual, reference))
    if failures:
        raise SystemExit("Verification failed:\n- " + "\n- ".join(failures))
    print(f"PASS {experiment}: Table I and class-wise Table II quantities match paper precision.")


def tables(environment: str) -> None:
    manifest = load_manifest()
    for experiment in manifest["experiments"]:
        _run(experiment, environment)
    cfg = OmegaConf.to_container(_compose(environment), resolve=True)
    output = Path(cfg["environment_cfg"]["logging_base_path"]) / "paper"
    lines = ["| Experiment | AMOTA | AMOTP | IDS | FRAG |", "|---|---:|---:|---:|---:|"]
    for experiment, entry in manifest["experiments"].items():
        result = load_json(metrics_file(output / experiment.removeprefix("paper/")))
        lines.append(f"| `{experiment}` | {result['amota']:.3f} | {result['amotp']:.3f} | {int(result['ids'])} | {int(result['frag'])} |")
    (output / "table_i.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "table_i.md")


def figures(environment: str) -> None:
    cfg = OmegaConf.to_container(_compose(environment), resolve=True)
    output = Path(cfg["environment_cfg"]["logging_base_path"]) / "paper" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "canmot.utils.covariance_viz",
        "--local-r", "config/model_cfg/matrix_r/centerpoint/local/trainval/sample_cov.yaml",
        "--global-r", "config/model_cfg/matrix_r/centerpoint/global/trainval/sample_cov.yaml",
        "--local-q", "config/model_cfg/matrix_q/centerpoint/local/trainval/CV_sample_cov.yaml",
        "--global-q", "config/model_cfg/matrix_q/centerpoint/global/trainval/CV_sample_cov.yaml",
        "--output", str(output / "figure_1_covariances")]
    subprocess.run(command, cwd=ROOT, check=True)
    for experiment, label in (
        ("paper/local/canmot/sc", "canmot_sc"),
        ("paper/global/sample/trainval", "global_sample_trainval"),
        ("paper/local/canmot/opt", "canmot_opt"),
        ("paper/baseline/poly_mot", "poly_mot"),
    ):
        _run(experiment, environment, [
            "visualization.enable=true", "visualization.sequence_ids=[144]",
            "visualization.every_n_steps=35",
            f"visualization.output_subdir={output / 'figure_2_scene_144' / label}",
        ])
    print(f"Figures 1 and 2 generated below {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="canmot")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--environment", required=True)
    p = sub.add_parser("run"); p.add_argument("experiment"); p.add_argument("--environment", required=True)
    p = sub.add_parser("reproduce"); p.add_argument("target", choices=["tables"]); p.add_argument("--environment", required=True)
    p = sub.add_parser("verify"); p.add_argument("output_directory", type=Path)
    p = sub.add_parser("estimate-covariances"); p.add_argument("--environment", required=True)
    p = sub.add_parser("optimize"); p.add_argument("recipe"); p.add_argument("--environment", required=True)
    p = sub.add_parser("figures"); p.add_argument("--environment", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight": preflight(args.environment)
    elif args.command == "run": _run(args.experiment, args.environment)
    elif args.command == "reproduce": tables(args.environment)
    elif args.command == "verify": verify(args.output_directory)
    elif args.command == "estimate-covariances":
        cfg = OmegaConf.to_container(_compose(args.environment), resolve=True)
        covariance_root = Path(cfg["environment_cfg"]["logging_base_path"]) / "covariance_estimation"
        for split in ("train", "val", "trainval"):
            subprocess.run([sys.executable, "-m", "canmot.data.script.compute_sample_covariance_nusc", f"environment_cfg={args.environment}", f"dataset_cfg.split={split}", f"+covariance_output_path={covariance_root / split}"], cwd=ROOT, check=True)
    elif args.command == "optimize":
        subprocess.run([sys.executable, str(ROOT / "src" / "optimize.py"), f"environment_cfg={args.environment}", f"+optimization_recipe={args.recipe}"], cwd=ROOT, check=True)
    elif args.command == "figures": figures(args.environment)


if __name__ == "__main__":
    main()
