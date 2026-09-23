"""Seeded, resumable diagonal Bayesian optimization for CANMOT paper recipes."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import hydra
import numpy as np
from bayes_opt import BayesianOptimization
from omegaconf import DictConfig, OmegaConf

from canmot.config_resolvers import register_resolvers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test import run  # noqa: E402

register_resolvers()

CLASS_NAMES = ["bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"]


def parameter_bounds(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return one diagonal CV parameter vector for the recipe."""
    bounds: dict[str, tuple[float, float]] = {}
    recipe = cfg["optimization_recipe"]
    for matrix_name in recipe["matrices"]:
        for parameter, bound_name in cfg["bayes_opt"]["params"][matrix_name].items():
            low, high = cfg["bayes_opt"]["pbounds"][bound_name]
            bounds[f"{matrix_name}_{parameter}"] = (float(low), float(high))
    return bounds


def scopes(recipe: dict[str, Any]) -> list[tuple[str, int | None]]:
    if recipe["mode"] == "shared":
        return [("shared", None)]
    if recipe["mode"] == "class_aware":
        return [(name, index) for index, name in enumerate(CLASS_NAMES)]
    raise ValueError(f"Unknown optimization mode: {recipe['mode']}")


def apply_parameters(
    cfg: dict[str, Any], parameters: dict[str, float], class_index: int | None,
) -> dict[str, Any]:
    """Apply diagonal parameters while preserving recipe-fixed matrices."""
    recipe = cfg["optimization_recipe"]
    target_classes = range(len(CLASS_NAMES)) if class_index is None else [class_index]
    for matrix_name in recipe["matrices"]:
        matrices = [np.asarray(matrix, dtype=float).copy() for matrix in cfg["model_cfg"][f"matrix_{matrix_name}"]]
        order = cfg["bayes_opt"]["param_order"]["cv"][matrix_name]
        for index in target_classes:
            diagonal = np.diag(matrices[index]).copy()
            for parameter in cfg["bayes_opt"]["params"][matrix_name]:
                diagonal[order.index(parameter)] = math.exp(float(parameters[f"{matrix_name}_{parameter}"]))
            matrices[index] = np.diag(diagonal)
        cfg["model_cfg"][f"matrix_{matrix_name}"] = [matrix.tolist() for matrix in matrices]
    cfg["limit_classes"] = None if class_index is None else [class_index]
    cfg["eval_only"] = False
    cfg["environment_cfg"]["n_processes"] = int(recipe.get("tracking_processes", 0))
    return cfg


def objective(metrics: dict[str, Any], class_name: str) -> float:
    if class_name == "shared":
        return float(metrics["amota"])
    return float(metrics["label_metrics"]["amota"][class_name])


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_ledger(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    base = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(base, dict)
    recipe = base["optimization_recipe"]
    ledger_path = Path(recipe["ledger"])
    previous = read_ledger(ledger_path)
    bounds = parameter_bounds(base)
    evaluations_per_parameter = int(recipe["evaluations_per_parameter"])
    budget = len(bounds) * evaluations_per_parameter
    best_by_scope: dict[str, dict[str, Any]] = {}

    for scope_name, class_index in scopes(recipe):
        optimizer = BayesianOptimization(
            f=None, pbounds=bounds,
            random_state=int(recipe["seed"]) + (class_index or 0),
            allow_duplicate_points=True, verbose=0,
        )
        scope_records = [record for record in previous if record["scope"] == scope_name]
        for record in scope_records:
            optimizer.register(params=record["params"], target=record["target"])

        while len(scope_records) < budget:
            if len(scope_records) < int(recipe["initial_random_points"]):
                raw = optimizer.random_sample(1)[0]
                parameters = {name: float(value) for name, value in zip(bounds, raw)}
            else:
                parameters = {name: float(value) for name, value in optimizer.suggest().items()}
            iteration = len(scope_records)
            evaluation_cfg = apply_parameters(json.loads(json.dumps(base)), parameters, class_index)
            evaluation_root = ledger_path.parent / "evaluations" / scope_name / f"{iteration:04d}"
            evaluation_cfg["result_path"] = str(evaluation_root / "result")
            evaluation_cfg["eval_path"] = str(evaluation_root / "eval")
            metrics = run(OmegaConf.create(evaluation_cfg))
            target = objective(metrics, scope_name)
            record = {
                "recipe": recipe["name"], "scope": scope_name, "iteration": iteration,
                "params": parameters, "target": target,
            }
            append_ledger(ledger_path, record)
            optimizer.register(params=parameters, target=target)
            scope_records.append(record)
            print(f"{scope_name}: {len(scope_records)}/{budget}, AMOTA={target:.6f}")

        best_by_scope[scope_name] = max(scope_records, key=lambda item: item["target"])

    best_path = Path(recipe["best_result"])
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(best_by_scope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best parameters written to {best_path}")


if __name__ == "__main__":
    main()
