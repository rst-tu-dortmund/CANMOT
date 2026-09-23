import importlib.util
import math
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from canmot.config_resolvers import register_resolvers


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("canmot_optimize", ROOT / "src" / "optimize.py")
OPTIMIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPTIMIZE)


def _recipe(name):
    register_resolvers()
    overrides = [
        "environment_cfg.dataset_base_path=/dataset",
        "environment_cfg.detector_base_path=/detectors",
        "environment_cfg.logging_base_path=/logs",
        "environment_cfg.cache_path=/cache",
        "environment_cfg.n_processes=0",
        f"+optimization_recipe={name}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return OmegaConf.to_container(compose(config_name="config", overrides=overrides), resolve=True)


def test_paper_optimization_recipes():
    expectations = {
        "paper/local/canmot": (19, 7),
        "paper/local/canmot_sc": (10, 7),
        "paper/local/shared": (19, 1),
        "paper/global/shared": (19, 1),
    }
    for recipe_name, (parameter_count, scope_count) in expectations.items():
        cfg = _recipe(recipe_name)
        bounds = OPTIMIZE.parameter_bounds(cfg)
        assert len(bounds) == parameter_count
        assert len(OPTIMIZE.scopes(cfg["optimization_recipe"])) == scope_count
        assert cfg["optimization_recipe"]["evaluations_per_parameter"] == 10
        assert all(np.allclose(bound, [math.log(1e-4), math.log(3)]) for bound in bounds.values())


def test_canmot_sc_keeps_full_measurement_covariance_fixed():
    cfg = _recipe("paper/local/canmot_sc")
    original_r = np.asarray(cfg["model_cfg"]["matrix_r"])
    bounds = OPTIMIZE.parameter_bounds(cfg)
    midpoint = {name: sum(bound) / 2 for name, bound in bounds.items()}
    updated = OPTIMIZE.apply_parameters(cfg, midpoint, class_index=0)
    assert np.array_equal(np.asarray(updated["model_cfg"]["matrix_r"]), original_r)
    assert np.any(np.abs(original_r[0] - np.diag(np.diag(original_r[0]))) > 0)
