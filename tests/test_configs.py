from pathlib import Path
import subprocess

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml

from canmot.config_resolvers import register_resolvers
from canmot.reproducibility import load_manifest


def test_all_paper_configs_compose():
    register_resolvers()
    config_dir = Path(__file__).resolve().parents[1] / "config"
    overrides = [
        "environment_cfg.dataset_base_path=/dataset",
        "environment_cfg.detector_base_path=/detectors",
        "environment_cfg.logging_base_path=/logs",
        "environment_cfg.cache_path=/cache",
        "environment_cfg.n_processes=0",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for experiment in load_manifest()["experiments"]:
            cfg = compose(config_name="config", overrides=[*overrides, f"+experiment={experiment}"])
            resolved = OmegaConf.to_container(cfg, resolve=True)
            assert resolved["experiment_id"] == experiment
            assert resolved["dataset_cfg"]["split"] == "val"
            assert "???" not in OmegaConf.to_yaml(cfg, resolve=True)


def test_only_environment_template_is_versionable():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    folder = "config/environment_cfg"
    tracked = subprocess.run(
        ["git", "ls-files", folder], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.split()
    assert tracked == [f"{folder}/base_environment.yaml"]
    # Local machine configs must be ignored, whether or not one exists yet.
    ignored = subprocess.run(["git", "check-ignore", "-q", f"{folder}/my_machine.yaml"], cwd=root)
    assert ignored.returncode == 0


def test_hydra_does_not_create_prevalidation_output_directories():
    path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["hydra"]["run"]["dir"] == "."
    assert config["hydra"]["output_subdir"] is None
    assert config["hydra"]["job"]["chdir"] is False
