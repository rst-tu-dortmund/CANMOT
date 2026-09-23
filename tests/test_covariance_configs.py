from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _matrices(relative: str):
    return yaml.safe_load((ROOT / relative).read_text())


def test_empirical_covariance_keeps_off_diagonal_terms():
    matrices = _matrices("config/model_cfg/matrix_q/centerpoint/local/trainval/CV_sample_cov.yaml")
    assert any(np.any(np.abs(np.asarray(matrix) - np.diag(np.diag(matrix))) > 0) for matrix in matrices)


def test_optimized_covariance_is_diagonal():
    matrices = _matrices("config/model_cfg/matrix_q/centerpoint/local/canmot_qr.yaml")
    for matrix in matrices:
        array = np.asarray(matrix)
        assert np.allclose(array, np.diag(np.diag(array)))
