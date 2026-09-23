import numpy as np

from canmot.motion_module.motion_model import CV, get_model_class_by_name
from canmot.utils.matching import Hungarian


def test_cv_prediction_and_rotation():
    model = CV(has_velo=True, dt=0.5)
    state = np.zeros((10, 1))
    state[6, 0] = 4.0
    predicted = model.getTransitionF(state) @ state
    assert predicted[0, 0] == 2.0
    state[-1, 0] = np.pi / 2
    rotation = model.get_rotation_matrix_for_state(state)
    covariance = np.diag([4.0, 1.0] + [1.0] * 8)
    rotated = rotation @ covariance @ rotation.T
    assert np.allclose(rotated[:2, :2], np.diag([1.0, 4.0]), atol=1e-12)


def test_hungarian_tracker_path():
    matched_dets, matched_tracks, unmatched_dets, unmatched_tracks = Hungarian(
        np.array([[0.1, 2.0], [2.0, 0.2]]), thresholds={0: 1.0}
    )
    assert matched_dets == [0, 1]
    assert matched_tracks == [0, 1]
    assert unmatched_dets.size == unmatched_tracks.size == 0


def test_only_paper_motion_models_are_public():
    for name in ("CV", "CA", "CTRA"):
        assert get_model_class_by_name(name).__name__ == name
