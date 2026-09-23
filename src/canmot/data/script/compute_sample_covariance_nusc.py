"""Compute class-wise sample covariance matrices for NuScenes detections and annotations.

This script estimates:
- Measurement noise R from matched detection-vs-GT residuals in measurement space.
- Process noise Q from one-step GT prediction residuals in CV full-state space.

Matching follows Poly-MOT association settings from the composed Hydra config.
"""

import os
import json
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from tqdm import tqdm

import hydra
import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nusc_splits
from scipy.spatial.transform import Rotation as R

from canmot.config_resolvers import register_resolvers

from canmot.data.script.NUSC_CONSTANT import CLASS_SEG_TO_STR_CLASS
from canmot.pre_processing import dictdet2array, arraydet2box, blend_nms
from canmot.utils.matching import Hungarian
from canmot.utils.script import mask_tras_dets, fast_compute_check, reorder_metrics, spec_metric_mask
from canmot.geometry.nusc_distance import (
    iou_bev,
    iou_3d,
    giou_bev,
    giou_3d,
)

register_resolvers()

TRACKING_CLASS_MAPPING = {
    "animal": None,
    "human.pedestrian.personal_mobility": None,
    "human.pedestrian.stroller": None,
    "human.pedestrian.wheelchair": None,
    "movable_object.barrier": None,
    "movable_object.debris": None,
    "movable_object.pushable_pullable": None,
    "movable_object.trafficcone": None,
    "static_object.bicycle_rack": None,
    "vehicle.emergency.ambulance": None,
    "vehicle.emergency.police": None,
    "vehicle.construction": None,
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

CLASS_ID_TO_NAME = {v: k for k, v in CLASS_SEG_TO_STR_CLASS.items()}
MATCHING_ALGOS = {"Hungarian": Hungarian}
METRIC_TWO_RET = {"iou_3d", "giou_3d"}
DT_KEYFRAME = 0.5  # NuScenes keyframe rate is 2Hz.
EPS = 1e-12


def wrap_angle_to_pi(arr: np.ndarray, idx: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return arr
    arr[:, idx] = (arr[:, idx] + np.pi) % (2 * np.pi) - np.pi
    return arr


def to_plain_dict(d: Dict) -> Dict:
    return {int(k): v for k, v in d.items()}


def to_threshold_dict(d: Dict, cls_num: int) -> Dict[int, float]:
    out = {int(k): float(v) for k, v in d.items()}
    missing = [k for k in range(cls_num) if k not in out]
    if missing:
        raise ValueError(f"Missing thresholds for class ids: {missing}")
    return out


def unique_filepath(directory: str, filename: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    stem, ext = os.path.splitext(filename)
    n = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{n}{ext}")
        n += 1
    return path


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_detections_for_split(detector_cfg, split: str) -> Dict:
    split_paths = detector_cfg["ordered_detections_path"]
    if not isinstance(split_paths, (dict, DictConfig)):
        raise ValueError("detector_cfg.ordered_detections_path must contain train and val")

    def load_results(split_name: str) -> Dict:
        if split_name not in split_paths:
            raise ValueError(f"Missing ordered_detections_path['{split_name}']")
        data = load_json(split_paths[split_name])
        if "results" not in data:
            raise ValueError(f"Invalid detector file for split '{split_name}': missing 'results'")
        return data["results"]

    if split == "trainval":
        merged = {}
        merged.update(load_results("train"))
        merged.update(load_results("val"))
        return merged

    if split not in {"train", "val"}:
        raise ValueError("split must be one of train/val/trainval")

    return load_results(split)


def get_version_for_split(split: str) -> str:
    return "v1.0-trainval"


def scene_names_for_split(split: str) -> set:
    sp = nusc_splits.create_splits_scenes()
    if split == "trainval":
        return set(sp["train"] + sp["val"])
    return set(sp[split])


def preprocess_detections(raw_sample_dets: List[Dict], cfg) -> Dict:
    sf_thre = to_threshold_dict(cfg["preprocessing"]["SF_thre"], cfg["basic"]["CLASS_NUM"])
    nms_thre = float(cfg["preprocessing"]["NMS_thre"])
    nms_type = cfg["preprocessing"]["NMS_type"]
    nms_metric = cfg["preprocessing"]["NMS_metric"]

    list_dets, _ = dictdet2array(
        raw_sample_dets,
        "translation",
        "size",
        "velocity",
        "rotation",
        "detection_score",
        "detection_name",
    )

    if len(list_dets) == 0:
        return {
            "np_dets": np.zeros((0, 14), dtype=float),
            "np_dets_bottom_corners": np.zeros((0, 4, 2), dtype=float),
            "box_dets": np.zeros((0,), dtype=object),
            "det_num": 0,
            "no_dets": True,
        }

    np_dets = np.array([det for det in list_dets if det[-2] > sf_thre[int(det[-1])]], dtype=float)
    if np_dets.shape[0] == 0:
        return {
            "np_dets": np.zeros((0, 14), dtype=float),
            "np_dets_bottom_corners": np.zeros((0, 4, 2), dtype=float),
            "box_dets": np.zeros((0,), dtype=object),
            "det_num": 0,
            "no_dets": True,
        }

    box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
    tmp_infos = {"np_dets": np_dets, "np_dets_bottom_corners": np_dets_bottom_corners}
    keep = globals()[nms_type](box_infos=tmp_infos, metrics=nms_metric, thre=nms_thre)

    if len(keep) == 0:
        return {
            "np_dets": np.zeros((0, 14), dtype=float),
            "np_dets_bottom_corners": np.zeros((0, 4, 2), dtype=float),
            "box_dets": np.zeros((0,), dtype=object),
            "det_num": 0,
            "no_dets": True,
        }

    return {
        "np_dets": np_dets[keep],
        "np_dets_bottom_corners": np_dets_bottom_corners[keep],
        "box_dets": box_dets[keep],
        "det_num": len(keep),
        "no_dets": False,
    }


def extract_gt_for_sample(nusc: NuScenes, sample: Dict) -> Dict:
    det_like_rows = []
    states_meas = []
    states_full = []
    quats = []
    class_ids = []
    instance_tokens = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls_name = TRACKING_CLASS_MAPPING.get(ann["category_name"], None)
        if cls_name is None:
            continue

        box = nusc.get_box(ann_token)
        vel = nusc.box_velocity(ann_token)
        if np.isnan(vel).any():
            continue

        class_id = CLASS_SEG_TO_STR_CLASS[cls_name]
        quat_wxyz = box.orientation.elements
        yaw = box.orientation.yaw_pitch_roll[0]

        det_like_rows.append(
            [
                box.center[0],
                box.center[1],
                box.center[2],
                box.wlh[0],
                box.wlh[1],
                box.wlh[2],
                vel[0],
                vel[1],
                quat_wxyz[0],
                quat_wxyz[1],
                quat_wxyz[2],
                quat_wxyz[3],
                1.0,
                class_id,
            ]
        )
        states_meas.append([box.center[0], box.center[1], box.center[2], box.wlh[0], box.wlh[1], box.wlh[2], vel[0], vel[1], yaw])
        states_full.append([box.center[0], box.center[1], box.center[2], box.wlh[0], box.wlh[1], box.wlh[2], vel[0], vel[1], vel[2], yaw])
        quats.append(quat_wxyz)
        class_ids.append(class_id)
        instance_tokens.append(ann["instance_token"])

    if len(det_like_rows) == 0:
        return {
            "np_dets": np.zeros((0, 14), dtype=float),
            "np_dets_bottom_corners": np.zeros((0, 4, 2), dtype=float),
            "box_dets": np.zeros((0,), dtype=object),
            "states_meas": np.zeros((0, 9), dtype=float),
            "states_full": np.zeros((0, 10), dtype=float),
            "quats": np.zeros((0, 4), dtype=float),
            "class_ids": np.zeros((0,), dtype=int),
            "instance_tokens": np.zeros((0,), dtype=object),
            "no_gts": True,
        }

    np_dets = np.array(det_like_rows, dtype=float)
    box_dets, bottoms = arraydet2box(np_dets)
    return {
        "np_dets": np_dets,
        "np_dets_bottom_corners": bottoms,
        "box_dets": box_dets,
        "states_meas": np.array(states_meas, dtype=float),
        "states_full": np.array(states_full, dtype=float),
        "quats": np.array(quats, dtype=float),
        "class_ids": np.array(class_ids, dtype=int),
        "instance_tokens": np.array(instance_tokens, dtype=object),
        "no_gts": False,
    }


def build_cost_matrices(det_infos: Dict, gt_infos: Dict, cfg) -> Dict[str, np.ndarray]:
    cls_num = int(cfg["basic"]["CLASS_NUM"])
    det_labels = det_infos["np_dets"][:, -1].astype(int)
    gt_labels = gt_infos["np_dets"][:, -1].astype(int)

    valid_mask = mask_tras_dets(cls_num, det_labels, gt_labels)
    metric_map = to_plain_dict(cfg["association"]["category_metrics"])
    second_metric = str(cfg["association"]["second_metric"])
    re_metrics = reorder_metrics(metric_map)
    fast = fast_compute_check(metric_map, second_metric)

    tra_cost_infos = {
        "np_dets": gt_infos["np_dets"],
        "np_dets_bottom_corners": gt_infos["np_dets_bottom_corners"],
        "states_meas": None,
        "innovation_information": None,
    }

    if fast:
        two_cost, first_cost_single = giou_3d(det_infos, tra_cost_infos)
        first_cost = first_cost_single[None, :, :].repeat(cls_num, axis=0)
        if "giou_bev" in re_metrics:
            first_cost[re_metrics["giou_bev"]] = two_cost
    else:
        metric_fn = globals()[second_metric]
        two_raw = metric_fn(det_infos, tra_cost_infos)
        two_cost = two_raw[1] if second_metric in METRIC_TWO_RET else two_raw

        first_cost = np.zeros((cls_num, det_infos["det_num"], gt_infos["np_dets"].shape[0]), dtype=float)
        for metric, cls_list in re_metrics.items():
            tra_cost_infos["mask"] = spec_metric_mask(cls_list, det_labels, gt_labels)
            metric_fn = globals()[metric]
            raw = metric_fn(det_infos, tra_cost_infos)
            cost = raw[1] if metric in METRIC_TWO_RET else raw
            first_cost[cls_list] = cost

    invert_first = bool(cfg.get("association", {}).get("invert_first_metric", True))
    invert_second = bool(cfg.get("association", {}).get("invert_second_metric", True))

    first_cost[np.where(~valid_mask)] = -np.inf if invert_first else np.inf
    first_cost = (1 - first_cost) if invert_first else first_cost

    if two_cost is not None:
        two_cost = (1 - two_cost) if invert_second else two_cost

    return {"one_stage": first_cost, "two_stage": two_cost}


def run_matching(det_infos: Dict, gt_infos: Dict, cfg) -> Tuple[np.ndarray, np.ndarray]:
    if det_infos["det_num"] == 0 or gt_infos["np_dets"].shape[0] == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    f_thre = to_threshold_dict(cfg["association"]["first_thre"], int(cfg["basic"]["CLASS_NUM"]))
    s_thre = to_threshold_dict(cfg["association"]["second_thre"], 1)
    algorithm = str(cfg["association"]["algorithm"])
    two_stage = bool(cfg["association"]["two_stage"])

    cost_mats = build_cost_matrices(det_infos, gt_infos, cfg)
    cost1, cost2 = cost_mats["one_stage"], cost_mats["two_stage"]

    matcher = MATCHING_ALGOS[algorithm]
    m_dets_1, m_gts_1, um_dets_1, um_gts_1 = matcher(cost1, f_thre)

    if two_stage:
        inf_cost = np.ones_like(cost2) * np.inf
        inf_cost[np.ix_(um_dets_1, um_gts_1)] = 0
        cost2 = cost2 + inf_cost
        m_dets_2, m_gts_2, _, _ = matcher(cost2, s_thre)
        m_dets_1 += m_dets_2
        m_gts_1 += m_gts_2

    return np.array(m_dets_1, dtype=int), np.array(m_gts_1, dtype=int)


def measurement_errors(matched_det_meas: np.ndarray, matched_gt_meas: np.ndarray, matched_gt_quats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if matched_det_meas.shape[0] == 0:
        return np.zeros((0, 9), dtype=float), np.zeros((0, 9), dtype=float)

    g_err = matched_det_meas - matched_gt_meas
    g_err = wrap_angle_to_pi(g_err, 8)

    l_err = g_err.copy()
    rinv = R.from_quat(matched_gt_quats, scalar_first=True).inv()

    l_err[:, :3] = rinv.apply(g_err[:, :3]) # x y z
    vel_g = np.zeros((g_err.shape[0], 3), dtype=float)
    vel_g[:, :2] = g_err[:, 6:8]
    vel_l = rinv.apply(vel_g)
    l_err[:, 6:8] = vel_l[:, :2]

    return g_err, l_err


def process_errors_cv(curr_states: np.ndarray, next_states: np.ndarray, curr_quats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if curr_states.shape[0] == 0:
        return np.zeros((0, 10), dtype=float), np.zeros((0, 10), dtype=float)

    pred = curr_states.copy()
    pred[:, :3] += pred[:, 6:9] * DT_KEYFRAME

    g_err = pred - next_states
    g_err = wrap_angle_to_pi(g_err, 9)

    l_err = g_err.copy()
    rinv = R.from_quat(curr_quats, scalar_first=True).inv()
    l_err[:, :3] = rinv.apply(g_err[:, :3])
    l_err[:, 6:9] = rinv.apply(g_err[:, 6:9])

    return g_err, l_err


def covariance_or_identity(samples: np.ndarray, dim: int) -> np.ndarray:
    if samples.shape[0] < 2:
        return np.eye(dim, dtype=float)
    cov = np.cov(samples, rowvar=False)
    cov = np.atleast_2d(cov)
    if cov.shape != (dim, dim) or not np.isfinite(cov).all():
        return np.eye(dim, dtype=float)
    cov = 0.5 * (cov + cov.T)
    return cov


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.diag(cov)
    std = np.sqrt(np.maximum(d, EPS))
    corr = cov / (std[:, None] * std[None, :])
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def write_yaml(data: Dict, directory: str, filename: str) -> str:
    path = unique_filepath(directory, filename)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def payload_to_cov_list(payload: Dict) -> List:
    classes = len(CLASS_SEG_TO_STR_CLASS)
    cov_list = []
    for cid in range(classes):
        cnm = CLASS_ID_TO_NAME[cid]
        if cnm not in payload["classes"]:
            raise ValueError(f"Missing class_id {cid} in payload classes")
        cov_list.append(
            payload["classes"][cnm]["covariance"]
        )

    return cov_list


def extract_next_pairs(curr_gt: Dict, next_gt: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if curr_gt["no_gts"] or next_gt["no_gts"]:
        return (
            np.zeros((0, 10), dtype=float),
            np.zeros((0, 10), dtype=float),
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    next_idx = {tok: idx for idx, tok in enumerate(next_gt["instance_tokens"]) }
    curr_indices = []
    next_indices = []
    for idx, tok in enumerate(curr_gt["instance_tokens"]):
        if tok in next_idx:
            curr_indices.append(idx)
            next_indices.append(next_idx[tok])

    if len(curr_indices) == 0:
        return (
            np.zeros((0, 10), dtype=float),
            np.zeros((0, 10), dtype=float),
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    curr_indices = np.array(curr_indices, dtype=int)
    next_indices = np.array(next_indices, dtype=int)
    return (
        curr_gt["states_full"][curr_indices],
        next_gt["states_full"][next_indices],
        curr_gt["quats"][curr_indices],
        curr_gt["class_ids"][curr_indices],
        curr_gt["instance_tokens"][curr_indices],
    )


def compute_sample_covariance_nusc(cfg) -> None:
    split = str(cfg["dataset_cfg"]["split"])
    if split not in {"train", "val", "trainval"}:
        raise ValueError("dataset_cfg.split must be one of train/val/trainval")

    model_names = {str(v).upper() for v in cfg["motion_model"]["model"].values()}
    if model_names != {"CV"}:
        raise NotImplementedError("Only CV motion model is implemented in compute_sample_covariance_nusc.py")

    det_results = load_detections_for_split(cfg["detector_cfg"], split)

    nusc = NuScenes(
        version=get_version_for_split(split),
        dataroot=cfg["dataset_cfg"]["dataset_path"],
        verbose=True,
    )

    allowed_scenes = scene_names_for_split(split)
    class_ids = sorted(CLASS_ID_TO_NAME.keys())

    agg = {
        c: {
            "r_global": [],
            "r_local": [],
            "q_global": [],
            "q_local": [],
            "q_instances": [],
        }
        for c in class_ids
    }

    tokens = list(det_results.keys())
    for sample_token in tqdm(tokens, desc="Processing samples", unit="sample"):
        sample = nusc.get("sample", sample_token)
        scene = nusc.get("scene", sample["scene_token"])
        if scene["name"] not in allowed_scenes:
            continue

        det_infos = preprocess_detections(det_results[sample_token], cfg)
        gt_infos = extract_gt_for_sample(nusc, sample)

        # Measurement residuals for R.
        if not det_infos["no_dets"] and not gt_infos["no_gts"]:
            m_det, m_gt = run_matching(det_infos, gt_infos, cfg)
            if m_det.shape[0] > 0:
                det_meas = np.concatenate(
                    [
                        det_infos["np_dets"][:, :8],
                        np.array([box.yaw for box in det_infos["box_dets"]], dtype=float)[:, None],
                    ],
                    axis=1,
                )
                matched_det_meas = det_meas[m_det]
                matched_gt_meas = gt_infos["states_meas"][m_gt]
                matched_gt_quats = gt_infos["quats"][m_gt]
                matched_det_classes = det_infos["np_dets"][m_det, -1].astype(int)

                r_g, r_l = measurement_errors(matched_det_meas, matched_gt_meas, matched_gt_quats)
                for c in np.unique(matched_det_classes):
                    mask = matched_det_classes == c
                    agg[c]["r_global"].append(r_g[mask])
                    agg[c]["r_local"].append(r_l[mask])

        # Process residuals for Q from GT one-step CV prediction.
        if sample["next"]:
            next_sample = nusc.get("sample", sample["next"])
            next_gt_infos = extract_gt_for_sample(nusc, next_sample)
            curr_states, next_states, curr_quats, q_classes, q_instances = extract_next_pairs(gt_infos, next_gt_infos)
            if curr_states.shape[0] > 0:
                q_g, q_l = process_errors_cv(curr_states, next_states, curr_quats)
                for c in np.unique(q_classes):
                    mask = q_classes == c
                    agg[c]["q_global"].append(q_g[mask])
                    agg[c]["q_local"].append(q_l[mask])
                    agg[c]["q_instances"].append(q_instances[mask])

    r_global_payload = {
        "split": split,
        "model_name": cfg["detector_cfg"]["model_name"],
        "state": "measurement_state_9d_[x,y,z,w,l,h,vx,vy,yaw]",
        "frame": "global",
        "classes": {},
    }
    r_local_payload = {
        "split": split,
        "model_name": cfg["detector_cfg"]["model_name"],
        "state": "measurement_state_9d_[x,y,z,w,l,h,vx,vy,yaw]",
        "frame": "local_annotation",
        "classes": {},
    }
    q_global_payload = {
        "split": split,
        "model_name": cfg["detector_cfg"]["model_name"],
        "motion_model": "CV",
        "dt": DT_KEYFRAME,
        "state": "cv_state_10d_[x,y,z,w,l,h,vx,vy,vz,yaw]",
        "frame": "global",
        "classes": {},
    }
    q_local_payload = {
        "split": split,
        "model_name": cfg["detector_cfg"]["model_name"],
        "motion_model": "CV",
        "dt": DT_KEYFRAME,
        "state": "cv_state_10d_[x,y,z,w,l,h,vx,vy,vz,yaw]",
        "frame": "local_annotation",
        "classes": {},
    }

    for c in class_ids:
        cname = CLASS_ID_TO_NAME[c]

        r_g_samples = np.concatenate(agg[c]["r_global"], axis=0) if len(agg[c]["r_global"]) > 0 else np.zeros((0, 9), dtype=float)
        r_l_samples = np.concatenate(agg[c]["r_local"], axis=0) if len(agg[c]["r_local"]) > 0 else np.zeros((0, 9), dtype=float)
        q_g_samples = np.concatenate(agg[c]["q_global"], axis=0) if len(agg[c]["q_global"]) > 0 else np.zeros((0, 10), dtype=float)
        q_l_samples = np.concatenate(agg[c]["q_local"], axis=0) if len(agg[c]["q_local"]) > 0 else np.zeros((0, 10), dtype=float)
        q_instances = np.concatenate(agg[c]["q_instances"], axis=0) if len(agg[c]["q_instances"]) > 0 else np.zeros((0,), dtype=int)
        q_unique_instances, q_instance_counts = np.unique(q_instances, return_counts=True)

        r_g_cov = covariance_or_identity(r_g_samples, 9)
        r_l_cov = covariance_or_identity(r_l_samples, 9)
        q_g_cov = covariance_or_identity(q_g_samples, 10)
        q_l_cov = covariance_or_identity(q_l_samples, 10)

        r_global_payload["classes"][cname] = {
            "class_id": c,
            "num_samples": int(r_g_samples.shape[0]),
            "covariance": r_g_cov.tolist(),
        }
        r_local_payload["classes"][cname] = {
            "class_id": c,
            "num_samples": int(r_l_samples.shape[0]),
            "covariance": r_l_cov.tolist(),
        }
        q_global_payload["classes"][cname] = {
            "class_id": c,
            "num_samples": int(q_g_samples.shape[0]),
            "num_instances": int(q_unique_instances.shape[0]),
            "covariance": q_g_cov.tolist(),
            "correlation": correlation_from_cov(q_g_cov).tolist(),
        }
        q_local_payload["classes"][cname] = {
            "class_id": c,
            "num_samples": int(q_l_samples.shape[0]),
            "num_instances": int(q_unique_instances.shape[0]),
            "covariance": q_l_cov.tolist(),
            "correlation": correlation_from_cov(q_l_cov).tolist(),
        }

    model_name = cfg["detector_cfg"]["model_name"]
    output_root = str(cfg.get("covariance_output_path", "outputs/covariances"))
    r_dir = os.path.join(output_root, "diagnostics", "matrix_r")
    os.makedirs(r_dir, exist_ok=True)

    q_dir = os.path.join(output_root, "diagnostics", "matrix_q")
    os.makedirs(q_dir, exist_ok=True)

    r_global_file = f"global_{model_name}_sample_cov_{split}.yaml"
    r_local_file = f"local_{model_name}_sample_cov_{split}.yaml"
    q_global_file = f"global_{model_name}_CV_sample_cov_{split}.yaml"
    q_local_file = f"local_{model_name}_CV_sample_cov_{split}.yaml"

    p1 = write_yaml(r_global_payload, r_dir, r_global_file)
    p2 = write_yaml(r_local_payload, r_dir, r_local_file)
    p3 = write_yaml(q_global_payload, q_dir, q_global_file)
    p4 = write_yaml(q_local_payload, q_dir, q_local_file)

    r_global_model_cfg = payload_to_cov_list(r_global_payload)
    r_local_model_cfg = payload_to_cov_list(r_local_payload)
    q_global_model_cfg = payload_to_cov_list(q_global_payload)
    q_local_model_cfg = payload_to_cov_list(q_local_payload)

    r_model_cfg_dir = os.path.join(output_root, "hydra", "matrix_r")
    os.makedirs(r_model_cfg_dir, exist_ok=True)
    q_model_cfg_dir = os.path.join(output_root, "hydra", "matrix_q")
    os.makedirs(q_model_cfg_dir, exist_ok=True)

    write_yaml(r_global_model_cfg, r_model_cfg_dir, r_global_file)
    write_yaml(r_local_model_cfg, r_model_cfg_dir, r_local_file)
    write_yaml(q_global_model_cfg, q_model_cfg_dir, q_global_file)
    write_yaml(q_local_model_cfg, q_model_cfg_dir, q_local_file)


@hydra.main(config_path="../../../../config", config_name="config", version_base=None)
def main(cfg):
    compute_sample_covariance_nusc(cfg)


if __name__ == "__main__":
    print(os.getcwd())
    main()
