"""
Non-Maximum Suppression(NMS) ops for the NuScenes dataset
Blend NMS used by the CANMOT paper experiments.
"""


import numpy as np
from typing import List
from canmot.data.script.NUSC_CONSTANT import *
from canmot.geometry.nusc_distance import (
    iou_bev_s,
    iou_3d_s,
    giou_bev_s,
    giou_3d_s,
)
from canmot.geometry.nusc_distance import iou_bev, iou_3d, giou_bev, giou_3d


def blend_nms(box_infos: dict, metrics: str, thre: float) -> List[int]:
    """
    :param box_infos: dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param metrics: overlap metric used for NMS
    :param thre: float, threshold of filter
    :return: keep box index, List[int]
    """
    assert metrics in [
        "iou_bev",
        "iou_3d",
        "giou_bev",
        "giou_3d",
    ], "unsupported NMS metrics"
    assert (
        "np_dets" in box_infos and "np_dets_bottom_corners" in box_infos
    ), "must contain specified keys"

    infos, corners = box_infos["np_dets"], box_infos["np_dets_bottom_corners"]
    sort_idxs, keep = np.argsort(-infos[:, -2]), []
    while sort_idxs.size > 0:
        i = sort_idxs[0]
        keep.append(i)
        # only one box left
        if sort_idxs.size == 1:
            break
        left, first = [
            {"np_dets_bottom_corners": corners[idx], "np_dets": infos[idx]}
            for idx in [sort_idxs[1:], i]
        ]
        # the return value number varies by distinct metrics
        if metrics not in METRIC:
            distances = globals()[metrics](first, left)[0]
        else:
            distances = globals()[metrics](first, left)[1][0]
        sort_idxs = sort_idxs[1:][distances <= thre]
    return keep
