"""
Organize detector files in chronological order on the NuScenes dataset
"""

import os, json, sys

import hydra

sys.path.append("../..")
from canmot.utils.io import load_file
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

from canmot.config_resolvers import register_resolvers

register_resolvers()

OUTPUT_ROOT_PATH = "../detector/"


def reorder_detection(
    detector_path,
    dataset_path,
    dataset_name="NuScenes",
    dataset_version="trainval",
    detector_name="centerpoint",
):
    """
    :param detector_path: path of detection file
    :param dataset_path: root path of dataset file
    :param dataset_name: name of dataset
    :param dataset_version: nuScenes trainval version used by the paper
    :param detector_name: name of detector eg: CenterPoint..
    :return: Reorganized detection files .json
    """
    if dataset_version != "trainval" or dataset_name.lower() != "nuscenes":
        raise ValueError("CANMOT only orders the paper's nuScenes trainval detections")

    nusc = NuScenes(version="v1.0-trainval", dataroot=dataset_path, verbose=True)
    chaos_detector_json = load_file(detector_path)
    detector_results = chaos_detector_json["results"]
    first_token_path = (
        "./data/utils/first_token_table/{}/nusc_first_token.json".format(
            dataset_version
        )
    )
    print(f"loading first token table from {hydra.utils.to_absolute_path(first_token_path)}")
    all_token_table = from_first_to_all(
        nusc, first_token_path, allowed_tokens=set(detector_results)
    )
    if len(all_token_table) != len(detector_results):
        raise ValueError("Detector file contains tokens outside nuScenes trainval")

    order_file = {
        "results": {
            token: detector_results[token]
            for token in all_token_table
        },
        "meta": chaos_detector_json["meta"],
    }

    version = "val"
    os.makedirs(OUTPUT_ROOT_PATH + version, exist_ok=True)
    output_path = OUTPUT_ROOT_PATH + version + f"/{version}_{detector_name}.json"
    print(f"write order detection file to {output_path}")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(order_file, stream)


def from_first_to_all(nusc, first_token_path, allowed_tokens=None):
    """
    :param nusc: NuScenes class
    :param first_token_path: path of first frame token for each seq
    :return: list format token table
    """
    first_token_table = load_file(first_token_path)
    all_token_table = []
    for first_token in first_token_table:
        curr_token = first_token
        while curr_token != "":
            if allowed_tokens is None or curr_token in allowed_tokens:
                all_token_table.append(curr_token)
            curr_token = nusc.get("sample", curr_token)["next"]

    return all_token_table


@hydra.main(config_path="../../../../config", config_name="config", version_base=None)
def main(cfg):
    dataset_cfg = cfg["dataset_cfg"]
    detector_cfg = cfg["detector_cfg"]
    dataset_path = dataset_cfg["dataset_path"]

    detections_path = detector_cfg["ordered_detections_path"]["val"]
    reorder_detection(
        dataset_path=dataset_path,
        detector_path=detections_path,
        dataset_name=dataset_cfg["name"],
        dataset_version="trainval",
        detector_name=detector_cfg["model_name"],
    )


if __name__ == "__main__":
    main()
