"""Generate the first-frame token table for nuScenes trainval."""

import json
import os
import sys

import hydra

sys.path.append("../..")
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

FIRST_TOKEN_ROOT_PATH = "../utils/first_token_table/"


def extract_first_token(dataset_path, dataset_name="NuScenes", dataset_version="trainval"):
    """
    :param dataset_path: path of dataset
    :param dataset_name: name of dataset
    :param dataset_version: nuScenes trainval version used by the paper
    :return: first frame token table .json
    """
    if dataset_version != "trainval" or dataset_name != "NuScenes":
        raise ValueError("CANMOT only provides the paper's nuScenes trainval token table")

    nusc = NuScenes(version="v1.0-trainval", dataroot=dataset_path, verbose=True)
    first_token_table = [scene["first_sample_token"] for scene in tqdm(nusc.scene, desc="scenes")]
    os.makedirs(FIRST_TOKEN_ROOT_PATH + dataset_version, exist_ok=True)
    first_token_path = FIRST_TOKEN_ROOT_PATH + dataset_version + "/nusc_first_token.json"
    print(f"write token table to {first_token_path}")
    with open(first_token_path, "w", encoding="utf-8") as stream:
        json.dump(first_token_table, stream)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    dataset_cfg = cfg["dataset_cfg"]
    dataset_path = dataset_cfg["dataset_path"]
    extract_first_token(
        dataset_path=dataset_path,
        dataset_name="NuScenes",
        dataset_version="trainval",
    )


if __name__ == "__main__":
    main()
