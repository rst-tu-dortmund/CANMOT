# CANMOT

**CANMOT: Class-Aware Noise Modeling for Multi-Object Tracking in Autonomous Driving**  
Timo Osterburg, Stefan Schütte, Torsten Bertram — IROS 2026 · [Paper (arXiv)](https://arxiv.org/pdf/2606.03590)

This repository is the official implementation and reproduction release for **CANMOT**. It
keeps the recognizable Poly-MOT tracker architecture while adding the
covariance-aware filters, frozen paper configurations, calibration evaluation,
and figures used in the paper. Hydra experiment paths are the canonical public
interface.

CANMOT estimates class-specific process and measurement noise for a Kalman
filter and, in its local form, expresses uncertainty in the tracked object's
heading-aligned frame. Poly-MOT's two-stage GIoU/Hungarian association and
track lifecycle remain intact, making the covariance model—not a redesigned
tracker—the principal experimental change.

## Install

Use Python 3.10 and clone submodules:

```bash
git clone --recurse-submodules https://github.com/rst-tu-dortmund/CANMOT.git canmot
cd canmot
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e '.[test]' --no-deps
pytest
```

Copy the template and fill in local paths. This file is intentionally ignored:

```bash
cp config/environment_cfg/base_environment.yaml config/environment_cfg/my_machine.yaml
```

All five values are mandatory. `dataset_base_path/nuscenes` must contain
nuScenes; `detector_base_path/centerpoint_flip` must contain the detector JSONs;
`logging_base_path` and `cache_path` must be writable; `n_processes` is `0` for
a single process or a positive worker count.

```text
<dataset_base_path>/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-trainval/

<detector_base_path>/centerpoint_flip/
├── infos_train_10sweeps_withvelo_filter_True.json
└── infos_val_10sweeps_withvelo_filter_True.json
```

Then run:

```bash
canmot preflight --environment my_machine
python run.py environment_cfg=my_machine +experiment=paper/local/canmot/sc
canmot verify /path/from/logging_base_path/paper/local/canmot/sc
```

Unresolved `???` values are rejected before data is loaded or output is
created. Each run stores the fully resolved config, revisions, dependency
versions, input hash, and UTC timestamp.

## Paper experiments

| Hydra path | Table I row | Covariance frame/source |
|---|---|---|
| `paper/local/canmot/sc` | CANMOT-SC | local, fixed empirical |
| `paper/local/canmot/opt` | CANMOT-Opt | local, diagonal optimized |
| `paper/local/shared` | Local Shared-Opt | local, diagonal shared |
| `paper/local/sample/train` | Local SC-T | local, train sample |
| `paper/local/sample/val` | Local SC-V | local, val sample |
| `paper/local/sample/trainval` | Local SC-TV | local, trainval sample |
| `paper/global/shared` | Global Shared-Opt | global, diagonal shared |
| `paper/global/sample/train` | Global SC-T | global, train sample |
| `paper/global/sample/val` | Global SC-V | global, val sample |
| `paper/global/sample/trainval` | Global SC-TV | global, trainval sample |
| `paper/baseline/poly_mot` | Poly-MOT | published baseline |

Every path above is directly runnable by replacing the experiment in the
example command. The CLI is only a convenience wrapper over those same Hydra
configs:

```bash
canmot run paper/global/shared --environment my_machine
canmot reproduce tables --environment my_machine
```

The full suite is compute- and storage-intensive: allow multiple hours (or
longer, depending on CPU and storage), roughly 10 GiB of free working space,
plus nuScenes and detector storage. Existing output paths are explicit and
repeatable; move or remove an old run before rerunning if it must be preserved.

## Detector input

Download the CenterPoint `flip` result bundle linked from
[CenterPoint issue #249](https://github.com/tianweiy/CenterPoint/issues/249).
The official folder URL, expected filenames, and SHA-256 hashes are recorded in
`reproducibility/detectors.yaml`. Place the two files at the stated relative
paths. `preflight` refuses missing or mismatched files.

## Covariances and optimization

The fixed empirical matrices used by CANMOT-SC and the six sample-covariance
experiments are full matrices and contain measured off-diagonal terms. The
CANMOT-Opt and Shared-Opt matrices are diagonal.

Regenerate empirical matrices outside the Git tree:

```bash
canmot estimate-covariances --environment my_machine
```

Run a seeded paper recipe with a resumable JSONL ledger:

```bash
canmot optimize paper/local/canmot --environment my_machine
canmot optimize paper/local/canmot_sc --environment my_machine
canmot optimize paper/local/shared --environment my_machine
canmot optimize paper/global/shared --environment my_machine
```

The recipes search covariance diagonals in `[1e-4, 3]` with ten evaluations per
parameter. Class-aware recipes optimize per-class AMOTA independently; shared
recipes optimize overall AMOTA. Each search is seeded and records completed
evaluations in a resumable JSONL ledger. The frozen configurations reproduce
the paper tables directly.

## Figures and calibration results

```bash
canmot figures --environment my_machine
```

This creates Figure 1 from the frozen local/global covariance configurations,
then runs `paper/local/canmot/sc`, `paper/global/sample/trainval`,
`paper/local/canmot/opt`, and `paper/baseline/poly_mot` to render the scene-144,
step-35 comparison for Figure 2. The figure configurations are recorded in the
paper manifest. Figure outputs remain outside Git.

Complete unrounded evaluation and calibration statistics, including class
ANEES, violation percentages, and chi-square conclusions, live in
`reproducibility/reference/`. `paper_manifest.yaml` maps them to every paper
row. Verification requires IDS, FRAG, TP, and FP exactly and compares floating
metrics at displayed paper precision. The extremely large Poly-MOT ANEES is a
known loss of positive-semidefiniteness in the published baseline and is
reported rather than silently repaired.

## Smoke test and scripts

```bash
scripts/preflight.sh my_machine
scripts/smoke.sh
scripts/tables.sh my_machine
scripts/figures.sh my_machine
scripts/optimize.sh my_machine paper/local/canmot
```

The smoke fixture is synthetic and needs neither nuScenes nor detector files.
For errors, first check the ignored environment YAML, detector checksums, the
pinned submodule revision, available disk space, and that the command is run
from the repository root.

## Acknowledgements

CANMOT builds directly on [Poly-MOT](https://github.com/lixiaoyu2000/Poly-MOT)
by Xiaoyu Li et al.; its tracker architecture, association, and lifecycle
management are retained here. We thank its authors for releasing their code.
We also build on [py-motmetrics](https://github.com/cheind/py-motmetrics), the
[nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit), and
[CenterPoint](https://github.com/tianweiy/CenterPoint) detections, and parts
of the geometry code are inspired by
[SimpleTrack](https://github.com/tusen-ai/SimpleTrack),
[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT), and
[EagerMOT](https://github.com/aleksandrkim61/EagerMOT).

## Citation and licenses

See `UPSTREAM_PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`, and
`reproducibility/paper_manifest.yaml` for exact provenance.

CANMOT-authored material is BSD-3-Clause. Retained Poly-MOT and vendored
motmetrics material is MIT-licensed; the adapted devkit keeps its own notices.

```bibtex
@inproceedings{osterburg2026canmot,
  title     = {{CANMOT}: Class-Aware Noise Modeling for Multi-Object Tracking in Autonomous Driving},
  author    = {Osterburg, Timo and Sch{\"u}tte, Stefan and Bertram, Torsten},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
