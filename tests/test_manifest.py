from pathlib import Path

from canmot.reproducibility import compare_calibration, compare_metrics, load_manifest, sha256_file


def test_manifest_has_all_paper_experiments():
    manifest = load_manifest()
    assert len(manifest["experiments"]) == 11
    assert "paper/local/canmot/sc" in manifest["experiments"]
    assert "paper/baseline/poly_mot" in manifest["experiments"]


def test_reference_metrics_match_manifest():
    manifest = load_manifest()
    root = Path(__file__).resolve().parents[1] / "reproducibility"
    import json
    for entry in manifest["experiments"].values():
        actual = json.loads((root / entry["reference"]).read_text())
        assert compare_metrics(actual, entry["expected"]) == []
        assert compare_calibration(actual, actual) == []


def test_hashing(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"CANMOT\n")
    assert sha256_file(sample) == "3fad26553172ff0b419eae59b44a3b4ece3ebe668d2b5049f7cd74ab15f6c127"
