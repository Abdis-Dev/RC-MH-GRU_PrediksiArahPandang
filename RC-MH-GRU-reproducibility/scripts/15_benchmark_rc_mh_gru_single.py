"""Benchmark one deployable Ridge-calibrated MH-GRU instance.

The model, Ridge regularisation, and per-horizon fusion weights are loaded
from the validation-locked seed-42 run.  As for the MH-GRU benchmark, window
features are prepared before timing so the measurement covers prediction,
not data preparation.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT
REPRO = ROOT
OUT = REPRO / "results" / "improved_mh_gru"
torch.set_num_threads(1)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ext = load_module("extended_experiments", ROOT / "scripts" / "09_train_mh_gru_family.py")
fusion = load_module("fusion", ROOT / "scripts" / "05_ridge_calibrated_mh_gru.py")


def main() -> None:
    output = OUT / "single_inference_benchmark.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    train = np.load(REPRO / "data" / "windows_train.npz")
    val = np.load(REPRO / "data" / "windows_val.npz")
    test = np.load(REPRO / "data" / "windows_test.npz")
    weights, _, _ = fusion.fit_ridge(train["X"], train["Y"], val["X"], val["Y"])
    saved = np.load(OUT / "predictions_rc_mh_gru_seed42.npz")
    alphas = saved["alphas"]
    checkpoint = torch.load(REPRO / "models" / "extended" / "mh_gru_seed42.pt")
    model = ext.model_for(checkpoint["cfg"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    features = torch.from_numpy(ext.features(test["X"], True))
    ridge_design = fusion.design(test["X"])

    def call(index: int) -> np.ndarray:
        ridge = fusion.unit((ridge_design[index:index + 1] @ weights).reshape(1, 4, 3))
        with torch.no_grad():
            mh = ext.normalize_t(model(features[index:index + 1])).cpu().numpy()
        return fusion.fuse(ridge, mh, alphas)

    rng = np.random.default_rng(7)
    index = rng.choice(len(test["X"]), 3000, replace=False)
    for item in index[:200]:
        call(int(item))
    elapsed = []
    for item in index:
        start = __import__("time").perf_counter_ns()
        prediction = call(int(item))
        elapsed.append((__import__("time").perf_counter_ns() - start) / 1000.0)
        if prediction.shape != (1, 4, 3):
            raise RuntimeError(prediction.shape)

    row = {
        "method": "RC-MH-GRU (satu anggota)", "batch_size": 1, "n_calls": len(elapsed),
        "median_us_call": float(np.median(elapsed)), "p95_us_call": float(np.percentile(elapsed, 95)),
        "p99_us_call": float(np.percentile(elapsed, 99)), "n_parameters": 14604 + int(weights.size),
        "hardware": "Apple M3 8-core, 16 GB; PyTorch CPU one thread",
    }
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader(); writer.writerow(row)
    print(output)


if __name__ == "__main__":
    main()
