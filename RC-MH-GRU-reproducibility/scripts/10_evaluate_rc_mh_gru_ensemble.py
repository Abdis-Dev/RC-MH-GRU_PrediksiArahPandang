"""Final held-out evaluation of the validation-calibrated RC-MH-GRU ensemble.

Five independently trained shared MH-GRU members are averaged on the sphere.
A Ridge-AR branch is fit on training videos. Four fusion weights are selected
by validation angular error and locked before the test split is evaluated.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT
REPRO = ROOT
EXT = REPRO / "results" / "extended"
OUT = REPRO / "results" / "rc_mh_gru_ensemble"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ext = load_module("extended_experiments", ROOT / "scripts" / "09_train_mh_gru_family.py")
old_eval = load_module("old_eval", ROOT / "scripts" / "04_evaluate_and_report.py")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def design(x: np.ndarray) -> np.ndarray:
    z = ext.features(x, True).reshape(len(x), -1).astype(np.float64)
    return np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)


def fit_ridge(x_train, y_train, x_val, y_val):
    train_design, val_design = design(x_train), design(x_val)
    target = y_train.reshape(len(y_train), -1).astype(np.float64)
    gram, rhs = train_design.T @ train_design, train_design.T @ target
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        reg = np.eye(gram.shape[0]) * lam
        reg[-1, -1] = 0.0
        weights = np.linalg.solve(gram + reg, rhs)
        prediction = (val_design @ weights).reshape(-1, 4, 3)
        score = float(ext.angular_np(prediction, y_val).mean())
        if best is None or score < best[0]:
            best = (score, lam, weights)
    assert best is not None
    return best[2], float(best[1]), float(best[0])


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return unit((design(x) @ weights).reshape(-1, 4, 3))


def model_predictions(x_raw: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(ext.features(x_raw, True))
    members = []
    with torch.no_grad():
        for seed in (42, 43, 44, 45, 46):
            checkpoint = torch.load(REPRO / "models" / "extended" / f"mh_gru_seed{seed}.pt")
            model = ext.model_for(checkpoint["cfg"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            pieces = [ext.normalize_t(model(x[start : start + 4096])).cpu()
                      for start in range(0, len(x), 4096)]
            members.append(torch.cat(pieces).numpy())
    return unit(np.mean(members, axis=0))


def choose_alpha(ridge_val: np.ndarray, mh_val: np.ndarray, y_val: np.ndarray):
    alphas, scores = [], []
    for hi in range(4):
        candidates = []
        for alpha in np.linspace(0.0, 1.0, 101):
            pred = unit((1.0 - alpha) * ridge_val[:, hi] + alpha * mh_val[:, hi])
            candidates.append(float(ext.angular_np(pred, y_val[:, hi]).mean()))
        best = int(np.argmin(candidates))
        alphas.append(best / 100.0)
        scores.append(candidates[best])
    return np.asarray(alphas), np.asarray(scores)


def original_lstm_errors(x_test, y_test):
    x = torch.from_numpy(old_eval.add_velocity(x_test))
    columns = []
    with torch.no_grad():
        for hi, horizon in enumerate((100, 200, 300, 500)):
            model = old_eval.LSTMPredictor()
            model.load_state_dict(torch.load(REPRO / "models" / f"lstm_h{horizon}ms.pt"))
            model.eval()
            pred = [model(x[start : start + 4096]) for start in range(0, len(x), 4096)]
            columns.append(old_eval.angular_error_deg(torch.cat(pred).numpy(), y_test[:, hi]))
    return np.stack(columns, axis=1)


def summary_rows(label: str, errors: np.ndarray, params: int, seeds: int) -> list[dict]:
    rows = []
    for hi, horizon in enumerate((100, 200, 300, 500)):
        error = errors[:, hi]
        rows.append({
            "method": label, "horizon_ms": horizon, "mean_deg": float(error.mean()),
            "median_deg": float(np.median(error)), "rmse_deg": float(np.sqrt(np.mean(error**2))),
            "p90_deg": float(np.percentile(error, 90)), "n_parameters_total": params,
            "n_mh_gru_members": seeds,
        })
    return rows


def video_statistics(model_errors, comparison_errors, test_video, label):
    videos = np.unique(test_video)
    rng = np.random.default_rng(20260812)
    rows = []
    for hi, horizon in enumerate((100, 200, 300, 500)):
        model = np.array([model_errors[test_video == video, hi].mean() for video in videos])
        other = np.array([comparison_errors[test_video == video, hi].mean() for video in videos])
        diff = model - other
        boot = np.array([diff[rng.integers(0, len(videos), len(videos))].mean() for _ in range(10000)])
        lo, high = np.percentile(boot, (2.5, 97.5))
        statistic, p_value = wilcoxon(model, other, method="exact")
        rows.append({
            "horizon_ms": horizon, "comparison": f"RC-MH-GRU ensemble vs {label}",
            "n_videos": len(videos), "rc_mh_gru_macro_deg": float(model.mean()),
            "comparison_macro_deg": float(other.mean()), "mean_difference_deg": float(diff.mean()),
            "ci95_low_deg": float(lo), "ci95_high_deg": float(high),
            "wilcoxon_w": float(statistic), "p_exact": float(p_value),
            "rc_mh_gru_wins": int((diff < 0).sum()),
        })
    return rows


def main():
    train = np.load(REPRO / "data" / "windows_train.npz")
    val = np.load(REPRO / "data" / "windows_val.npz")
    test = np.load(REPRO / "data" / "windows_test.npz")
    weights, lam, ridge_val_mean = fit_ridge(train["X"], train["Y"], val["X"], val["Y"])
    ridge_val, ridge_test = ridge_predict(val["X"], weights), ridge_predict(test["X"], weights)
    mh_val, mh_test = model_predictions(val["X"]), model_predictions(test["X"])
    alphas, alpha_val_scores = choose_alpha(ridge_val, mh_val, val["Y"])
    prediction = unit((1.0 - alphas[None, :, None]) * ridge_test + alphas[None, :, None] * mh_test)
    rc_errors = ext.angular_np(prediction, test["Y"])
    ridge_errors = ext.angular_np(ridge_test, test["Y"])
    lstm_errors = original_lstm_errors(test["X"], test["Y"])
    persistence = np.load(REPRO / "results" / "baseline_errors.npz")
    baseline_rows = []
    for label, errors, parameters, members in (
        ("Persistence", np.stack([persistence[f"Persistence_{step}"] for step in (1, 2, 3, 5)], axis=1), 0, 0),
        ("Ekstrapolasi Linier", np.stack([persistence[f"LinearExtrap_{step}"] for step in (1, 2, 3, 5)], axis=1), 0, 0),
        ("Filter Kalman", np.stack([persistence[f"KalmanFilter_{step}"] for step in (1, 2, 3, 5)], axis=1), 0, 0),
        ("Ridge-AR", ridge_errors, int(weights.size), 0),
        ("empat LSTM terpisah", lstm_errors, 74508, 4),
        ("RC-MH-GRU ensemble", rc_errors, 5 * 14604 + int(weights.size), 5),
    ):
        baseline_rows.extend(summary_rows(label, errors, parameters, members))
    write_csv(OUT / "final_method_comparison.csv", baseline_rows)
    write_csv(OUT / "video_statistics.csv", video_statistics(rc_errors, ridge_errors, test["video_id"], "Ridge-AR") +
              video_statistics(rc_errors, lstm_errors, test["video_id"], "empat LSTM terpisah"))
    np.savez_compressed(OUT / "predictions_rc_mh_gru_ensemble.npz", pred=prediction, errors=rc_errors,
                        alphas=alphas, ridge_prediction=ridge_test, mh_prediction=mh_test)
    with (OUT / "calibration.json").open("w") as handle:
        json.dump({
            "ridge_lambda": lam, "ridge_validation_mean_deg": ridge_val_mean,
            "horizon_ms": [100, 200, 300, 500], "alpha_mh_gru": alphas.tolist(),
            "validation_error_deg": alpha_val_scores.tolist(), "members": [42, 43, 44, 45, 46],
            "test_split_used_only_after_calibration": True,
        }, handle, indent=2)
    print(OUT)


if __name__ == "__main__":
    main()
