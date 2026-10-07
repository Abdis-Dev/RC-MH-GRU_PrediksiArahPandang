"""Evaluate a validation-calibrated Ridge/MH-GRU spherical fusion.

The improved predictor keeps the compact shared MH-GRU, but adds a strong
autoregressive skip branch.  For each horizon, validation data select a scalar
alpha in [0, 1]:

    u_hat = normalize((1-alpha) * u_ridge + alpha * u_mh_gru)

Ridge is trained only on the training split. Alpha and the Ridge regularizer
are selected only on validation data. The test split is used once for final
evaluation. The same procedure is repeated inside each video-level CV fold.
"""

from __future__ import annotations

import csv
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT
REPRO = ROOT
EXT_OUT = REPRO / "results" / "extended"
OUT = REPRO / "results" / "improved_mh_gru"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ext = load_module("extended_experiments", ROOT / "scripts" / "09_train_mh_gru_family.py")
prep = load_module("preprocess", ROOT / "scripts" / "01_preprocess.py")
old_eval = load_module("old_eval", ROOT / "scripts" / "04_evaluate_and_report.py")


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def design(x: np.ndarray) -> np.ndarray:
    z = ext.features(x, True).reshape(len(x), -1).astype(np.float64)
    return np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    a = design(x_train)
    b = design(x_val)
    target = y_train.reshape(len(y_train), -1).astype(np.float64)
    gram = a.T @ a
    rhs = a.T @ target
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        reg = np.eye(gram.shape[0], dtype=np.float64) * lam
        reg[-1, -1] = 0.0
        weights = np.linalg.solve(gram + reg, rhs)
        pred = (b @ weights).reshape(-1, 4, 3)
        score = float(ext.angular_np(pred, y_val).mean())
        if best is None or score < best[0]:
            best = (score, lam, weights)
    assert best is not None
    return best[2], float(best[1]), float(best[0])


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return unit((design(x) @ weights).reshape(-1, 4, 3))


def load_mh_model(path: Path):
    checkpoint = torch.load(path)
    model = ext.model_for(checkpoint["cfg"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_mh(model, x_raw: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    x = torch.from_numpy(ext.features(x_raw, True))
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            parts.append(ext.normalize_t(model(x[start : start + batch_size])).cpu())
    return torch.cat(parts).numpy()


def fit_alphas(
    ridge_val: np.ndarray,
    mh_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 101)
    alphas = np.zeros(4, dtype=np.float64)
    scores = np.zeros(4, dtype=np.float64)
    for hi in range(4):
        candidate_scores = []
        for alpha in grid:
            pred = unit((1.0 - alpha) * ridge_val[:, hi] + alpha * mh_val[:, hi])
            candidate_scores.append(ext.angular_np(pred, y_val[:, hi]).mean())
        index = int(np.argmin(candidate_scores))
        alphas[hi] = grid[index]
        scores[hi] = candidate_scores[index]
    return alphas, scores


def fuse(ridge_pred: np.ndarray, mh_pred: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    return unit((1.0 - alphas[None, :, None]) * ridge_pred + alphas[None, :, None] * mh_pred)


def metric_rows(label: str, seed: int, pred: np.ndarray, y: np.ndarray, alphas=None) -> list[dict]:
    errors = ext.angular_np(pred, y)
    rows = []
    for hi, horizon in enumerate((100, 200, 300, 500)):
        err = errors[:, hi]
        row = {
            "method": label,
            "seed": seed,
            "horizon_ms": horizon,
            "mean_deg": float(err.mean()),
            "median_deg": float(np.median(err)),
            "rmse_deg": float(np.sqrt(np.mean(err**2))),
            "p90_deg": float(np.percentile(err, 90)),
        }
        if alphas is not None:
            row["alpha_validation"] = float(alphas[hi])
        rows.append(row)
    return rows


def original_lstm_errors(x_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(old_eval.add_velocity(x_test))
    columns = []
    with torch.no_grad():
        for hi, horizon in enumerate((100, 200, 300, 500)):
            model = old_eval.LSTMPredictor()
            model.load_state_dict(torch.load(REPRO / "models" / f"lstm_h{horizon}ms.pt"))
            model.eval()
            pred = []
            for start in range(0, len(x), 4096):
                pred.append(model(x[start : start + 4096]))
            p = torch.cat(pred).numpy()
            columns.append(old_eval.angular_error_deg(p, y_test[:, hi]))
    return np.stack(columns, axis=1)


def main_split() -> None:
    train = np.load(REPRO / "data" / "windows_train.npz")
    val = np.load(REPRO / "data" / "windows_val.npz")
    test = np.load(REPRO / "data" / "windows_test.npz")
    weights, lam, ridge_val_score = fit_ridge(train["X"], train["Y"], val["X"], val["Y"])
    ridge_val = ridge_predict(val["X"], weights)
    ridge_test = ridge_predict(test["X"], weights)
    np.savez_compressed(OUT / "ridge_branch_main_split.npz", weights=weights, lambda_value=lam)

    seeds = (42, 43, 44, 45, 46)
    rows: list[dict] = []
    fused_errors = []
    for seed in seeds:
        model = load_mh_model(REPRO / "models" / "extended" / f"mh_gru_seed{seed}.pt")
        mh_val = predict_mh(model, val["X"])
        mh_test = predict_mh(model, test["X"])
        alphas, val_scores = fit_alphas(ridge_val, mh_val, val["Y"])
        pred = fuse(ridge_test, mh_test, alphas)
        err = ext.angular_np(pred, test["Y"])
        fused_errors.append(err)
        seed_rows = metric_rows("RC-MH-GRU", seed, pred, test["Y"], alphas)
        for hi, row in enumerate(seed_rows):
            row["ridge_lambda"] = lam
            row["validation_mean_deg"] = float(val_scores[hi])
            row["ridge_validation_all_horizons_deg"] = ridge_val_score
        rows.extend(seed_rows)
        np.savez_compressed(
            OUT / f"predictions_rc_mh_gru_seed{seed}.npz",
            pred=pred,
            errors=err,
            alphas=alphas,
        )
    write_csv(OUT / "main_split_per_seed.csv", rows)

    stack = np.stack(fused_errors)
    aggregate = []
    for hi, horizon in enumerate((100, 200, 300, 500)):
        seed_means = stack[:, :, hi].mean(axis=1)
        expected = stack[:, :, hi].mean(axis=0)
        aggregate.append({
            "horizon_ms": horizon,
            "n_seeds": len(seeds),
            "mean_deg": float(expected.mean()),
            "sd_mean_across_seeds_deg": float(seed_means.std(ddof=1)),
            "median_deg": float(np.median(expected)),
            "rmse_deg": float(np.sqrt(np.mean(expected**2))),
            "p90_deg": float(np.percentile(expected, 90)),
            "min_seed_mean_deg": float(seed_means.min()),
            "max_seed_mean_deg": float(seed_means.max()),
        })
    write_csv(OUT / "main_split_summary.csv", aggregate)

    # Video-level inference uses the expected error across independent seeds,
    # matching the aggregation used by the current paper for MH-GRU.
    expected_error = stack.mean(axis=0)
    ridge_error = ext.angular_np(ridge_test, test["Y"])
    lstm_error = original_lstm_errors(test["X"], test["Y"])
    videos = np.unique(test["video_id"])
    rng = np.random.default_rng(20260812)
    stats = []
    for hi, horizon in enumerate((100, 200, 300, 500)):
        model_video = np.array([expected_error[test["video_id"] == v, hi].mean() for v in videos])
        for label, comparison in (("Ridge-AR", ridge_error), ("empat LSTM terpisah", lstm_error)):
            other_video = np.array([comparison[test["video_id"] == v, hi].mean() for v in videos])
            difference = model_video - other_video
            boot = np.array([
                difference[rng.integers(0, len(videos), len(videos))].mean()
                for _ in range(10000)
            ])
            lo, high = np.percentile(boot, (2.5, 97.5))
            statistic, p_value = wilcoxon(model_video, other_video, method="exact")
            stats.append({
                "horizon_ms": horizon,
                "comparison": f"RC-MH-GRU vs {label}",
                "n_videos": len(videos),
                "rc_mh_gru_macro_deg": float(model_video.mean()),
                "comparison_macro_deg": float(other_video.mean()),
                "mean_difference_deg": float(difference.mean()),
                "ci95_low_deg": float(lo),
                "ci95_high_deg": float(high),
                "wilcoxon_w": float(statistic),
                "p_exact": float(p_value),
                "rc_mh_gru_wins": int((difference < 0).sum()),
            })
    write_csv(OUT / "main_split_video_statistics.csv", stats)


def balanced_windows(sessions, videos, per_video: int, seed: int):
    rng = np.random.default_rng(seed)
    parts = []
    for video in videos:
        out = prep.build_windows(
            [session for session in sessions if session["video_id"] == video],
            max(prep.HORIZONS_STEPS),
        )
        x, y, v, u = out
        if len(x) > per_video:
            index = np.sort(rng.choice(len(x), per_video, replace=False))
            x, y, v, u = x[index], y[index], v[index], u[index]
        parts.append((x, y, v, u))
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


def video_cross_validation() -> None:
    sessions = prep.load_sessions()
    videos = np.array(sorted({session["video_id"] for session in sessions}))
    rng = np.random.default_rng(42026)
    rng.shuffle(videos)
    folds = np.array_split(videos, 5)
    per_video_rows = []
    fold_rows = []
    for fold_index, test_videos in enumerate(folds, 1):
        remaining = np.array([video for video in videos if video not in set(test_videos)])
        local = np.random.default_rng(1000 + fold_index)
        local.shuffle(remaining)
        n_val = max(8, round(len(remaining) * 0.15))
        val_videos = remaining[:n_val]
        train_videos = remaining[n_val:]
        train = balanced_windows(sessions, train_videos, 2500, 10000 + fold_index)
        val = balanced_windows(sessions, val_videos, 2500, 20000 + fold_index)
        test = balanced_windows(sessions, test_videos, 3000, 30000 + fold_index)
        weights, lam, _ = fit_ridge(train[0], train[1], val[0], val[1])
        ridge_val = ridge_predict(val[0], weights)
        ridge_test = ridge_predict(test[0], weights)
        seed = 100 + fold_index
        model = load_mh_model(
            REPRO / "models" / "extended" / f"mh_gru_cv_f{fold_index}_seed{seed}.pt"
        )
        mh_val = predict_mh(model, val[0])
        mh_test = predict_mh(model, test[0])
        alphas, val_scores = fit_alphas(ridge_val, mh_val, val[1])
        rc_test = fuse(ridge_test, mh_test, alphas)
        rc_error = ext.angular_np(rc_test, test[1])
        mh_error = ext.angular_np(mh_test, test[1])
        ridge_error = ext.angular_np(ridge_test, test[1])
        for video in test_videos:
            mask = test[2] == video
            for hi, horizon in enumerate((100, 200, 300, 500)):
                per_video_rows.append({
                    "fold": fold_index,
                    "video_id": int(video),
                    "horizon_ms": horizon,
                    "rc_mh_gru_mean_deg": float(rc_error[mask, hi].mean()),
                    "mh_gru_mean_deg": float(mh_error[mask, hi].mean()),
                    "ridge_ar_mean_deg": float(ridge_error[mask, hi].mean()),
                    "alpha_validation": float(alphas[hi]),
                    "n_windows": int(mask.sum()),
                })
        fold_rows.append({
            "fold": fold_index,
            "ridge_lambda": lam,
            "alpha_100": alphas[0],
            "alpha_200": alphas[1],
            "alpha_300": alphas[2],
            "alpha_500": alphas[3],
            "validation_100_deg": val_scores[0],
            "validation_200_deg": val_scores[1],
            "validation_300_deg": val_scores[2],
            "validation_500_deg": val_scores[3],
            "n_train_videos": len(train_videos),
            "n_val_videos": len(val_videos),
            "n_test_videos": len(test_videos),
        })
    write_csv(OUT / "video_cv_per_video.csv", per_video_rows)
    write_csv(OUT / "video_cv_folds.csv", fold_rows)

    summary = []
    rng = np.random.default_rng(20260812)
    for horizon in (100, 200, 300, 500):
        selected = [row for row in per_video_rows if row["horizon_ms"] == horizon]
        rc = np.array([row["rc_mh_gru_mean_deg"] for row in selected])
        for label, key in (("MH-GRU", "mh_gru_mean_deg"), ("Ridge-AR", "ridge_ar_mean_deg")):
            other = np.array([row[key] for row in selected])
            difference = rc - other
            boot = np.array([
                difference[rng.integers(0, len(difference), len(difference))].mean()
                for _ in range(10000)
            ])
            lo, high = np.percentile(boot, (2.5, 97.5))
            statistic, p_value = wilcoxon(rc, other)
            summary.append({
                "horizon_ms": horizon,
                "comparison": f"RC-MH-GRU vs {label}",
                "n_videos": len(selected),
                "rc_mh_gru_macro_deg": float(rc.mean()),
                "comparison_macro_deg": float(other.mean()),
                "mean_difference_deg": float(difference.mean()),
                "ci95_low_deg": float(lo),
                "ci95_high_deg": float(high),
                "wilcoxon_w": float(statistic),
                "p_value": float(p_value),
                "rc_mh_gru_wins": int((difference < 0).sum()),
            })
    write_csv(OUT / "video_cv_summary.csv", summary)


def benchmark() -> None:
    train = np.load(REPRO / "data" / "windows_train.npz")
    val = np.load(REPRO / "data" / "windows_val.npz")
    test = np.load(REPRO / "data" / "windows_test.npz")
    weights, _, _ = fit_ridge(train["X"], train["Y"], val["X"], val["Y"])
    model = load_mh_model(REPRO / "models" / "extended" / "mh_gru_seed42.pt")
    ridge_val = ridge_predict(val["X"], weights)
    mh_val = predict_mh(model, val["X"])
    alphas, _ = fit_alphas(ridge_val, mh_val, val["Y"])
    rng = np.random.default_rng(7)
    indices = rng.choice(len(test["X"]), 3000, replace=False)
    x_features = torch.from_numpy(ext.features(test["X"], True))

    def call(index: int):
        ridge = unit((design(test["X"][index : index + 1]) @ weights).reshape(1, 4, 3))
        mh = ext.normalize_t(model(x_features[index : index + 1])).cpu().numpy()
        return fuse(ridge, mh, alphas)

    model.eval()
    with torch.no_grad():
        for index in indices[:200]:
            call(int(index))
        elapsed = []
        for index in indices:
            start = time.perf_counter_ns()
            result = call(int(index))
            elapsed.append((time.perf_counter_ns() - start) / 1000.0)
            if result.shape != (1, 4, 3):
                raise RuntimeError(result.shape)
    write_csv(OUT / "batch1_benchmark.csv", [{
        "method": "RC-MH-GRU",
        "batch_size": 1,
        "n_calls": len(elapsed),
        "median_us_call": float(np.median(elapsed)),
        "p95_us_call": float(np.percentile(elapsed, 95)),
        "p99_us_call": float(np.percentile(elapsed, 99)),
        "n_parameters": 14604 + int(weights.size),
        "hardware": "Apple M3 8-core, 16 GB; PyTorch CPU one thread",
    }])


def main() -> None:
    main_split()
    video_cross_validation()
    benchmark()
    print(f"Results written to {OUT}")


if __name__ == "__main__":
    main()
