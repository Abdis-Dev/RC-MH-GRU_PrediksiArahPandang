"""Train and evaluate a parameter-matched Tiny Transformer baseline.

The baseline uses exactly the same 10 x 6 input representation, four target
horizons, video-disjoint train/validation/test split, cosine objective, and
five random seeds as the MH-GRU branch used in the manuscript.  Model
selection is based only on the validation set.  The test set is evaluated
after the best epoch has been restored.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT
OUT = REPRO / "results" / "tiny_transformer_review1"
MODELS = REPRO / "models" / "tiny_transformer_review1"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

HORIZONS = (100, 200, 300, 500)
SEEDS = (42, 43, 44, 45, 46)


def features(x: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(x)
    delta[:, 1:] = x[:, 1:] - x[:, :-1]
    return np.concatenate([x, delta], axis=-1).astype(np.float32)


def normalize_t(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)


def angular_np(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    true = true / (np.linalg.norm(true, axis=-1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(pred * true, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


class TinyTransformer(nn.Module):
    """One-layer encoder sized to match the compact RC-MH-GRU budget."""

    def __init__(self, sequence_length: int = 10, d_model: int = 40) -> None:
        super().__init__()
        self.input_projection = nn.Linear(6, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=80,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 12)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x) + self.position[:, : x.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1]).reshape(-1, 4, 3)


def evaluate(model: nn.Module, x: torch.Tensor, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            pred.append(normalize_t(model(x[start : start + 4096])).cpu())
    prediction = torch.cat(pred).numpy()
    return prediction, angular_np(prediction, y)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_seed(
    seed: int,
    train: np.lib.npyio.NpzFile,
    val: np.lib.npyio.NpzFile,
    test: np.lib.npyio.NpzFile,
    max_epochs: int,
) -> tuple[list[dict], np.ndarray]:
    model_path = MODELS / f"tiny_transformer_seed{seed}.pt"
    history_path = OUT / f"history_seed{seed}.csv"
    prediction_path = OUT / f"predictions_seed{seed}.npz"
    if model_path.exists() and history_path.exists() and prediction_path.exists():
        saved = np.load(prediction_path)
        rows = json.loads((OUT / f"metrics_seed{seed}.json").read_text())
        return rows, saved["errors"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train = torch.from_numpy(features(train["X"]))
    y_train = torch.from_numpy(train["Y"].astype(np.float32))
    x_val = torch.from_numpy(features(val["X"]))
    x_test = torch.from_numpy(features(test["X"]))
    model = TinyTransformer(sequence_length=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        running = 0.0
        for start in range(0, len(order), 1024):
            index = order[start : start + 1024]
            optimizer.zero_grad()
            prediction = normalize_t(model(x_train[index]))
            loss = (1.0 - (prediction * y_train[index]).sum(dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * len(index)
        _, val_error = evaluate(model, x_val, val["Y"])
        val_score = float(val_error.mean())
        scheduler.step(val_score)
        history.append(
            {
                "epoch": epoch,
                "train_cosine_loss": running / len(x_train),
                "validation_mean_deg": val_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(f"seed={seed} epoch={epoch:02d} validation={val_score:.5f}", flush=True)
        if val_score < best - 1e-4:
            best = val_score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break

    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)
    prediction, errors = evaluate(model, x_test, test["Y"])
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    rows = []
    for index, horizon in enumerate(HORIZONS):
        error = errors[:, index]
        rows.append(
            {
                "method": "Tiny Transformer",
                "seed": seed,
                "horizon_ms": horizon,
                "mean_deg": float(error.mean()),
                "median_deg": float(np.median(error)),
                "rmse_deg": float(np.sqrt(np.mean(error**2))),
                "p90_deg": float(np.percentile(error, 90)),
                "best_validation_all_horizons_deg": best,
                "n_parameters": n_parameters,
            }
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "architecture": {
                "sequence_length": int(x_train.shape[1]),
                "input_dim": 6,
                "d_model": 40,
                "n_heads": 4,
                "feedforward_dim": 80,
                "encoder_layers": 1,
                "dropout": 0.0,
                "output_dim": 12,
            },
            "best_validation_mean_deg": best,
            "n_parameters": n_parameters,
        },
        model_path,
    )
    write_csv(history_path, history)
    np.savez_compressed(prediction_path, prediction=prediction, errors=errors)
    (OUT / f"metrics_seed{seed}.json").write_text(json.dumps(rows, indent=2))
    return rows, errors


def aggregate(per_seed_rows: list[dict]) -> list[dict]:
    rows = []
    for horizon in HORIZONS:
        selected = [row for row in per_seed_rows if row["horizon_ms"] == horizon]
        row = {
            "method": "Tiny Transformer",
            "horizon_ms": horizon,
            "n_seeds": len(selected),
            "mean_deg": float(np.mean([item["mean_deg"] for item in selected])),
            "sd_mean_across_seeds_deg": float(np.std([item["mean_deg"] for item in selected], ddof=1)),
            "median_deg": float(np.mean([item["median_deg"] for item in selected])),
            "rmse_deg": float(np.mean([item["rmse_deg"] for item in selected])),
            "p90_deg": float(np.mean([item["p90_deg"] for item in selected])),
            "n_parameters": selected[0]["n_parameters"],
        }
        rows.append(row)
    return rows


def video_statistics(test: np.lib.npyio.NpzFile, transformer_errors: dict[int, np.ndarray]) -> None:
    videos = np.unique(test["video_id"])
    rc_errors = {
        seed: np.load(
            REPRO / "results" / "improved_mh_gru" / f"predictions_rc_mh_gru_seed{seed}.npz"
        )["errors"]
        for seed in SEEDS
    }
    rng = np.random.default_rng(20261007)
    rows = []
    for hi, horizon in enumerate(HORIZONS):
        transformer_video = np.array(
            [
                np.mean(
                    [transformer_errors[seed][test["video_id"] == video, hi].mean() for seed in SEEDS]
                )
                for video in videos
            ]
        )
        rc_video = np.array(
            [
                np.mean([rc_errors[seed][test["video_id"] == video, hi].mean() for seed in SEEDS])
                for video in videos
            ]
        )
        difference = rc_video - transformer_video
        bootstrap = np.array(
            [difference[rng.integers(0, len(videos), len(videos))].mean() for _ in range(10000)]
        )
        statistic, p_value = wilcoxon(rc_video, transformer_video, method="exact")
        rows.append(
            {
                "horizon_ms": horizon,
                "comparison": "RC-MH-GRU vs Tiny Transformer",
                "n_videos": len(videos),
                "rc_mh_gru_macro_deg": float(rc_video.mean()),
                "tiny_transformer_macro_deg": float(transformer_video.mean()),
                "mean_difference_deg": float(difference.mean()),
                "ci95_low_deg": float(np.percentile(bootstrap, 2.5)),
                "ci95_high_deg": float(np.percentile(bootstrap, 97.5)),
                "wilcoxon_w": float(statistic),
                "p_exact": float(p_value),
                "rc_mh_gru_wins": int((difference < 0).sum()),
            }
        )
    write_csv(OUT / "video_statistics_vs_rc_mh_gru.csv", rows)


def benchmark(test: np.lib.npyio.NpzFile) -> None:
    torch.set_num_threads(1)
    checkpoint = torch.load(MODELS / "tiny_transformer_seed42.pt")
    model = TinyTransformer(sequence_length=10)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.from_numpy(features(test["X"]))
    rng = np.random.default_rng(7)
    indexes = rng.choice(len(x), 3000, replace=False)
    with torch.no_grad():
        for index in indexes[:200]:
            normalize_t(model(x[index : index + 1]))
        elapsed = []
        for index in indexes:
            start = time.perf_counter_ns()
            result = normalize_t(model(x[index : index + 1]))
            elapsed.append((time.perf_counter_ns() - start) / 1000.0)
            if result.shape != (1, 4, 3):
                raise RuntimeError(result.shape)
    row = {
        "method": "Tiny Transformer",
        "batch_size": 1,
        "n_calls": len(elapsed),
        "median_us_call": float(np.median(elapsed)),
        "p95_us_call": float(np.percentile(elapsed, 95)),
        "p99_us_call": float(np.percentile(elapsed, 99)),
        "n_parameters": checkpoint["n_parameters"],
        "hardware": "Apple M3 8-core, 16 GB; PyTorch CPU one thread",
    }
    write_csv(OUT / "inference_benchmark.csv", [row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()
    selected_seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    unsupported = set(selected_seeds) - set(SEEDS)
    if unsupported:
        raise ValueError(f"Unsupported seeds: {sorted(unsupported)}")

    torch.set_num_threads(1)
    train = np.load(REPRO / "data" / "windows_train.npz")
    val = np.load(REPRO / "data" / "windows_val.npz")
    test = np.load(REPRO / "data" / "windows_test.npz")
    all_rows = []
    errors = {}
    for seed in selected_seeds:
        rows, seed_errors = train_seed(seed, train, val, test, args.max_epochs)
        all_rows.extend(rows)
        errors[seed] = seed_errors
    write_csv(OUT / "per_seed_results.csv", all_rows)

    if set(selected_seeds) == set(SEEDS):
        write_csv(OUT / "summary.csv", aggregate(all_rows))
        video_statistics(test, errors)
        benchmark(test)
    print(OUT)


if __name__ == "__main__":
    main()
