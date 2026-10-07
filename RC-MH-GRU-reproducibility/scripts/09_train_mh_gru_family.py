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
OUT = ROOT / "results" / "extended"
MODELS = ROOT / "models" / "extended"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def features(x, delta=True):
    if not delta:
        return x.astype(np.float32)
    d = np.zeros_like(x)
    d[:, 1:] = x[:, 1:] - x[:, :-1]
    return np.concatenate([x, d], axis=-1).astype(np.float32)


def normalize_t(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-9)


def angular_t(pred, true):
    pred, true = normalize_t(pred), normalize_t(true)
    dot = torch.clamp((pred * true).sum(-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def angular_np(pred, true):
    pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    true = true / (np.linalg.norm(true, axis=-1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(pred * true, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


class MultiHorizonRNN(nn.Module):
    def __init__(self, kind="lstm", in_dim=6, hidden=64):
        super().__init__()
        if kind == "lstm":
            self.core = nn.LSTM(in_dim, hidden, batch_first=True)
        elif kind == "gru":
            self.core = nn.GRU(in_dim, hidden, batch_first=True)
        else:
            raise ValueError(kind)
        self.kind = kind
        self.head = nn.Linear(hidden, 12)

    def forward(self, x):
        if self.kind == "lstm":
            _, (h, _) = self.core(x)
        else:
            _, h = self.core(x)
        return self.head(h[-1]).reshape(-1, 4, 3)


class MultiHorizonTCN(nn.Module):
    def __init__(self, in_dim=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 12)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))[:, :, -1]
        return self.head(z).reshape(-1, 4, 3)


class SphericalResidualAttentionGRU(nn.Module):
    def __init__(self, in_dim=6, hidden=64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.queries = nn.Parameter(torch.randn(4, hidden) * 0.02)
        self.head = nn.Linear(hidden, 3)
        self.scale = hidden ** -0.5

    def forward(self, x):
        seq, _ = self.gru(x)
        score = torch.einsum("bth,kh->bkt", seq, self.queries) * self.scale
        weight = torch.softmax(score, dim=-1)
        context = torch.einsum("bkt,bth->bkh", weight, seq)
        residual = self.head(context)
        return x[:, -1, :3].unsqueeze(1) + residual


def model_for(cfg):
    in_dim = 6 if cfg["delta"] else 3
    if cfg["kind"] in ("lstm", "gru"):
        return MultiHorizonRNN(cfg["kind"], in_dim=in_dim)
    if cfg["kind"] == "sragru":
        return SphericalResidualAttentionGRU(in_dim=in_dim)
    return MultiHorizonTCN(in_dim=in_dim)


def loss_value(pred, y, loss_name):
    if loss_name == "cosine":
        return (1.0 - (normalize_t(pred) * normalize_t(y)).sum(-1)).mean()
    return nn.functional.mse_loss(pred, y)


def evaluate(model, x, y, batch=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            preds.append(normalize_t(model(x[i:i + batch])).cpu())
    pred = torch.cat(preds)
    err = angular_t(pred, y.cpu()).numpy()
    return pred.numpy(), err


def train_one(cfg, seed, data, max_epochs=20, patience=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr_raw, ytr, xva_raw, yva, xte_raw, yte = data
    xtr = torch.from_numpy(features(xtr_raw, cfg["delta"]))
    xva = torch.from_numpy(features(xva_raw, cfg["delta"]))
    xte = torch.from_numpy(features(xte_raw, cfg["delta"]))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    yva_t = torch.from_numpy(yva.astype(np.float32))
    yte_t = torch.from_numpy(yte.astype(np.float32))
    model = model_for(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    batch = 1024
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(xtr))
        running = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            pred = model(xtr[idx])
            loss = loss_value(pred, ytr_t[idx], cfg["loss"])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        _, val_err = evaluate(model, xva, yva_t)
        val = float(val_err.mean())
        history.append({"epoch": epoch, "train_loss": running / len(xtr), "val_mean_deg": val})
        print(cfg["name"], "seed", seed, "epoch", epoch, "val", f"{val:.4f}", flush=True)
        if val < best - 1e-4:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    pred, err = evaluate(model, xte, yte_t)
    n_params = sum(p.numel() for p in model.parameters())
    path = MODELS / f"{cfg['name']}_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "seed": seed}, path)
    with (OUT / f"history_{cfg['name']}_seed{seed}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    rows = []
    for hi, h in enumerate((100, 200, 300, 500)):
        e = err[:, hi]
        rows.append({
            "config": cfg["name"], "seed": seed, "horizon_ms": h,
            "mean_deg": float(e.mean()), "median_deg": float(np.median(e)),
            "rmse_deg": float(np.sqrt(np.mean(e ** 2))),
            "p90_deg": float(np.percentile(e, 90)),
            "best_val_all_horizons_deg": best, "n_params": n_params,
        })
    np.savez_compressed(OUT / f"predictions_{cfg['name']}_seed{seed}.npz", pred=pred, errors=err)
    return rows


def ridge_baseline(data):
    xtr, ytr, xva, yva, xte, yte = data
    def design(x):
        z = features(x, True).reshape(len(x), -1).astype(np.float64)
        return np.concatenate([z, np.ones((len(z), 1))], axis=1)
    a, b, c = design(xtr), design(xva), design(xte)
    yt = ytr.reshape(len(ytr), -1).astype(np.float64)
    gram = a.T @ a
    rhs = a.T @ yt
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        reg = np.eye(gram.shape[0]) * lam
        reg[-1, -1] = 0
        w = np.linalg.solve(gram + reg, rhs)
        pv = (b @ w).reshape(-1, 4, 3)
        ev = angular_np(pv, yva)
        score = float(ev.mean())
        if best is None or score < best[0]:
            best = (score, lam, w)
    score, lam, w = best
    pred = (c @ w).reshape(-1, 4, 3)
    err = angular_np(pred, yte)
    rows = []
    for hi, h in enumerate((100, 200, 300, 500)):
        e = err[:, hi]
        rows.append({"config": "ridge_ar", "seed": 0, "horizon_ms": h,
                     "mean_deg": float(e.mean()), "median_deg": float(np.median(e)),
                     "rmse_deg": float(np.sqrt(np.mean(e ** 2))),
                     "p90_deg": float(np.percentile(e, 90)),
                     "best_val_all_horizons_deg": score, "n_params": int(w.size),
                     "lambda": lam})
    np.savez_compressed(OUT / "predictions_ridge_ar.npz", pred=pred, errors=err)
    return rows


def write_rows(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate_seed_results(rows):
    selected = [r for r in rows if r["config"] == "smh_lstm"]
    out = []
    for h in (100, 200, 300, 500):
        rr = [r for r in selected if r["horizon_ms"] == h]
        out.append({"horizon_ms": h, "n_seeds": len(rr),
                    "mean_of_mean_deg": float(np.mean([r["mean_deg"] for r in rr])),
                    "sd_of_mean_deg": float(np.std([r["mean_deg"] for r in rr], ddof=1)),
                    "min_mean_deg": float(np.min([r["mean_deg"] for r in rr])),
                    "max_mean_deg": float(np.max([r["mean_deg"] for r in rr]))})
    write_rows(OUT / "multiseed_summary.csv", out)


def cluster_stats(test_video, proposed_err):
    base = np.load(ROOT / "results" / "baseline_errors.npz")
    rng = np.random.default_rng(2026)
    vids = np.unique(test_video)
    rows = []
    for hi, hstep in enumerate((1, 2, 3, 5)):
        le = proposed_err[:, hi]
        for bname in ("Persistence", "LinearExtrap", "KalmanFilter"):
            be = base[f"{bname}_{hstep}"]
            lv = np.array([le[test_video == v].mean() for v in vids])
            bv = np.array([be[test_video == v].mean() for v in vids])
            diff = lv - bv
            boots = []
            for _ in range(10000):
                idx = rng.integers(0, len(vids), len(vids))
                boots.append(diff[idx].mean())
            lo, hi95 = np.percentile(boots, [2.5, 97.5])
            stat, p = wilcoxon(lv, bv, method="exact")
            rows.append({"horizon_ms": hstep * 100, "comparison": f"SMH-LSTM vs {bname}",
                         "n_videos": len(vids), "macro_lstm_deg": lv.mean(),
                         "macro_baseline_deg": bv.mean(), "mean_diff_deg": diff.mean(),
                         "ci95_diff_low_deg": lo, "ci95_diff_high_deg": hi95,
                         "wilcoxon_w": stat, "p_exact": p,
                         "wins": int(np.sum(diff < 0))})
    write_rows(OUT / "video_cluster_statistics.csv", rows)


def benchmark_batch1(model, xraw, learned_label="SMH-LSTM"):
    import importlib.util
    spec = importlib.util.spec_from_file_location("baseline", ROOT / "scripts" / "02_baselines.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    x = torch.from_numpy(features(xraw, True))
    rng = np.random.default_rng(7)
    idx = rng.choice(len(xraw), 3000, replace=False)
    methods = {
        "Persistence": lambda i: baseline.predict_persistence(xraw[i:i + 1]),
        "LinearExtrap": lambda i: baseline.predict_linear(xraw[i:i + 1]),
        "KalmanFilter": lambda i: baseline.predict_kalman(xraw[i:i + 1]),
        learned_label: lambda i: model(x[i:i + 1]),
    }
    rows = []
    model.eval()
    with torch.no_grad():
        for name, fn in methods.items():
            for i in idx[:200]:
                fn(int(i))
            times = []
            for i in idx:
                t0 = time.perf_counter_ns()
                result = fn(int(i))
                times.append((time.perf_counter_ns() - t0) / 1000)
                if result is None:
                    raise RuntimeError(name)
            rows.append({"method": name, "batch_size": 1, "n_calls": len(times),
                         "median_us_call": float(np.median(times)),
                         "p95_us_call": float(np.percentile(times, 95)),
                         "p99_us_call": float(np.percentile(times, 99)),
                         "hardware": "Apple M3 8-core, 16 GB; PyTorch CPU one thread"})
    write_rows(OUT / "batch1_benchmark.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    tr = np.load(ROOT / "data" / "windows_train.npz")
    va = np.load(ROOT / "data" / "windows_val.npz")
    te = np.load(ROOT / "data" / "windows_test.npz")
    data = (tr["X"], tr["Y"], va["X"], va["Y"], te["X"], te["Y"])
    configs = [
        {"name": "smh_lstm", "kind": "lstm", "delta": True, "loss": "cosine"},
        {"name": "smh_lstm_position_only", "kind": "lstm", "delta": False, "loss": "cosine"},
        {"name": "smh_lstm_mse", "kind": "lstm", "delta": True, "loss": "mse"},
        {"name": "mh_gru", "kind": "gru", "delta": True, "loss": "cosine"},
        {"name": "mh_gru_position_only", "kind": "gru", "delta": False, "loss": "cosine"},
        {"name": "mh_gru_mse", "kind": "gru", "delta": True, "loss": "mse"},
        {"name": "mh_tcn", "kind": "tcn", "delta": True, "loss": "cosine"},
        {"name": "sra_gru", "kind": "sragru", "delta": True, "loss": "cosine"},
    ]
    all_rows = []
    if args.only:
        configs = [c for c in configs if c["name"] == args.only]
    seeds = [42] if args.quick else [42, 43, 44, 45, 46]
    for cfg in configs:
        cfg_seeds = seeds if cfg["name"] in ("smh_lstm", "mh_gru") else [42]
        for seed in cfg_seeds:
            all_rows.extend(train_one(cfg, seed, data, max_epochs=8 if args.quick else 20,
                                      patience=2 if args.quick else 4))
    if args.only:
        write_rows(OUT / f"results_{args.only}.csv", all_rows)
        print("WROTE", OUT / f"results_{args.only}.csv")
        return
    all_rows.extend(ridge_baseline(data))
    write_rows(OUT / "all_extended_results.csv", all_rows)
    aggregate_seed_results(all_rows)
    chosen = np.load(OUT / "predictions_smh_lstm_seed42.npz")
    cluster_stats(te["video_id"], chosen["errors"])
    ckpt = torch.load(MODELS / "smh_lstm_seed42.pt")
    model = model_for(ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    benchmark_batch1(model, te["X"])
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
