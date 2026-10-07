"""
04_evaluate_and_report.py
- Evaluates trained LSTM models on the held-out TEST set (same windows used
  for the baselines) for all horizons.
- Combines LSTM + baseline results into one master table.
- Runs paired Wilcoxon signed-rank tests (LSTM vs each baseline) per horizon.
- Produces publication-ready figures (PNG, 300dpi) and CSV/LaTeX tables.
"""
import json, os, time, csv
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

torch.set_num_threads(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # project root = parent of scripts/
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "split_meta.json")) as f:
    META = json.load(f)
HORIZONS_STEPS = META["horizons_steps"]
HORIZON_MS = META["horizons_ms"]

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "Persistence": "#999999",
    "LinearExtrap": "#4C72B0",
    "KalmanFilter": "#DD8452",
    "LSTM": "#55A868",
}
METHOD_ORDER = ["Persistence", "LinearExtrap", "KalmanFilter", "LSTM"]


class LSTMPredictor(nn.Module):
    def __init__(self, in_dim=6, hidden=64, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.head(h[-1])


def add_velocity(X):
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1).astype(np.float32)


def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-9)


def angular_error_deg(pred, true):
    pred = normalize(pred)
    true = normalize(true)
    dot = np.clip(np.sum(pred * true, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def evaluate_lstm_on_test():
    test = np.load(os.path.join(DATA_DIR, "windows_test.npz"))
    X_raw, Y = test["X"], test["Y"]
    Xv = add_velocity(X_raw)
    Xt = torch.from_numpy(Xv)

    rows = []
    errors = {}
    for hi, h_steps in enumerate(HORIZONS_STEPS):
        h_ms = h_steps * 100
        model = LSTMPredictor()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"lstm_h{h_ms}ms.pt")))
        model.eval()

        with torch.no_grad():
            t0 = time.perf_counter()
            preds = []
            bs = 4096
            for i in range(0, Xt.shape[0], bs):
                preds.append(model(Xt[i:i + bs]))
            pred = torch.cat(preds, 0).numpy()
            t1 = time.perf_counter()
        per_sample_ms = (t1 - t0) / Xt.shape[0] * 1000.0

        err = angular_error_deg(pred, Y[:, hi, :])
        errors[h_steps] = err
        rows.append({
            "method": "LSTM",
            "horizon_ms": h_ms,
            "mean_ang_err_deg": float(err.mean()),
            "median_ang_err_deg": float(np.median(err)),
            "rmse_ang_err_deg": float(np.sqrt(np.mean(err ** 2))),
            "p90_ang_err_deg": float(np.percentile(err, 90)),
            "std_ang_err_deg": float(err.std()),
            "inference_ms_per_sample": per_sample_ms,
        })
        print(f"LSTM  H={h_ms:>4}ms  mean={err.mean():.3f} deg  rmse={np.sqrt(np.mean(err**2)):.3f}  "
              f"infer={per_sample_ms*1000:.3f} us/sample  n_params={sum(p.numel() for p in model.parameters())}")
    return rows, errors, Y.shape[0]


def main():
    # ---- 1. gather all results ----
    lstm_rows, lstm_errors, n_test = evaluate_lstm_on_test()

    with open(os.path.join(RES_DIR, "baseline_results.csv")) as f:
        baseline_rows = list(csv.DictReader(f))
    for r in baseline_rows:
        for k in ["horizon_ms", "mean_ang_err_deg", "median_ang_err_deg", "rmse_ang_err_deg",
                  "p90_ang_err_deg", "std_ang_err_deg", "inference_ms_per_sample"]:
            r[k] = float(r[k]) if k != "horizon_ms" else int(float(r[k]))

    all_rows = baseline_rows + lstm_rows
    with open(os.path.join(RES_DIR, "all_results_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print("\nSaved combined results table: results/all_results_table.csv")

    # ---- 2. statistical significance: LSTM vs each baseline, per horizon ----
    baseline_errors = np.load(os.path.join(RES_DIR, "baseline_errors.npz"))
    stat_rows = []
    for h_steps in HORIZONS_STEPS:
        h_ms = h_steps * 100
        lstm_err = lstm_errors[h_steps]
        for base_name in ["Persistence", "LinearExtrap", "KalmanFilter"]:
            base_err = baseline_errors[f"{base_name}_{h_steps}"]
            # paired Wilcoxon signed-rank test (non-parametric, robust to skew)
            stat, p = wilcoxon(lstm_err, base_err)
            mean_diff = lstm_err.mean() - base_err.mean()
            stat_rows.append({
                "horizon_ms": h_ms,
                "comparison": f"LSTM vs {base_name}",
                "lstm_mean_deg": lstm_err.mean(),
                "baseline_mean_deg": base_err.mean(),
                "mean_diff_deg": mean_diff,
                "wilcoxon_stat": stat,
                "p_value": p,
                "significant_at_0.05": p < 0.05,
                "n_test_samples": n_test,
            })
    with open(os.path.join(RES_DIR, "significance_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=stat_rows[0].keys())
        w.writeheader()
        w.writerows(stat_rows)
    print("Saved significance tests: results/significance_tests.csv")
    for r in stat_rows:
        sig = "***" if r["p_value"] < 0.001 else ("*" if r["significant_at_0.05"] else "ns")
        print(f"  H={r['horizon_ms']:>4}ms  {r['comparison']:<24} "
              f"diff={r['mean_diff_deg']:+.3f} deg  p={r['p_value']:.2e}  {sig}")

    # ---- 3. FIGURE 1: mean angular error vs horizon (line plot) ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for method in METHOD_ORDER:
        xs, ys, yerr = [], [], []
        for h_ms in HORIZON_MS:
            rows = [r for r in all_rows if r["method"] == method and int(r["horizon_ms"]) == h_ms]
            xs.append(h_ms)
            ys.append(rows[0]["mean_ang_err_deg"])
            yerr.append(rows[0]["std_ang_err_deg"] / np.sqrt(n_test))  # SEM
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=method,
                    color=METHOD_COLORS[method], linewidth=2)
    ax.set_xlabel("Prediction horizon (ms)")
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title("Prediction accuracy vs. compensated latency")
    ax.set_xticks(HORIZON_MS)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_error_vs_horizon.png"))
    plt.close(fig)

    # ---- 4. FIGURE 2: boxplot of error distribution at H=200ms ----
    h_show = 2  # steps -> 200ms
    fig, ax = plt.subplots(figsize=(6, 4.2))
    data = [
        baseline_errors[f"Persistence_{h_show}"],
        baseline_errors[f"LinearExtrap_{h_show}"],
        baseline_errors[f"KalmanFilter_{h_show}"],
        lstm_errors[h_show],
    ]
    bp = ax.boxplot(data, tick_labels=METHOD_ORDER, showfliers=False, patch_artist=True)
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.6)
    ax.set_ylabel("Angular error (degrees)")
    ax.set_title("Error distribution at 200 ms prediction horizon")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_boxplot_200ms.png"))
    plt.close(fig)

    # ---- 5. FIGURE 3: inference time comparison (log scale bar chart) ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    times = []
    for method in METHOD_ORDER:
        rows = [r for r in all_rows if r["method"] == method and int(r["horizon_ms"]) == 100]
        times.append(rows[0]["inference_ms_per_sample"] * 1000)  # to microseconds
    bars = ax.bar(METHOD_ORDER, times, color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("Inference time (microseconds/sample, log scale)")
    ax.set_title("Computational cost per prediction")
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_inference_time.png"))
    plt.close(fig)

    # ---- 6. FIGURE 4: qualitative trajectory example (yaw/pitch over time) ----
    test = np.load(os.path.join(DATA_DIR, "windows_test.npz"))
    X_raw, Y = test["X"], test["Y"]
    sample_idx = 12345
    hist = X_raw[sample_idx]  # (10,3)
    true_targets = Y[sample_idx]  # (4,3)

    def vec_to_yaw_pitch(v):
        yaw = np.arctan2(v[..., 1], v[..., 0])
        pitch = np.arcsin(np.clip(v[..., 2], -1, 1))
        return np.degrees(yaw), np.degrees(pitch)

    hy, hp = vec_to_yaw_pitch(hist)
    ty, tp = vec_to_yaw_pitch(true_targets)

    Xv_single = add_velocity(X_raw[sample_idx:sample_idx + 1])
    xt_single = torch.from_numpy(Xv_single)
    pred_yaw, pred_pitch = {}, {}
    for h_steps in HORIZONS_STEPS:
        h_ms = h_steps * 100
        model = LSTMPredictor()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"lstm_h{h_ms}ms.pt")))
        model.eval()
        with torch.no_grad():
            p = model(xt_single).numpy()[0]
        py, pp = vec_to_yaw_pitch(p)
        pred_yaw[h_steps], pred_pitch[h_steps] = py, pp

    t_hist = np.arange(-9, 1) * 100  # ms, history
    t_fut = np.array(HORIZON_MS)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(t_hist, hy, "o-", color="black", label="History (observed)")
    axes[0].plot(t_fut, ty, "s-", color="black", alpha=0.4, label="Ground truth (future)")
    axes[0].plot(t_fut, [pred_yaw[h] for h in HORIZONS_STEPS], "^--", color=METHOD_COLORS["LSTM"], label="LSTM prediction")
    axes[0].set_ylabel("Yaw (deg)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].axvline(0, color="gray", linestyle=":")

    axes[1].plot(t_hist, hp, "o-", color="black")
    axes[1].plot(t_fut, tp, "s-", color="black", alpha=0.4)
    axes[1].plot(t_fut, [pred_pitch[h] for h in HORIZONS_STEPS], "^--", color=METHOD_COLORS["LSTM"])
    axes[1].set_ylabel("Pitch (deg)")
    axes[1].set_xlabel("Time relative to prediction point (ms)")
    axes[1].axvline(0, color="gray", linestyle=":")

    fig.suptitle("Example: observed history vs. predicted future head orientation")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_example_trajectory.png"))
    plt.close(fig)

    print("\nAll figures saved to", FIG_DIR)

    # ---- 7. Pretty printed summary table (markdown) for quick inspection ----
    with open(os.path.join(RES_DIR, "summary_table.md"), "w") as f:
        f.write("| Method | Horizon (ms) | Mean err (deg) | Median (deg) | RMSE (deg) | P90 (deg) | Inference (us/sample) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for h_ms in HORIZON_MS:
            for method in METHOD_ORDER:
                r = [x for x in all_rows if x["method"] == method and int(x["horizon_ms"]) == h_ms][0]
                f.write(f"| {method} | {h_ms} | {r['mean_ang_err_deg']:.3f} | {r['median_ang_err_deg']:.3f} | "
                        f"{r['rmse_ang_err_deg']:.3f} | {r['p90_ang_err_deg']:.3f} | "
                        f"{r['inference_ms_per_sample']*1000:.4f} |\n")
    print("Saved markdown summary table: results/summary_table.md")


if __name__ == "__main__":
    main()
