"""
03_train_lstm.py
Trains one small LSTM per prediction horizon to forecast the future
head-orientation unit vector from a 1-second history window.
Input features per timestep: [x,y,z, vx,vy,vz] (position + finite-diff velocity)
Output: predicted unit vector at t+H (denormalized at loss time via MSE,
normalized at evaluation time for angular error).
"""
import json, os, time, csv
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # project root = parent of scripts/
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "split_meta.json")) as f:
    META = json.load(f)
HORIZONS_STEPS = META["horizons_steps"]
WINDOW_SIZE = META["window_size"]

BATCH_SIZE = 512
MAX_EPOCHS = 20
PATIENCE = 4
LR = 1e-3
HIDDEN = 64


def add_velocity(X):
    # X: (N, W, 3) -> return (N, W, 6) with velocity channel (finite diff, first step = 0)
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1).astype(np.float32)


class LSTMPredictor(nn.Module):
    def __init__(self, in_dim=6, hidden=HIDDEN, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last = h[-1]  # (batch, hidden)
        return self.head(last)


def angular_error_deg_t(pred, true):
    pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-9)
    true = true / (true.norm(dim=-1, keepdim=True) + 1e-9)
    dot = torch.clamp((pred * true).sum(-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def load_split(name, hi):
    d = np.load(os.path.join(DATA_DIR, f"windows_{name}.npz"))
    X = add_velocity(d["X"])
    Y = d["Y"][:, hi, :].astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)


def iterate_minibatches(X, Y, batch_size, shuffle=True):
    n = X.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield X[j], Y[j]


def train_one_horizon(h_steps, h_ms):
    print(f"\n=== Training LSTM for horizon {h_ms} ms ({h_steps} steps) ===")
    hi = HORIZONS_STEPS.index(h_steps)
    Xtr, Ytr = load_split("train", hi)
    Xva, Yva = load_split("val", hi)
    print(f"  train={Xtr.shape}  val={Xva.shape}")

    model = LSTMPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        nb = 0
        for xb, yb in iterate_minibatches(Xtr, Ytr, BATCH_SIZE, shuffle=True):
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        train_loss = total_loss / nb

        model.eval()
        with torch.no_grad():
            val_preds = []
            for xb, yb in iterate_minibatches(Xva, Yva, 2048, shuffle=False):
                val_preds.append(model(xb))
            val_pred = torch.cat(val_preds, 0)
            val_ang = angular_error_deg_t(val_pred, Yva).mean().item()
            val_mse = loss_fn(val_pred, Yva).item()

        dt = time.perf_counter() - t0
        print(f"  epoch {epoch:2d}  train_mse={train_loss:.5f}  val_mse={val_mse:.5f}  "
              f"val_ang_err={val_ang:.3f} deg  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_mse, "val_ang_err_deg": val_ang})

        if val_ang < best_val - 1e-4:
            best_val = val_ang
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val_ang_err={best_val:.3f} deg)")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"lstm_h{h_ms}ms.pt"))

    with open(os.path.join(RES_DIR, f"lstm_train_history_h{h_ms}ms.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Saved best model (val_ang_err={best_val:.3f} deg, params={n_params})")
    return best_val, n_params


def main():
    summary = []
    for h_steps in HORIZONS_STEPS:
        h_ms = h_steps * 100
        best_val, n_params = train_one_horizon(h_steps, h_ms)
        summary.append({"horizon_ms": h_ms, "best_val_ang_err_deg": best_val, "n_params": n_params})

    with open(os.path.join(RES_DIR, "lstm_training_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print("\nAll horizons trained. Summary:")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
