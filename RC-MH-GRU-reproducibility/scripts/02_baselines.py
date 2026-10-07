"""
02_baselines.py
Implements three non-learned baselines evaluated on the SAME test windows
used for the LSTM, so comparisons are apples-to-apples:
  1. Persistence      : predict = last observed orientation
  2. Linear extrap.    : least-squares linear fit over the input window,
                          extrapolated to t+H, renormalized to unit sphere
  3. Kalman Filter     : constant-velocity KF run across the input window,
                          state propagated forward H steps

All operate on (window_size, 3) unit-vector histories and predict a
(3,) unit vector per horizon. Also records per-call inference time.
"""
import json, time, os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # project root = parent of scripts/
DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "split_meta.json")) as f:
    META = json.load(f)
WINDOW_SIZE = META["window_size"]
HORIZONS_STEPS = META["horizons_steps"]
HORIZON_MS = META["horizons_ms"]
DT = 0.1  # seconds per sample (10 Hz)


def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-9)


# ---------- 1. Persistence ----------
def predict_persistence(X):
    # X: (N, W, 3) -> for every horizon just repeat the last observed vector
    last = X[:, -1, :]  # (N,3)
    preds = {h: last.copy() for h in HORIZONS_STEPS}
    return preds


# ---------- 2. Linear extrapolation (least-squares fit per window) ----------
def predict_linear(X):
    N, W, D = X.shape
    t = np.arange(W, dtype=np.float64) * DT  # (W,)
    t_mean = t.mean()
    t_c = t - t_mean
    denom = np.sum(t_c ** 2)
    # slope b and intercept a for each sample & each of the 3 dims, vectorized
    # X: (N,W,3) ; t_c: (W,)
    Xc = X - X.mean(axis=1, keepdims=True)
    b = np.tensordot(Xc, t_c, axes=([1], [0])) / denom  # (N,3)  slope
    a = X.mean(axis=1)  # (N,3) intercept at t_mean
    last_t = t[-1]
    preds = {}
    for h in HORIZONS_STEPS:
        future_t = last_t + h * DT
        pred = a + b * (future_t - t_mean)
        preds[h] = normalize(pred).astype(np.float32)
    return preds


# ---------- 3. Kalman Filter (constant velocity, 3D) ----------
def predict_kalman(X, q=1e-3, r=1e-2):
    """
    State: [x,y,z,vx,vy,vz]. Runs the filter across the W observed samples
    of each window (vectorized across the batch via a Python loop over the
    small W=10 timesteps, matrix ops vectorized across N samples).
    """
    N, W, D = X.shape
    dt = DT
    F = np.eye(6)
    for i in range(3):
        F[i, i + 3] = dt
    Hm = np.zeros((3, 6))
    Hm[:, :3] = np.eye(3)
    Q = np.eye(6) * q
    R = np.eye(3) * r

    # initialize state: position = first obs, velocity = 0
    state = np.zeros((N, 6))
    state[:, :3] = X[:, 0, :]
    P = np.tile((np.eye(6) * 1.0)[None, :, :], (N, 1, 1))

    for t in range(1, W):
        # predict
        state = state @ F.T
        P = np.einsum('ij,njk,kl->nil', F, P, F.T) + Q
        # update
        z = X[:, t, :]
        y_res = z - state @ Hm.T
        S = np.einsum('ij,njk,kl->nil', Hm, P, Hm.T) + R
        S_inv = np.linalg.inv(S)
        K = np.einsum('njk,kl,nlm->njm', P, Hm.T, S_inv)
        state = state + np.einsum('nij,nj->ni', K, y_res)
        I_KH = np.eye(6)[None, :, :] - np.einsum('nij,jk->nik', K, Hm)
        P = np.einsum('nij,njk->nik', I_KH, P)

    preds = {}
    for h in HORIZONS_STEPS:
        Fh = np.eye(6)
        for i in range(3):
            Fh[i, i + 3] = dt * h
        fut = state @ Fh.T
        preds[h] = normalize(fut[:, :3]).astype(np.float32)
    return preds


def angular_error_deg(pred, true):
    pred = normalize(pred)
    true = normalize(true)
    dot = np.clip(np.sum(pred * true, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def evaluate_method(name, predict_fn, X, Y, kwargs=None):
    kwargs = kwargs or {}
    t0 = time.perf_counter()
    preds = predict_fn(X, **kwargs)
    t1 = time.perf_counter()
    per_sample_ms = (t1 - t0) / X.shape[0] * 1000.0
    rows = []
    errors_by_h = {}
    for hi, h in enumerate(HORIZONS_STEPS):
        err = angular_error_deg(preds[h], Y[:, hi, :])
        errors_by_h[h] = err
        rows.append({
            "method": name,
            "horizon_ms": h * 100,
            "mean_ang_err_deg": float(err.mean()),
            "median_ang_err_deg": float(np.median(err)),
            "rmse_ang_err_deg": float(np.sqrt(np.mean(err ** 2))),
            "p90_ang_err_deg": float(np.percentile(err, 90)),
            "std_ang_err_deg": float(err.std()),
            "inference_ms_per_sample": per_sample_ms,
        })
    return rows, errors_by_h


def main():
    test = np.load(os.path.join(DATA_DIR, "windows_test.npz"))
    X, Y = test["X"], test["Y"]
    print("Test set:", X.shape, Y.shape)

    all_rows = []
    all_errors = {}

    for name, fn, kw in [
        ("Persistence", predict_persistence, {}),
        ("LinearExtrap", predict_linear, {}),
        ("KalmanFilter", predict_kalman, {}),
    ]:
        print(f"Evaluating {name} ...")
        rows, errs = evaluate_method(name, fn, X, Y, kw)
        all_rows.extend(rows)
        all_errors[name] = errs
        for r in rows:
            print(f"  H={r['horizon_ms']:>4}ms  mean={r['mean_ang_err_deg']:.3f} deg  "
                  f"rmse={r['rmse_ang_err_deg']:.3f}  infer={r['inference_ms_per_sample']*1000:.4f} us/sample")

    import csv
    out_csv = os.path.join(RES_DIR, "baseline_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print("Saved:", out_csv)

    # save per-sample errors for statistical testing later
    np.savez_compressed(
        os.path.join(RES_DIR, "baseline_errors.npz"),
        **{f"{m}_{h}": all_errors[m][h] for m in all_errors for h in HORIZONS_STEPS}
    )
    print("Saved per-sample errors for stat tests.")


if __name__ == "__main__":
    main()
