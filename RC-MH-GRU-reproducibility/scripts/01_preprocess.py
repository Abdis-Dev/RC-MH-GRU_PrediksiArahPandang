"""
01_preprocess.py
Parses the aggregated 360 head-tracking dataset, converts (pitch,yaw) to
3D unit vectors on the viewing sphere (to correctly handle yaw wrap-around),
splits sessions by VIDEO ID (train/val/test) to avoid data leakage, and
builds fixed-length sliding windows -> multi-horizon prediction targets.

Output: data/windows_{split}.npz  with arrays:
  X          (N, window_size, 3)  float32   input unit-vector history
  Y          (N, n_horizons, 3)   float32   target unit-vector at t+H
  video_id   (N,)                 int32
  user_idx   (N,)                 int32
Also writes data/split_meta.json and data/dataset_summary.csv
"""
import glob, os, json, random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # project root = parent of scripts/
RAW_DIR = os.path.join(BASE_DIR, "dataset", "AggregatedDataset")
OUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 10                 # 1.0 s history at 10 Hz
HORIZONS_STEPS = [1, 2, 3, 5]    # -> 100, 200, 300, 500 ms at 10 Hz
HORIZON_MS = [h * 100 for h in HORIZONS_STEPS]
STRIDE = 3                       # stride when sliding the window over a session
SEED = 42

MAX_TRAIN_WINDOWS = 200_000
MAX_VAL_WINDOWS = 40_000
MAX_TEST_WINDOWS = 60_000

random.seed(SEED)
np.random.seed(SEED)


def pitch_yaw_to_vec(pitch, yaw):
    # pitch: elevation angle (rad), yaw: azimuth angle (rad)
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    v = np.stack([x, y, z], axis=-1)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return v.astype(np.float32)


def load_sessions():
    files = sorted(
        [f for f in glob.glob(os.path.join(RAW_DIR, "*.txt")) if "readMe" not in f],
        key=lambda x: int(os.path.basename(x).replace(".txt", "")),
    )
    sessions = []  # list of dict(video_id, user_idx, vec (T,3))
    for fp in files:
        vid = int(os.path.basename(fp).replace(".txt", ""))
        with open(fp) as f:
            lines = f.readlines()
        n_users = (len(lines) - 1) // 2
        for i in range(n_users):
            pitch = np.array([float(x) for x in lines[1 + 2 * i].split()], dtype=np.float64)
            yaw = np.array([float(x) for x in lines[2 + 2 * i].split()], dtype=np.float64)
            L = min(len(pitch), len(yaw))
            if L < WINDOW_SIZE + max(HORIZONS_STEPS) + 1:
                continue
            vec = pitch_yaw_to_vec(pitch[:L], yaw[:L])
            sessions.append({"video_id": vid, "user_idx": i, "vec": vec, "length": L})
    return sessions


def build_windows(sessions, max_h):
    """Vectorized sliding-window extraction per session."""
    Xs, Ys, vids, uidx = [], [], [], []
    for s in sessions:
        vec = s["vec"]
        L = vec.shape[0]
        last_start = L - WINDOW_SIZE - max_h
        if last_start < 0:
            continue
        starts = np.arange(0, last_start + 1, STRIDE)
        if len(starts) == 0:
            continue
        # input windows: (n, WINDOW_SIZE, 3)
        idx = starts[:, None] + np.arange(WINDOW_SIZE)[None, :]
        X = vec[idx]  # (n, W, 3)
        # targets for each horizon: (n, n_horizons, 3)
        tgt_idx = starts[:, None] + (WINDOW_SIZE - 1) + np.array(HORIZONS_STEPS)[None, :]
        Y = vec[tgt_idx]  # (n, n_horizons, 3)
        Xs.append(X)
        Ys.append(Y)
        vids.append(np.full(len(starts), s["video_id"], dtype=np.int32))
        uidx.append(np.full(len(starts), s["user_idx"], dtype=np.int32))
    if not Xs:
        return None
    return (
        np.concatenate(Xs, 0),
        np.concatenate(Ys, 0),
        np.concatenate(vids, 0),
        np.concatenate(uidx, 0),
    )


def subsample(X, Y, vid, uidx, max_n):
    n = X.shape[0]
    if n <= max_n:
        return X, Y, vid, uidx
    sel = np.random.choice(n, max_n, replace=False)
    sel.sort()
    return X[sel], Y[sel], vid[sel], uidx[sel]


def main():
    print("Loading & parsing sessions ...")
    sessions = load_sessions()
    n_videos = len(set(s["video_id"] for s in sessions))
    print(f"  sessions={len(sessions)}  unique videos={n_videos}")

    video_ids = sorted(set(s["video_id"] for s in sessions))
    rng = random.Random(SEED)
    rng.shuffle(video_ids)
    n = len(video_ids)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_v = set(video_ids[:n_train])
    val_v = set(video_ids[n_train:n_train + n_val])
    test_v = set(video_ids[n_train + n_val:])

    split_meta = {
        "window_size": WINDOW_SIZE,
        "horizons_steps": HORIZONS_STEPS,
        "horizons_ms": HORIZON_MS,
        "stride": STRIDE,
        "seed": SEED,
        "n_videos_total": n,
        "train_videos": sorted(train_v),
        "val_videos": sorted(val_v),
        "test_videos": sorted(test_v),
    }

    max_h = max(HORIZONS_STEPS)
    summary_rows = []
    for split_name, vid_set, cap in [
        ("train", train_v, MAX_TRAIN_WINDOWS),
        ("val", val_v, MAX_VAL_WINDOWS),
        ("test", test_v, MAX_TEST_WINDOWS),
    ]:
        split_sessions = [s for s in sessions if s["video_id"] in vid_set]
        out = build_windows(split_sessions, max_h)
        if out is None:
            raise RuntimeError(f"No windows for split {split_name}")
        X, Y, vid, uidx = out
        n_before = X.shape[0]
        X, Y, vid, uidx = subsample(X, Y, vid, uidx, cap)
        np.savez_compressed(
            os.path.join(OUT_DIR, f"windows_{split_name}.npz"),
            X=X, Y=Y, video_id=vid, user_idx=uidx,
        )
        print(f"  {split_name}: videos={len(vid_set)} sessions={len(split_sessions)} "
              f"windows_generated={n_before} windows_saved={X.shape[0]}")
        summary_rows.append({
            "split": split_name,
            "n_videos": len(vid_set),
            "n_sessions": len(split_sessions),
            "n_windows_generated": n_before,
            "n_windows_saved": X.shape[0],
        })

    with open(os.path.join(OUT_DIR, "split_meta.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    import csv
    with open(os.path.join(OUT_DIR, "dataset_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)

    print("Done. Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
