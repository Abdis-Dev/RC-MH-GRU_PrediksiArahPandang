"""Verify that bundled result files match the values reported in the paper."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(label: str, actual: float, expected: float, tolerance: float = 0.0006) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected} (tol={tolerance})")


rc = rows(ROOT / "results" / "rc_mh_gru" / "main_split_summary.csv")
tr = rows(ROOT / "results" / "tiny_transformer" / "summary.csv")
stats = rows(ROOT / "results" / "tiny_transformer" / "video_statistics_vs_rc_mh_gru.csv")

expected_rc = {100: 0.958, 200: 2.033, 300: 3.324, 500: 6.041}
expected_tr = {100: 1.026, 200: 2.090, 300: 3.383, 500: 6.118}
expected_p = {100: 0.0009765625, 200: 0.0009765625, 300: 0.0029296875, 500: 0.0244140625}

for row in rc:
    horizon = int(row["horizon_ms"])
    assert_close(f"RC-MH-GRU {horizon} ms", float(row["mean_deg"]), expected_rc[horizon])

for row in tr:
    horizon = int(row["horizon_ms"])
    assert_close(f"Tiny Transformer {horizon} ms", float(row["mean_deg"]), expected_tr[horizon])
    if int(float(row["n_parameters"])) != 14412:
        raise AssertionError("Tiny Transformer parameter count is not 14,412")

for row in stats:
    horizon = int(row["horizon_ms"])
    assert_close(f"Wilcoxon p {horizon} ms", float(row["p_exact"]), expected_p[horizon], 1e-12)

print("OK: ringkasan RC-MH-GRU, Tiny Transformer, parameter, dan nilai p cocok dengan naskah.")
