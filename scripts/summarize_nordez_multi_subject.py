"""Summarize direct 3DUS versus generic OpenSim MTU lengths across subjects."""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ultrasound_validation" / "nordez_mtu"
POSTURES = ["GAS1", "GAS2a", "GAS2b", "GAS3", "GAS4", "GAS5"]


def correlation(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    rows = []
    for path in glob.glob(str(DATA / "S??_GAS*_mtu_comparison.json")):
        data = json.loads(Path(path).read_text())
        if "subject" not in data or "opensim_lengths_mm" not in data:
            continue
        for muscle in ("gasmed_r", "gaslat_r"):
            measured = data["direct_3dus_lengths_mm"][muscle]
            modeled = data["opensim_lengths_mm"][muscle]["mean_mm"]
            rows.append({
                "subject": data["subject"],
                "posture": data["posture"],
                "muscle": muscle,
                "direct_mm": measured,
                "opensim_mm": modeled,
                "error_mm": modeled - measured,
                "error_pct": 100.0 * (modeled - measured) / measured,
                "opensim_rom_mm": data["opensim_lengths_mm"][muscle]["rom_mm"],
            })
    detail = pd.DataFrame(rows).sort_values(["subject", "muscle", "posture"])
    subject_rows = []
    for (subject, muscle), group in detail.groupby(["subject", "muscle"]):
        if len(group) < 3:
            continue
        direct = group["direct_mm"].to_numpy()
        modeled = group["opensim_mm"].to_numpy()
        subject_rows.append({
            "subject": subject,
            "muscle": muscle,
            "n_postures": len(group),
            "posture_correlation": correlation(direct, modeled),
            "mean_absolute_error_mm": float(np.mean(np.abs(modeled - direct))),
            "mean_signed_error_pct": float(group["error_pct"].mean()),
            "direct_rom_mm": float(direct.max() - direct.min()),
            "opensim_rom_mm": float(modeled.max() - modeled.min()),
        })
    subject_summary = pd.DataFrame(subject_rows)
    pooled = (
        detail.groupby("muscle")
        .agg(
            n_subjects=("subject", "nunique"),
            n_posture_records=("posture", "size"),
            median_posture_correlation=("error_pct", lambda _: float(subject_summary[subject_summary["muscle"] == _.name if False else "muscle"].get("posture_correlation", pd.Series(dtype=float)).median()) if False else np.nan),
            mean_absolute_error_mm=("error_mm", lambda x: float(np.mean(np.abs(x)))),
            mean_signed_error_pct=("error_pct", "mean"),
        )
        .reset_index()
    )
    pooled_corr = subject_summary.groupby("muscle")["posture_correlation"].agg(["median", "mean"]).reset_index()
    pooled = pooled.drop(columns=["median_posture_correlation"])
    pooled = pooled.merge(pooled_corr, on="muscle", how="left")
    pooled = pooled.rename(columns={"median": "median_subject_posture_correlation", "mean": "mean_subject_posture_correlation"})

    detail.to_csv(DATA / "nordez_direct_mtu_multi_subject_detail.csv", index=False)
    subject_summary.to_csv(DATA / "nordez_direct_mtu_subject_summary.csv", index=False)
    pooled.to_csv(DATA / "nordez_direct_mtu_pooled_summary.csv", index=False)
    summary = {
        "subjects_with_records": sorted(detail["subject"].unique().tolist()),
        "n_subjects": int(detail["subject"].nunique()),
        "n_posture_records": int(len(detail) // 2),
        "missing_posture_records": 84 - int(len(detail) // 2),
        "median_subject_correlation_gasmed": float(subject_summary.loc[subject_summary.muscle == "gasmed_r", "posture_correlation"].median()),
        "median_subject_correlation_gaslat": float(subject_summary.loc[subject_summary.muscle == "gaslat_r", "posture_correlation"].median()),
        "mean_signed_error_percent_gasmed": float(detail.loc[detail.muscle == "gasmed_r", "error_pct"].mean()),
        "mean_signed_error_percent_gaslat": float(detail.loc[detail.muscle == "gaslat_r", "error_pct"].mean()),
    }
    (DATA / "nordez_direct_mtu_multi_subject_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
