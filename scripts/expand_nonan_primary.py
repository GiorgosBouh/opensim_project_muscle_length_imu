"""Expand the NONAN primary demonstration with additional participants.

The selection rule and sagittal mapping match the original primary analysis:
three D01/B01 trials per participant and ten Gait2392 muscles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ilab/venv")
import top_tier_experiments as base  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/ilab/venv")
OUT = PROJECT / "nonan_expanded"
SUBJECTS = ["S135", "S146", "S140", "S142"]
TRIALS = {
    subject: [f"{subject}_G03_D01_B01_T0{trial}" for trial in (1, 2, 3)]
    for subject in SUBJECTS
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_norm = []
    scope = []

    for subject, trials in TRIALS.items():
        for trial in trials:
            csv_path = DATA_ROOT / subject / f"{trial}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            imu = pd.read_csv(csv_path)
            heel_strikes = base.detect_heel_strikes(imu)
            lengths = base.compute_lengths(imu, "sagittal", 0.0)
            normalized = base.normalize_cycles(
                imu, lengths, subject, trial, "sagittal", 0.0
            )
            all_norm.append(normalized)
            scope.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "n_frames": len(imu),
                    "n_detected_heel_strikes": len(heel_strikes),
                    "n_cycles": max(len(heel_strikes) - 1, 0),
                }
            )
            print(subject, trial, len(imu), max(len(heel_strikes) - 1, 0))

    normalized = pd.concat(all_norm, ignore_index=True)
    metrics = base.trial_metrics(normalized)
    trial_summary = (
        metrics.groupby(["subject", "muscle", "muscle_label"])
        .agg(
            n_trials=("trial", "nunique"),
            n_cycles=("n_cycles", "sum"),
            mean_rom_m=("rom_m", "mean"),
            sd_rom_m=("rom_m", "std"),
            mean_peak_phase_pct=("peak_phase_pct", "mean"),
            sd_peak_phase_pct=("peak_phase_pct", "std"),
        )
        .reset_index()
    )
    trial_summary["cv_rom_pct"] = trial_summary["sd_rom_m"] / trial_summary["mean_rom_m"] * 100

    normalized.to_csv(OUT / "nonan_expanded_normalized_cycles.csv", index=False)
    metrics.to_csv(OUT / "nonan_expanded_trial_metrics.csv", index=False)
    trial_summary.to_csv(OUT / "nonan_expanded_participant_muscle_summary.csv", index=False)
    pd.DataFrame(scope).to_csv(OUT / "nonan_expanded_trial_scope.csv", index=False)

    summary = {
        "participants": len(SUBJECTS),
        "subjects": SUBJECTS,
        "trials": len(scope),
        "cycles": int(sum(row["n_cycles"] for row in scope)),
        "muscles": len(base.ALL_MUSCLES),
        "median_participant_rom_cv_pct": float(trial_summary["cv_rom_pct"].median()),
        "median_participant_mean_rom_m": float(trial_summary["mean_rom_m"].median()),
        "participant_level_rows": int(len(trial_summary)),
    }
    (OUT / "nonan_expanded_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
