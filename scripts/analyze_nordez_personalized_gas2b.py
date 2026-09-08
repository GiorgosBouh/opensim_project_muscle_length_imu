"""Compare generic and ultrasound-personalized Nordez GAS2b models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_nordez_static_ik import DATA, estimate_lengths, run_ik


def direct_length(path: Path) -> float:
    data = json.loads(path.read_text())
    item = next(item for item in data["markups"][0]["measurements"] if item["name"] == "length")
    return float(item["value"])


def main():
    rows = []
    for number in range(1, 15):
        subject = f"S{number:02d}"
        trc = Path("/tmp") / f"{subject}_GAS2b_KF.trc"
        if not trc.exists():
            continue
        generic_model = DATA / f"Model_RaabeetChaudari_OS_v44_{subject}.osim"
        personalized_model = DATA / f"Model_RaabeetChaudari_OS_v44_{subject}_Allpoints_GAS2b.osim"
        output_mot = DATA / f"{subject}_GAS2b_KF_personalized_ik.mot"
        run_ik(trc, output_mot, personalized_model)
        generic = json.loads((DATA / f"{subject}_GAS2b_KF_mtu_comparison.json").read_text())
        personalized = estimate_lengths(output_mot, personalized_model)
        for muscle in ("gasmed_r", "gaslat_r"):
            measured = direct_length(DATA / f"{subject}_GAS2b_GM_MTU.mrk.json" if muscle == "gasmed_r" else DATA / f"{subject}_GAS2b_GL_MTU.mrk.json")
            rows.append({
                "subject": subject,
                "muscle": muscle,
                "direct_3dus_mm": measured,
                "generic_opensim_mm": generic["opensim_lengths_mm"][muscle]["mean_mm"],
                "personalized_opensim_mm": personalized[muscle]["mean_mm"],
            })
    result = pd.DataFrame(rows)
    result["generic_error_pct"] = 100 * (result["generic_opensim_mm"] - result["direct_3dus_mm"]) / result["direct_3dus_mm"]
    result["personalized_error_pct"] = 100 * (result["personalized_opensim_mm"] - result["direct_3dus_mm"]) / result["direct_3dus_mm"]
    result.to_csv(DATA / "nordez_personalized_gas2b_comparison.csv", index=False)
    summary = result.groupby("muscle").agg(
        n_subjects=("subject", "nunique"),
        median_abs_generic_error_pct=("generic_error_pct", lambda x: float(x.abs().median())),
        median_abs_personalized_error_pct=("personalized_error_pct", lambda x: float(x.abs().median())),
        mean_signed_generic_error_pct=("generic_error_pct", "mean"),
        mean_signed_personalized_error_pct=("personalized_error_pct", "mean"),
    ).reset_index()
    summary.to_csv(DATA / "nordez_personalized_gas2b_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
