"""Summarize the corrected S01 multi-posture direct-MTU comparison."""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ultrasound_validation" / "nordez_mtu"


def pearson(x, y):
    return statistics.correlation(x, y)


def main():
    rows = [json.loads(Path(path).read_text()) for path in glob.glob(str(DATA / "S01_*_mtu_comparison.json"))]
    rows.sort(key=lambda row: row["posture"])
    summary = {"dataset": "Nordez/Guenanten", "subject": "S01", "postures": [row["posture"] for row in rows]}
    for muscle in ("gasmed_r", "gaslat_r"):
        direct = [row["direct_3dus_lengths_mm"][muscle] for row in rows]
        opensim = [row["opensim_lengths_mm"][muscle]["mean_mm"] for row in rows]
        signed_percent = [100.0 * (model - measured) / measured for measured, model in zip(direct, opensim)]
        summary[muscle] = {
            "direct_rom_mm": max(direct) - min(direct),
            "opensim_rom_mm": max(opensim) - min(opensim),
            "mean_absolute_error_mm": statistics.mean(abs(model - measured) for measured, model in zip(direct, opensim)),
            "mean_signed_error_percent": statistics.mean(signed_percent),
            "posture_level_pearson_r": pearson(direct, opensim),
        }
    output = DATA / "s01_multi_posture_mtu_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
