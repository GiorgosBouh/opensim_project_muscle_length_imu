"""Run OpenSim IK for Nordez static postures and summarize MTU lengths."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import re
from pathlib import Path

import opensim as osim


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ultrasound_validation" / "nordez_mtu"
POSTURE_LABELS = {"GAS1": "Static", "GAS2a": "APF", "GAS2b": "KF", "GAS3": "ADF", "GAS4": "KF_ADF", "GAS5": "Stand"}


def direct_length(filename: str) -> float:
    markup = json.loads((DATA / filename).read_text())
    measurement = next(item for item in markup["markups"][0]["measurements"]
                        if item["name"] == "length")
    return float(measurement["value"])


def run_ik(trc_path: Path, mot_path: Path, model_path: Path) -> None:
    tool = osim.InverseKinematicsTool()
    tool.set_model_file(str(model_path))
    tool.set_marker_file(str(trc_path))
    tool.set_output_motion_file(str(mot_path))
    tool.run()


def estimate_lengths(mot_path: Path, model_path: Path):
    table = osim.TimeSeriesTable(str(mot_path))
    model = osim.Model(str(model_path))
    state = model.initSystem()
    coordinates = model.updCoordinateSet()
    muscles = model.getMuscles()
    targets = {name: muscles.get(name) for name in ("gasmed_r", "gaslat_r")}
    columns = {str(name): table.getDependentColumn(name) for name in table.getColumnLabels()}
    values = {name: [] for name in targets}
    for row in range(table.getNumRows()):
        for name, column in columns.items():
            if name == "time":
                continue
            try:
                value = column.getElt(row, 0)
                if name not in {"pelvis_tx", "pelvis_ty", "pelvis_tz"}:
                    value = math.radians(value)
                coordinates.get(name).setValue(state, value, False)
            except Exception:
                continue
        model.realizePosition(state)
        for name, muscle in targets.items():
            values[name].append(muscle.getLength(state) * 1000.0)
    return {
        name: {
            "mean_mm": statistics.mean(series),
            "min_mm": min(series),
            "max_mm": max(series),
            "rom_mm": max(series) - min(series),
        }
        for name, series in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    parser.add_argument("posture", choices=sorted(POSTURE_LABELS))
    args = parser.parse_args()
    stem = f"{args.subject}_{args.posture}_{POSTURE_LABELS[args.posture]}"
    trc = Path("/tmp") / f"{stem}.trc"
    mot = DATA / f"{stem}_ik.mot"
    model_path = DATA / f"Model_RaabeetChaudari_OS_v44_{args.subject}.osim"
    run_ik(trc, mot, model_path)
    result = {
        "subject": args.subject,
        "posture": args.posture,
        "marker_file": str(trc),
        "direct_3dus_lengths_mm": {
            "gasmed_r": direct_length(f"{args.subject}_{args.posture}_GM_MTU.mrk.json"),
            "gaslat_r": direct_length(f"{args.subject}_{args.posture}_GL_MTU.mrk.json"),
        },
        "opensim_lengths_mm": estimate_lengths(mot, model_path),
    }
    output = DATA / f"{stem}_mtu_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
