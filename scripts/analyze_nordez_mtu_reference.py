"""Pilot comparison of direct 3DUS MTU lengths with an OpenSim model.

This script uses the supplied S01 GAS1 static C3D after the C3D-to-TRC
conversion used for the pilot IK run. It is intentionally a small feasibility
analysis; the marker-fit error must be reported with any resulting comparison.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import opensim as osim


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ultrasound_validation" / "nordez_mtu"
IK_PATH = Path("/tmp/S01_GAS1_ik.mot")


def summarize(values):
    values_mm = [value * 1000.0 for value in values]
    return {
        "mean_mm": statistics.mean(values_mm),
        "min_mm": min(values_mm),
        "max_mm": max(values_mm),
        "rom_mm": max(values_mm) - min(values_mm),
    }


def main():
    table = osim.TimeSeriesTable(str(IK_PATH))
    model = osim.Model(str(DATA / "S01_generic_model.osim"))
    state = model.initSystem()
    coordinates = model.updCoordinateSet()
    muscles = model.getMuscles()
    muscle_names = [muscles.get(index).getName() for index in range(muscles.getSize())]
    columns = {name: table.getDependentColumn(name) for name in table.getColumnLabels()}
    values = {name: [] for name in muscle_names}

    for row in range(table.getNumRows()):
        for name in table.getColumnLabels():
            if name == "time":
                continue
            try:
                coordinates.get(name).setValue(state, columns[name].getElt(row, 0), False)
            except Exception:
                # The model does not contain every auxiliary IK coordinate.
                continue
        model.realizePosition(state)
        for name in muscle_names:
            values[name].append(muscles.get(name).getLength(state))

    direct = {}
    for label, filename in {
        "gasmed_r": "S01_GAS1_GM_MTU.mrk.json",
        "gaslat_r": "S01_GAS1_GL_MTU.mrk.json",
    }.items():
        markup = json.loads((DATA / filename).read_text())
        measurement = next(item for item in markup["markups"][0]["measurements"]
                            if item["name"] == "length")
        direct[label] = measurement["value"]

    result = {
        "dataset": "Nordez/Guenanten 3D ultrasound MTU dataset",
        "subject": "S01",
        "posture": "GAS1",
        "ik_file": str(IK_PATH),
        "n_ik_frames": table.getNumRows(),
        "direct_3dus_lengths_mm": direct,
        "opensim_generic_model_lengths": {
            name: summarize(values[name]) for name in ("gasmed_r", "gaslat_r")
        },
        "interpretation": (
            "Pilot comparison only. The C3D-to-TRC IK run used 20 of 24 model "
            "markers and must be replaced or checked against the official setup "
            "before treating this as validation."
        ),
    }
    output = DATA / "s01_gas1_mtu_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
