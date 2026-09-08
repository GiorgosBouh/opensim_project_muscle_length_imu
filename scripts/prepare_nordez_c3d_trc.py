"""Convert Nordez S01 static C3D files to OpenSim-compatible TRC files."""

from __future__ import annotations

import argparse
from pathlib import Path

import ezc3d


MODEL_MARKERS = [
    "rEIAS", "lEIAS", "rEIPS", "lEIPS", "rPICrest", "rAICrest",
    "lPICrest", "lAICrest", "rMedCond", "rLatCond", "rMedMall",
    "rLatMall", "rLatMeta", "rMedMeta", "rMidMeta", "rHip", "lHip",
    "rKnee", "rAnkle", "rCalcan",
]


def write_trc(c3d_path: Path, trc_path: Path) -> None:
    c3d = ezc3d.c3d(str(c3d_path))
    labels = [str(label).strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    label_to_index = {label: index for index, label in enumerate(labels)}
    selected = [label for label in MODEL_MARKERS if label in label_to_index]
    points = c3d["data"]["points"]
    rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    frames = points.shape[2]

    def tab(values):
        return chr(9).join(str(value) for value in values)

    lines = [
        tab(["PathFileType", "4", "(X/Y/Z)", trc_path.name]),
        tab(["DataRate", "CameraRate", "NumFrames", "NumMarkers", "Units", "OrigDataRate", "OrigDataStartFrame", "OrigNumFrames"]),
        tab([rate, rate, frames, len(selected), "m", rate, 1, frames]),
        tab(["Frame#", "Time"] + sum(([label, "", ""] for label in selected), [])),
        tab(["", ""] + sum(([f"X{i}", f"Y{i}", f"Z{i}"] for i in range(1, len(selected) + 1)), [])),
    ]
    for frame in range(frames):
        row = [frame + 1, frame / rate]
        for label in selected:
            index = label_to_index[label]
            row.extend(float(points[axis, index, frame]) / 1000.0 for axis in range(3))
        lines.append(tab(row))
    trc_path.write_text(chr(10).join(lines) + chr(10))
    print(f"{c3d_path.name}: {len(selected)}/{len(MODEL_MARKERS)} markers -> {trc_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("c3d", type=Path)
    parser.add_argument("trc", type=Path)
    args = parser.parse_args()
    write_trc(args.c3d, args.trc)


if __name__ == "__main__":
    main()
