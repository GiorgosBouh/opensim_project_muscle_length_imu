"""Download the small Nordez direct-MTU validation subset.

The full repository is several gigabytes because it contains ultrasound image
volumes. This script downloads only C3D, direct MTU Markups, and generic
OpenSim model files for the static gastrocnemius postures.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ultrasound_validation" / "nordez_mtu"
METADATA = DATA / "dataset_metadata.json"
BASE_URL = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/"
POSTURES = {
    "GAS1": "Static",
    "GAS2a": "APF",
    "GAS2b": "KF",
    "GAS3": "ADF",
    "GAS4": "KF_ADF",
    "GAS5": "Stand",
}


def entries():
    return json.loads(METADATA.read_text())["data"]["latestVersion"]["files"]


def persistent_id(entry):
    return entry["dataFile"]["persistentId"].rsplit("/", 1)[-1]


def find_file(files, label, directory_fragment=None):
    matches = [entry for entry in files if entry.get("label") == label]
    if directory_fragment is not None:
        matches = [entry for entry in matches if directory_fragment in entry.get("directoryLabel", "")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {label} / {directory_fragment}, found {len(matches)}")
    return matches[0]


def download(entry, output):
    if output.exists() and output.stat().st_size > 0:
        return "existing"
    url = BASE_URL + persistent_id(entry)
    with urlopen(url, timeout=120) as response:
        output.write_bytes(response.read())
    return "downloaded"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    files = entries()
    manifest = []
    for subject_number in range(1, 15):
        subject = f"S{subject_number:02d}"
        model_label = f"Model_RaabeetChaudari_OS_v44_{subject}.osim"
        try:
            model = find_file(files, model_label, "/OpenSim/Models/Generic")
        except RuntimeError as error:
            print(error)
            continue
        model_path = DATA / model_label
        print(subject, "model", download(model, model_path))
        for posture, posture_label in POSTURES.items():
            c3d_label = f"{subject}_{posture}_" + {
                "GAS1": "Static",
                "GAS2a": "APF",
                "GAS2b": "KF",
                "GAS3": "ADF",
                "GAS4": "KF_ADF",
                "GAS5": "Stand",
            }[posture] + ".c3d"
            try:
                c3d = find_file(files, c3d_label, "/OpenSim/C3D/dataC3D")
                gm = find_file(files, "GM MTU.mrk.json", f"/3DUS/{subject}/{posture}")
                gl = find_file(files, "GL MTU.mrk.json", f"/3DUS/{subject}/{posture}")
            except RuntimeError as error:
                print(subject, posture, error)
                continue
            outputs = {
                "c3d": DATA / c3d_label,
                "gm": DATA / f"{subject}_{posture}_GM_MTU.mrk.json",
                "gl": DATA / f"{subject}_{posture}_GL_MTU.mrk.json",
            }
            statuses = {kind: download(entry, outputs[kind]) for kind, entry in {"c3d": c3d, "gm": gm, "gl": gl}.items()}
            manifest.append({"subject": subject, "posture": posture, "model": str(model_path), **{key: str(value) for key, value in outputs.items()}, "status": statuses})
    (DATA / "nordez_validation_subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} posture records")


if __name__ == "__main__":
    main()
