"""Download the Nordez personalized GAS2b models for model comparison."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ultrasound_validation" / "nordez_mtu"
BASE_URL = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/"


def main():
    metadata = json.loads((DATA / "dataset_metadata.json").read_text())
    files = metadata["data"]["latestVersion"]["files"]
    manifest = []
    for number in range(1, 15):
        subject = f"S{number:02d}"
        label = f"Model_RaabeetChaudari_OS_v44_{subject}_Allpoints_GAS2b.osim"
        matches = [entry for entry in files if entry.get("label") == label]
        if len(matches) != 1:
            print(subject, "missing")
            continue
        entry = matches[0]
        identifier = entry["dataFile"]["persistentId"].rsplit("/", 1)[-1]
        output = DATA / label
        if not output.exists():
            with urlopen(BASE_URL + identifier, timeout=120) as response:
                output.write_bytes(response.read())
        manifest.append({"subject": subject, "model": str(output), "persistent_id": identifier})
        print(subject, output.name)
    (DATA / "nordez_personalized_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
