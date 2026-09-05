# Data Sources and Download Instructions

This package does not include large raw datasets. The analyses use public datasets and a standard OpenSim lower limb model.

## NONAN GaitPrint

Purpose in this study: primary demonstration of the IMU joint angle to OpenSim MTU workflow.

- Dataset/article: NONAN GaitPrint, an IMU gait database of healthy older adults
- DOI: `10.6084/m9.figshare.27815034.v1`
- Figshare API manifest: `https://api.figshare.com/v2/articles/27815034`
- Subject and trial characteristics zip: `https://ndownloader.figshare.com/files/50640363`

Download metadata:

```bash
mkdir -p external_validation_data/nonan_figshare_27815034
cd external_validation_data/nonan_figshare_27815034
wget -N https://api.figshare.com/v2/articles/27815034 -O figshare_article_27815034.json
wget -N https://ndownloader.figshare.com/files/50640363 -O subject_trial_characteristics.zip
unzip -o subject_trial_characteristics.zip -d .
cd ../..
```

Expected metadata file:

```text
external_validation_data/nonan_figshare_27815034/subject_trial_characteristics/Gaitprint_subject_characteristics.csv
```

The primary trial folders used by the scripts should be placed at the repository root:

```text
S135/
S146/
```

Each folder should contain the corresponding processed commercial IMU joint angle CSV files and, if already generated, Gait2392 `.mot` files.

## Dorschky Lower Body IMU and Optical Motion Capture Dataset

Purpose in this study: external optical motion capture waveform reference.

- Dataset: Lower-body inertial sensor and optical motion capture recordings of gait and running
- DOI: `10.5281/zenodo.11522050`
- Zenodo API manifest: `https://zenodo.org/api/records/11522050`

Download the compact subset used in the manuscript analysis:

```bash
mkdir -p external_validation_data/dorschky_zenodo_11522050
cd external_validation_data/dorschky_zenodo_11522050
wget -N https://zenodo.org/api/records/11522050 -O zenodo_record_11522050.json
wget -N https://zenodo.org/api/records/11522050/files/README.txt/content -O README.txt
wget -N https://zenodo.org/api/records/11522050/files/ParticipantInfo.csv/content -O ParticipantInfo.csv
wget -N https://zenodo.org/api/records/11522050/files/P01_mean.parquet/content -O P01_mean.parquet
wget -N https://zenodo.org/api/records/11522050/files/P01_OMC.parquet/content -O P01_OMC.parquet
python -c "import pandas as pd; df = pd.read_parquet('P01_mean.parquet'); df.to_csv('P01_mean.csv', index=False); print(df.shape)"
cd ../..
```

The multi GB raw IMU parquet files are not required for the manuscript benchmark.

## GAITEX

Purpose in this study: same subject paired modality validation using wearable inertial and optical motion capture gait recordings.

- Dataset DOI: `10.5281/zenodo.15729055`
- Scientific Data article DOI: `10.1038/s41597-025-06439-x`

After downloading and extracting GAITEX, either place the data at:

```text
data/gaitex/
```

or point the script to the extracted dataset:

```bash
export GAITEX_DATA_DIR="/absolute/path/to/GAITEX/data"
```

The script expects the following structure for each natural gait participant:

```text
<participant>/ng/ik_imus/models/scaled_model_<participant>_ng.osim
<participant>/ng/ik_imus/marker_data_osim_format_<participant>_ng.trc
<participant>/ng/ik_imus/results_imu_ik/ik_segment_registered_imu_data_<participant>_ng.mot
<participant>/ng/timestamps_<participant>_ng.csv
```

The supplied GAITEX marker models contain scaled bodies, markers, and coordinates, but no muscle actuators. The script transfers their segment scale factors to Gait2392 and creates muscle models scaled to each participant.

## OpenSim Gait2392 Model

Purpose in this study: maps lower limb joint coordinates to muscle tendon paths.

This package includes:

```text
models/gait2392_simbody.osim
```

To use a different local Gait2392 file:

```bash
export GAIT2392_MODEL="/absolute/path/to/gait2392_simbody.osim"
```
