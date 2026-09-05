from pathlib import Path
import json
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opensim

import top_tier_experiments as base


BASE_DIR = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUT_DIR = BASE_DIR / "validation_scaling_reproducibility"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "scaled_models"
REPRO_DIR = BASE_DIR / "reproducibility_bundle"

NONAN_META = (
    BASE_DIR
    / "external_validation_data"
    / "nonan_figshare_27815034"
    / "subject_trial_characteristics"
    / "Gaitprint_subject_characteristics.csv"
)
DORSCHKY_DIR = BASE_DIR / "external_validation_data" / "dorschky_zenodo_11522050"
DORSCHKY_MEAN = DORSCHKY_DIR / "P01_mean.parquet"
DORSCHKY_MEAN_CSV = DORSCHKY_DIR / "P01_mean.csv"
DORSCHKY_INFO = DORSCHKY_DIR / "ParticipantInfo.csv"

GENERIC_MASS_KG = 75.1646
GENERIC_HEIGHT_CM = 180.0
GENERIC_THIGH_CM = 39.6
GENERIC_SHANK_CM = 43.0
GENERIC_FOOT_CM = 25.5
GENERIC_TORSO_CM = 50.0


def ensure_dirs():
    for path in [OUT_DIR, FIG_DIR, MODEL_DIR, REPRO_DIR]:
        path.mkdir(exist_ok=True)


def add_scale(scale_set, segment, factor):
    scale = opensim.Scale()
    scale.setSegmentName(segment)
    scale.setScaleFactors(opensim.Vec3(float(factor), float(factor), float(factor)))
    scale_set.cloneAndAppend(scale)


def subject_scale_factors(row):
    height_factor = float(row["height"]) / GENERIC_HEIGHT_CM
    right_thigh = float(row["rt_thigh"])
    right_shank = float(row["rt_shank"])
    right_foot = float(row["rt_foot"])
    left_thigh = float(row["lt_thigh"])
    left_shank = float(row["lt_shank"])
    left_foot = float(row["lt_foot"])
    torso = float(row["lumbar_thoracic"])

    return {
        "pelvis": height_factor,
        "torso": torso / GENERIC_TORSO_CM,
        "femur_r": right_thigh / GENERIC_THIGH_CM,
        "tibia_r": right_shank / GENERIC_SHANK_CM,
        "talus_r": right_foot / GENERIC_FOOT_CM,
        "calcn_r": right_foot / GENERIC_FOOT_CM,
        "toes_r": right_foot / GENERIC_FOOT_CM,
        "femur_l": left_thigh / GENERIC_THIGH_CM,
        "tibia_l": left_shank / GENERIC_SHANK_CM,
        "talus_l": left_foot / GENERIC_FOOT_CM,
        "calcn_l": left_foot / GENERIC_FOOT_CM,
        "toes_l": left_foot / GENERIC_FOOT_CM,
    }


def build_scaled_model(subject, row):
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(base.MODEL_PATH))
    state = model.initSystem()
    scale_set = opensim.ScaleSet()
    factors = subject_scale_factors(row)
    for segment, factor in factors.items():
        add_scale(scale_set, segment, factor)
    ok = model.scale(state, scale_set, False, float(row["weight"]))
    if not ok:
        raise RuntimeError(f"OpenSim scaling failed for {subject}")
    out_path = MODEL_DIR / f"{subject}_gait2392_anthro_scaled.osim"
    model.printToXML(str(out_path))
    return out_path, factors


def load_model(model_path):
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    return model, state, model.getCoordinateSet(), model.getMuscles()


def compute_lengths_with_model(df, model_path, condition="legacy_hip3d"):
    mapping = base.available_mapping(df, condition)
    model, state, coord_set, muscles = load_model(model_path)
    time = df["time"].to_numpy()
    coord_arrays = {}
    for src_col, coord_name in mapping.items():
        coord_arrays[coord_name] = np.deg2rad(df[src_col].to_numpy(dtype=float))

    length_data = {"time": time.copy()}
    for muscle_name in base.ALL_MUSCLES:
        length_data[muscle_name + "_length"] = np.zeros(len(df), dtype=float)

    for i, t in enumerate(time):
        state.setTime(float(t))
        for coord_name, values in coord_arrays.items():
            coord_set.get(coord_name).setValue(state, float(values[i]), False)
        model.realizePosition(state)
        for muscle_name in base.ALL_MUSCLES:
            length_data[muscle_name + "_length"][i] = muscles.get(muscle_name).getLength(state)
    return pd.DataFrame(length_data)


def scaling_analysis():
    meta = pd.read_csv(NONAN_META)
    selected = meta[meta["id"].isin(base.TRIALS.keys())].copy()
    selected.to_csv(OUT_DIR / "nonan_subject_characteristics_S135_S146.csv", index=False)

    factor_rows = []
    all_norm = []
    for _, row in selected.iterrows():
        subject = row["id"]
        scaled_model, factors = build_scaled_model(subject, row)
        factor_rows.append(
            {
                "subject": subject,
                "height_cm": row["height"],
                "mass_kg": row["weight"],
                "generic_height_cm": GENERIC_HEIGHT_CM,
                "generic_mass_kg": GENERIC_MASS_KG,
                **{f"scale_{k}": v for k, v in factors.items()},
            }
        )
        for trial in base.TRIALS[subject]:
            imu = pd.read_csv(BASE_DIR / subject / f"{trial}.csv")
            unscaled = compute_lengths_with_model(imu, base.MODEL_PATH, "legacy_hip3d")
            scaled = compute_lengths_with_model(imu, scaled_model, "legacy_hip3d")
            all_norm.append(base.normalize_cycles(imu, unscaled, subject, trial, "unscaled", 0.0))
            all_norm.append(base.normalize_cycles(imu, scaled, subject, trial, "scaled", 0.0))

    factors_df = pd.DataFrame(factor_rows)
    factors_df.to_csv(OUT_DIR / "nonan_anthropometric_scale_factors.csv", index=False)

    norm = pd.concat(all_norm, ignore_index=True)
    metrics = base.trial_metrics(norm)
    metrics.to_csv(OUT_DIR / "scaled_vs_unscaled_trial_metrics.csv", index=False)

    sensitivity = compare_conditions(norm, "unscaled", "scaled", "scaled_vs_unscaled")
    sensitivity.to_csv(OUT_DIR / "scaled_vs_unscaled_waveform_sensitivity.csv", index=False)
    return factors_df, sensitivity, metrics


def compare_conditions(norm_df, base_condition, test_condition, label):
    rows = []
    means = (
        norm_df[norm_df["condition"].isin([base_condition, test_condition])]
        .groupby(["subject", "trial", "condition", "phase"])[[m + "_length" for m in base.ALL_MUSCLES]]
        .mean()
        .reset_index()
    )
    for subject in means["subject"].unique():
        for trial in means.loc[means["subject"] == subject, "trial"].unique():
            a_df = means[(means["subject"] == subject) & (means["trial"] == trial) & (means["condition"] == base_condition)]
            b_df = means[(means["subject"] == subject) & (means["trial"] == trial) & (means["condition"] == test_condition)]
            if a_df.empty or b_df.empty:
                continue
            for muscle in base.ALL_MUSCLES:
                a = a_df[muscle + "_length"].to_numpy()
                b = b_df[muscle + "_length"].to_numpy()
                rmse = float(np.sqrt(np.mean((b - a) ** 2)))
                rom = float(np.max(a) - np.min(a))
                rows.append(
                    {
                        "comparison": label,
                        "subject": subject,
                        "trial": trial,
                        "muscle": muscle,
                        "muscle_label": base.MUSCLE_LABELS[muscle],
                        "rmse_m": rmse,
                        "nrmse_pct_of_base_rom": float(rmse / rom * 100) if rom else np.nan,
                        "corr": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan,
                        "peak_delta_m": float(np.max(b) - np.max(a)),
                        "min_delta_m": float(np.min(b) - np.min(a)),
                        "rom_delta_pct": float(((np.max(b) - np.min(b)) - rom) / rom * 100) if rom else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def dorschky_mean_to_motion(df):
    out = pd.DataFrame()
    out["time"] = np.linspace(0, 1, len(df))
    out["Hip Flexion RT (deg)"] = np.rad2deg(df["HIP_FLEXION_ANGLE_R_MEAN"].to_numpy(dtype=float))
    # Dorschky labels this as knee extension; invert it to match the Noraxon/Gait2392 knee-flexion convention used here.
    out["Knee Flexion RT (deg)"] = -np.rad2deg(df["KNEE_EXTENSION_ANGLE_R_MEAN"].to_numpy(dtype=float))
    out["Ankle Dorsiflexion RT (deg)"] = np.rad2deg(df["ANKLE_DORSIFLEXION_ANGLE_R_MEAN"].to_numpy(dtype=float))
    out["Contact RT"] = 1
    return out


def dorschky_reference_analysis():
    if DORSCHKY_MEAN_CSV.exists():
        dorschky = pd.read_csv(DORSCHKY_MEAN_CSV)
    else:
        dorschky = pd.read_parquet(DORSCHKY_MEAN)
    info = pd.read_csv(DORSCHKY_INFO)
    info.to_csv(OUT_DIR / "dorschky_participant_info.csv", index=False)
    walks = ["slowwalking", "normwalking", "fastwalking"]
    rows = []
    wave_rows = []
    for movement in walks:
        movement_df = dorschky[dorschky["MOVEMENT"] == movement].reset_index(drop=True)
        motion = dorschky_mean_to_motion(movement_df)
        lengths = base.compute_lengths(motion, "sagittal", 0.0)
        phase = np.linspace(0, 100, len(lengths))
        for muscle in base.ALL_MUSCLES:
            values = lengths[muscle + "_length"].to_numpy()
            rows.append(
                {
                    "dataset": "Dorschky2019_P01_OMC_mean",
                    "movement": movement,
                    "muscle": muscle,
                    "muscle_label": base.MUSCLE_LABELS[muscle],
                    "peak_m": float(np.max(values)),
                    "min_m": float(np.min(values)),
                    "rom_m": float(np.max(values) - np.min(values)),
                    "peak_phase_pct": float(phase[int(np.argmax(values))]),
                    "min_phase_pct": float(phase[int(np.argmin(values))]),
                }
            )
            for p, v in zip(phase, values):
                wave_rows.append(
                    {
                        "dataset": "Dorschky2019_P01_OMC_mean",
                        "movement": movement,
                        "phase": float(p),
                        "muscle": muscle,
                        "muscle_label": base.MUSCLE_LABELS[muscle],
                        "length_m": float(v),
                    }
                )
    metrics = pd.DataFrame(rows)
    waves = pd.DataFrame(wave_rows)
    metrics.to_csv(OUT_DIR / "dorschky_p01_omc_reference_mtu_metrics.csv", index=False)
    waves.to_csv(OUT_DIR / "dorschky_p01_omc_reference_waveforms.csv", index=False)
    comparison = compare_nonan_to_dorschky(waves)
    comparison.to_csv(OUT_DIR / "nonan_vs_dorschky_omc_reference_comparison.csv", index=False)
    return metrics, waves, comparison


def compare_nonan_to_dorschky(dorschky_waves):
    cols = ["condition", "perturb_deg", "subject", "phase"] + [m + "_length" for m in base.ALL_MUSCLES]
    nonan = pd.read_csv(BASE_DIR / "top_tier_experiments" / "normalized_mtu_cycles.csv", usecols=cols)
    nonan = nonan[(nonan["condition"] == "sagittal") & (nonan["perturb_deg"] == 0)]
    nonan_mean = nonan.groupby("phase")[[m + "_length" for m in base.ALL_MUSCLES]].mean().reset_index()
    rows = []
    for movement in sorted(dorschky_waves["movement"].unique()):
        mov = dorschky_waves[dorschky_waves["movement"] == movement]
        for muscle in base.ALL_MUSCLES:
            ref = mov[mov["muscle"] == muscle].sort_values("phase")
            ref_y = np.interp(nonan_mean["phase"], ref["phase"], ref["length_m"])
            nonan_y = nonan_mean[muscle + "_length"].to_numpy()
            rmse = float(np.sqrt(np.mean((nonan_y - ref_y) ** 2)))
            ref_rom = float(np.max(ref_y) - np.min(ref_y))
            corr = float(np.corrcoef(nonan_y, ref_y)[0, 1]) if np.std(nonan_y) > 0 and np.std(ref_y) > 0 else np.nan
            rows.append(
                {
                    "reference_dataset": "Dorschky2019_P01_OMC_mean",
                    "reference_movement": movement,
                    "nonan_dataset": "NONAN_GaitPrint_older_S135_S146",
                    "muscle": muscle,
                    "muscle_label": base.MUSCLE_LABELS[muscle],
                    "rmse_m": rmse,
                    "nrmse_pct_of_reference_rom": float(rmse / ref_rom * 100) if ref_rom else np.nan,
                    "corr": corr,
                    "nonan_rom_m": float(np.max(nonan_y) - np.min(nonan_y)),
                    "reference_rom_m": ref_rom,
                }
            )
    return pd.DataFrame(rows)


def make_plots(scaling_sensitivity, dorschky_metrics, dorschky_comparison):
    scaling_summary = (
        scaling_sensitivity.groupby(["muscle", "muscle_label"])["nrmse_pct_of_base_rom"]
        .mean()
        .reset_index()
        .sort_values("nrmse_pct_of_base_rom", ascending=False)
    )
    plt.figure(figsize=(9, 4))
    plt.bar(scaling_summary["muscle_label"], scaling_summary["nrmse_pct_of_base_rom"], color="#2A9D8F")
    plt.ylabel("Scaled vs unscaled NRMSE (% unscaled ROM)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scaled_vs_unscaled_sensitivity.png", dpi=300)
    plt.close()

    ham = dorschky_metrics[dorschky_metrics["muscle"].isin(base.HAMSTRING_MUSCLES)]
    pivot = ham.pivot_table(index="muscle_label", columns="movement", values="rom_m")
    pivot = pivot.loc[[base.MUSCLE_LABELS[m] for m in base.HAMSTRING_MUSCLES]]
    pivot.plot(kind="bar", figsize=(8, 4), color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("ROM (m)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dorschky_hamstring_rom_reference.png", dpi=300)
    plt.close()

    comp = dorschky_comparison[dorschky_comparison["reference_movement"] == "normwalking"].copy()
    comp = comp.sort_values("corr", ascending=True)
    plt.figure(figsize=(9, 4))
    plt.bar(comp["muscle_label"], comp["corr"], color="#7A5195")
    plt.ylabel("Waveform correlation with P01 OMC normwalking")
    plt.ylim(-1, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "nonan_vs_dorschky_normwalking_correlations.png", dpi=300)
    plt.close()


def write_repro_bundle(summary):
    for script in [
        "top_tier_experiments.py",
        "validation_scaling_reproducibility.py",
        "batch_csv_to_mot.py",
        "run_muscle_lengths.py",
        "plot_all_subjects.py",
    ]:
        src = BASE_DIR / script
        if src.exists():
            shutil.copy2(src, REPRO_DIR / script)

    small_outputs = REPRO_DIR / "outputs"
    small_outputs.mkdir(exist_ok=True)
    for src in [
        OUT_DIR / "nonan_subject_characteristics_S135_S146.csv",
        OUT_DIR / "nonan_anthropometric_scale_factors.csv",
        OUT_DIR / "scaled_vs_unscaled_waveform_sensitivity.csv",
        OUT_DIR / "dorschky_p01_omc_reference_mtu_metrics.csv",
        OUT_DIR / "nonan_vs_dorschky_omc_reference_comparison.csv",
        BASE_DIR / "top_tier_experiments" / "sagittal_vs_3d_sensitivity.csv",
        BASE_DIR / "top_tier_experiments" / "uncertainty_joint_offset_summary.csv",
    ]:
        if src.exists():
            shutil.copy2(src, small_outputs / src.name)

    readme = f"""# Reproducibility bundle

This bundle collects the scripts and compact outputs needed to reproduce the
workflow demonstration for IMU-derived OpenSim muscle-tendon lengths.

## Main paths

- Project root: repository root
- Manuscript: `manuscript/main.tex` if included
- Compiled PDF: `manuscript/main.pdf` if included
- OpenSim model: set with `GAIT2392_MODEL`
- OpenSim Python: Python environment with OpenSim installed

## Data

The bundled scripts expect the local NONAN trial folders:

- `S135/`
- `S146/`

Additional public metadata/reference subsets downloaded during this revision:

- NONAN older-adult Figshare metadata DOI: `10.6084/m9.figshare.27815034.v1`
- Dorschky OMC/IMU Zenodo dataset DOI: `10.5281/zenodo.11522050`

Only small metadata and P01 OMC/mean files were downloaded, not the multi-GB
raw IMU archives.

## Commands

```bash
cd /path/to/repository
python top_tier_experiments.py
python validation_scaling_reproducibility.py
cd /path/to/repository/latex
tectonic --keep-logs main.tex
```

## Summary

```json
{json.dumps(summary, indent=2)}
```
"""
    (REPRO_DIR / "README.md").write_text(readme)

    reqs = """pandas
numpy
matplotlib
pyarrow
opensim
"""
    (REPRO_DIR / "requirements.txt").write_text(reqs)


def main():
    ensure_dirs()
    factors, scaling_sensitivity, scaling_metrics = scaling_analysis()
    dorschky_metrics, dorschky_waves, dorschky_comparison = dorschky_reference_analysis()
    make_plots(scaling_sensitivity, dorschky_metrics, dorschky_comparison)

    summary = {
        "nonan_subjects_scaled": int(factors["subject"].nunique()),
        "nonan_trials_in_scaling_analysis": int(scaling_sensitivity[["subject", "trial"]].drop_duplicates().shape[0]),
        "max_scaled_vs_unscaled_nrmse_pct_rom": float(scaling_sensitivity["nrmse_pct_of_base_rom"].max()),
        "mean_scaled_vs_unscaled_nrmse_pct_rom": float(scaling_sensitivity["nrmse_pct_of_base_rom"].mean()),
        "dorschky_reference_subjects_downloaded": 1,
        "dorschky_reference_movements": sorted(dorschky_metrics["movement"].unique().tolist()),
        "mean_nonan_vs_dorschky_normwalking_corr": float(
            dorschky_comparison[dorschky_comparison["reference_movement"] == "normwalking"]["corr"].mean()
        ),
        "median_nonan_vs_dorschky_normwalking_corr": float(
            dorschky_comparison[dorschky_comparison["reference_movement"] == "normwalking"]["corr"].median()
        ),
        "downloaded_nonan_metadata_doi": "10.6084/m9.figshare.27815034.v1",
        "downloaded_dorschky_dataset_doi": "10.5281/zenodo.11522050",
    }
    (OUT_DIR / "validation_scaling_reproducibility_summary.json").write_text(json.dumps(summary, indent=2))
    write_repro_bundle(summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {OUT_DIR}")
    print(f"Wrote reproducibility bundle to {REPRO_DIR}")


if __name__ == "__main__":
    main()
