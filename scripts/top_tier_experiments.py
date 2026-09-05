import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import opensim


BASE_DIR = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
MODEL_PATH = Path(
    os.environ.get("GAIT2392_MODEL", BASE_DIR / "models" / "gait2392_simbody.osim")
).expanduser()
OUT_DIR = BASE_DIR / "top_tier_experiments"
FIG_DIR = OUT_DIR / "figures"

TRIALS = {
    "S135": [
        "S135_G03_D01_B01_T01",
        "S135_G03_D01_B01_T02",
        "S135_G03_D01_B01_T03",
    ],
    "S146": [
        "S146_G03_D01_B01_T01",
        "S146_G03_D01_B01_T02",
        "S146_G03_D01_B01_T03",
    ],
}

PRIMARY_MUSCLES = [
    "glut_med1_r",
    "rect_fem_r",
    "vas_lat_r",
    "tib_ant_r",
    "soleus_r",
    "med_gas_r",
]

HAMSTRING_MUSCLES = [
    "semimem_r",
    "semiten_r",
    "bifemlh_r",
    "bifemsh_r",
]

ALL_MUSCLES = PRIMARY_MUSCLES + HAMSTRING_MUSCLES

MUSCLE_LABELS = {
    "glut_med1_r": "Glut. medius",
    "rect_fem_r": "Rect. femoris",
    "vas_lat_r": "Vast. lateralis",
    "tib_ant_r": "Tib. anterior",
    "soleus_r": "Soleus",
    "med_gas_r": "Med. gastrocnemius",
    "semimem_r": "Semimembranosus",
    "semiten_r": "Semitendinosus",
    "bifemlh_r": "Biceps femoris LH",
    "bifemsh_r": "Biceps femoris SH",
}

SAGITTAL_MAP = {
    "Hip Flexion RT (deg)": "hip_flexion_r",
    "Knee Flexion RT (deg)": "knee_angle_r",
    "Ankle Dorsiflexion RT (deg)": "ankle_angle_r",
}

LEGACY_HIP3D_EXTRA_MAP = {
    "Hip Abduction RT (deg)": "hip_adduction_r",
    "Hip Rotation Ext RT (deg)": "hip_rotation_r",
}

EXPANDED_3D_EXTRA_MAP = {
    "Pelvic Tilt LT (deg)": "pelvis_tilt",
    "Pelvic Obliquity RT (deg)": "pelvis_list",
    "Pelvic Rotation RT (deg)": "pelvis_rotation",
}

N_PHASE = 101


def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def detect_heel_strikes(df):
    if "Contact RT" in df.columns:
        contact = df["Contact RT"].to_numpy()
        return [i for i in range(1, len(contact)) if contact[i - 1] < 0.5 <= contact[i]]

    col = "Noraxon MyoMotion-Trajectories-Heel RT-y (mm)"
    if col not in df.columns:
        raise ValueError("Missing Contact RT and heel trajectory columns.")
    y = df[col].to_numpy()
    hs = []
    min_distance = 50
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] < y[i + 1]:
            if not hs or i - hs[-1] > min_distance:
                hs.append(i)
    return hs


def build_model():
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(MODEL_PATH))
    state = model.initSystem()
    return model, state, model.getCoordinateSet(), model.getMuscles()


def available_mapping(df, condition):
    mapping = dict(SAGITTAL_MAP)
    if condition in {"legacy_hip3d", "expanded_3d"}:
        mapping.update(LEGACY_HIP3D_EXTRA_MAP)
    if condition == "expanded_3d":
        mapping.update(EXPANDED_3D_EXTRA_MAP)
    missing = [src for src in mapping if src not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {condition}: {missing}")
    return mapping


def compute_lengths(df, condition, perturb_deg=0.0):
    mapping = available_mapping(df, condition)
    model, state, coord_set, muscles = build_model()
    time = df["time"].to_numpy()

    coord_arrays = {}
    for src_col, coord_name in mapping.items():
        values = df[src_col].to_numpy(dtype=float)
        if src_col in SAGITTAL_MAP:
            values = values + perturb_deg
        coord_arrays[coord_name] = np.deg2rad(values)

    length_data = {"time": time.copy()}
    for muscle_name in ALL_MUSCLES:
        length_data[muscle_name + "_length"] = np.zeros(len(df), dtype=float)

    for i, t in enumerate(time):
        state.setTime(float(t))
        for coord_name, values in coord_arrays.items():
            coord_set.get(coord_name).setValue(state, float(values[i]), False)
        model.realizePosition(state)
        for muscle_name in ALL_MUSCLES:
            length_data[muscle_name + "_length"][i] = muscles.get(muscle_name).getLength(state)

    return pd.DataFrame(length_data)


def normalize_cycles(imu_df, length_df, subject, trial, condition, perturb_deg):
    hs = detect_heel_strikes(imu_df)
    phase_new = np.linspace(0, 100, N_PHASE)
    records = []

    for cycle_idx in range(len(hs) - 1):
        start = hs[cycle_idx]
        end = hs[cycle_idx + 1]
        if end <= start + 5:
            continue
        x_old = np.linspace(0, 1, end - start + 1)
        x_new = np.linspace(0, 1, N_PHASE)
        base = {
            "subject": subject,
            "trial": trial,
            "condition": condition,
            "perturb_deg": perturb_deg,
            "cycle": cycle_idx,
        }
        interpolated = {}
        for muscle_name in ALL_MUSCLES:
            col = muscle_name + "_length"
            interpolated[col] = np.interp(x_new, x_old, length_df[col].to_numpy()[start : end + 1])
        for k, phase in enumerate(phase_new):
            row = dict(base)
            row["phase"] = phase
            for col, values in interpolated.items():
                row[col] = values[k]
            records.append(row)

    return pd.DataFrame.from_records(records)


def trial_metrics(norm_df):
    rows = []
    for keys, group in norm_df.groupby(["subject", "trial", "condition", "perturb_deg"]):
        subject, trial, condition, perturb_deg = keys
        n_cycles = int(group["cycle"].nunique())
        mean_by_phase = group.groupby("phase")[[m + "_length" for m in ALL_MUSCLES]].mean()
        phases = mean_by_phase.index.to_numpy(dtype=float)
        for muscle_name in ALL_MUSCLES:
            values = mean_by_phase[muscle_name + "_length"].to_numpy()
            peak_i = int(np.argmax(values))
            min_i = int(np.argmin(values))
            rows.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "condition": condition,
                    "perturb_deg": perturb_deg,
                    "muscle": muscle_name,
                    "muscle_label": MUSCLE_LABELS[muscle_name],
                    "n_cycles": n_cycles,
                    "peak_m": float(values[peak_i]),
                    "min_m": float(values[min_i]),
                    "rom_m": float(values[peak_i] - values[min_i]),
                    "peak_phase_pct": float(phases[peak_i]),
                    "min_phase_pct": float(phases[min_i]),
                }
            )
    return pd.DataFrame(rows)


def compare_waveforms(norm_df, base_condition, test_condition):
    rows = []
    phase_means = (
        norm_df[norm_df["condition"].isin([base_condition, test_condition])]
        .query("perturb_deg == 0")
        .groupby(["subject", "trial", "condition", "phase"])[[m + "_length" for m in ALL_MUSCLES]]
        .mean()
        .reset_index()
    )

    for subject in TRIALS:
        for trial in TRIALS[subject]:
            base = phase_means[
                (phase_means["subject"] == subject)
                & (phase_means["trial"] == trial)
                & (phase_means["condition"] == base_condition)
            ]
            test = phase_means[
                (phase_means["subject"] == subject)
                & (phase_means["trial"] == trial)
                & (phase_means["condition"] == test_condition)
            ]
            if base.empty or test.empty:
                continue
            for muscle_name in ALL_MUSCLES:
                a = base[muscle_name + "_length"].to_numpy()
                b = test[muscle_name + "_length"].to_numpy()
                rmse = float(np.sqrt(np.mean((b - a) ** 2)))
                rom = float(np.max(a) - np.min(a))
                corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
                rows.append(
                    {
                        "subject": subject,
                        "trial": trial,
                        "muscle": muscle_name,
                        "muscle_label": MUSCLE_LABELS[muscle_name],
                        "rmse_m": rmse,
                        "nrmse_pct_of_sagittal_rom": float(rmse / rom * 100) if rom else np.nan,
                        "corr": corr,
                        "peak_delta_pct": float((np.max(b) - np.max(a)) / np.max(a) * 100),
                        "min_delta_pct": float((np.min(b) - np.min(a)) / np.min(a) * 100),
                        "rom_delta_pct": float(((np.max(b) - np.min(b)) - rom) / rom * 100) if rom else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def compare_to_existing(length_df, subject, trial):
    existing_path = BASE_DIR / f"{trial}_muscles.csv"
    if not existing_path.exists():
        return []
    existing = pd.read_csv(existing_path)
    n = min(len(existing), len(length_df))
    rows = []
    for muscle_name in PRIMARY_MUSCLES:
        col = muscle_name + "_length"
        if col not in existing.columns:
            continue
        a = existing[col].to_numpy(dtype=float)[:n]
        b = length_df[col].to_numpy(dtype=float)[:n]
        rmse = float(np.sqrt(np.mean((b - a) ** 2)))
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
        rows.append(
            {
                "subject": subject,
                "trial": trial,
                "muscle": muscle_name,
                "muscle_label": MUSCLE_LABELS[muscle_name],
                "n_frames_compared": n,
                "rmse_m": rmse,
                "max_abs_diff_m": float(np.max(np.abs(b - a))),
                "corr": corr,
            }
        )
    return rows


def summarize_metrics(metrics):
    baseline = metrics[(metrics["condition"] == "sagittal") & (metrics["perturb_deg"] == 0)]
    grouped = baseline.groupby(["muscle", "muscle_label"])
    rows = []
    for (muscle, label), group in grouped:
        for metric in ["peak_m", "min_m", "rom_m"]:
            mean = float(group[metric].mean())
            sd = float(group[metric].std(ddof=1))
            rows.append(
                {
                    "muscle": muscle,
                    "muscle_label": label,
                    "metric": metric,
                    "n_trials": int(len(group)),
                    "mean": mean,
                    "sd": sd,
                    "cv_pct": float(sd / mean * 100) if mean else np.nan,
                }
            )
    return pd.DataFrame(rows)


def uncertainty_summary(metrics):
    base = metrics[(metrics["condition"] == "sagittal") & (metrics["perturb_deg"] == 0)]
    pert = metrics[(metrics["condition"] == "sagittal") & (metrics["perturb_deg"] != 0)]
    merged = pert.merge(
        base,
        on=["subject", "trial", "muscle", "muscle_label"],
        suffixes=("_perturbed", "_baseline"),
    )
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "subject": row["subject"],
                "trial": row["trial"],
                "muscle": row["muscle"],
                "muscle_label": row["muscle_label"],
                "perturb_deg": row["perturb_deg_perturbed"],
                "peak_delta_pct": (row["peak_m_perturbed"] - row["peak_m_baseline"]) / row["peak_m_baseline"] * 100,
                "min_delta_pct": (row["min_m_perturbed"] - row["min_m_baseline"]) / row["min_m_baseline"] * 100,
                "rom_delta_pct": (row["rom_m_perturbed"] - row["rom_m_baseline"]) / row["rom_m_baseline"] * 100,
                "peak_phase_delta_pct_gait": row["peak_phase_pct_perturbed"] - row["peak_phase_pct_baseline"],
                "min_phase_delta_pct_gait": row["min_phase_pct_perturbed"] - row["min_phase_pct_baseline"],
            }
        )
    out = pd.DataFrame(rows)
    return (
        out.groupby(["muscle", "muscle_label", "perturb_deg"])
        .agg(
            peak_delta_mean_pct=("peak_delta_pct", "mean"),
            peak_delta_absmax_pct=("peak_delta_pct", lambda x: float(np.max(np.abs(x)))),
            min_delta_mean_pct=("min_delta_pct", "mean"),
            min_delta_absmax_pct=("min_delta_pct", lambda x: float(np.max(np.abs(x)))),
            rom_delta_mean_pct=("rom_delta_pct", "mean"),
            rom_delta_absmax_pct=("rom_delta_pct", lambda x: float(np.max(np.abs(x)))),
            peak_phase_absmax_pct_gait=("peak_phase_delta_pct_gait", lambda x: float(np.max(np.abs(x)))),
            min_phase_absmax_pct_gait=("min_phase_delta_pct_gait", lambda x: float(np.max(np.abs(x)))),
        )
        .reset_index()
    )


def make_plots(norm_df, sensitivity, uncertainty):
    sens_summary = (
        sensitivity.groupby(["muscle", "muscle_label"])["nrmse_pct_of_sagittal_rom"]
        .mean()
        .reset_index()
        .sort_values("nrmse_pct_of_sagittal_rom", ascending=False)
    )
    plt.figure(figsize=(9, 4))
    plt.bar(sens_summary["muscle_label"], sens_summary["nrmse_pct_of_sagittal_rom"], color="#4C78A8")
    plt.ylabel("Expanded 3D vs sagittal-only NRMSE (% sagittal ROM)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sagittal_vs_3d_sensitivity.png", dpi=300)
    plt.close()

    ham = norm_df[(norm_df["condition"] == "sagittal") & (norm_df["perturb_deg"] == 0)]
    means = ham.groupby(["subject", "phase"])[[m + "_length" for m in HAMSTRING_MUSCLES]].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, muscle_name in zip(axes.flat, HAMSTRING_MUSCLES):
        for subject, color in [("S135", "#4C78A8"), ("S146", "#F58518")]:
            sub = means[means["subject"] == subject]
            ax.plot(sub["phase"], sub[muscle_name + "_length"], label=subject, color=color)
        ax.set_title(MUSCLE_LABELS[muscle_name])
        ax.set_ylabel("Length (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Gait cycle (%)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hamstring_overlays.png", dpi=300)
    plt.close(fig)

    unc = uncertainty.groupby(["perturb_deg"])["rom_delta_absmax_pct"].max().reset_index()
    plt.figure(figsize=(6, 4))
    plt.plot(unc["perturb_deg"], unc["rom_delta_absmax_pct"], marker="o", color="#E45756")
    plt.xlabel("Uniform sagittal joint-angle offset (deg)")
    plt.ylabel("Maximum absolute ROM change (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "uncertainty_rom_change.png", dpi=300)
    plt.close()


def main():
    ensure_dirs()
    all_norm = []
    repro_rows = []
    scope_rows = []

    conditions = [("sagittal", 0.0), ("legacy_hip3d", 0.0), ("expanded_3d", 0.0)]
    for perturb in [-5.0, -2.0, 2.0, 5.0]:
        conditions.append(("sagittal", perturb))

    for subject, trials in TRIALS.items():
        for trial in trials:
            imu_path = BASE_DIR / subject / f"{trial}.csv"
            imu_df = pd.read_csv(imu_path)
            hs = detect_heel_strikes(imu_df)
            scope_rows.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "n_frames": len(imu_df),
                    "n_detected_heel_strikes": len(hs),
                    "n_cycles": max(len(hs) - 1, 0),
                }
            )
            print(f"{subject} {trial}: {len(imu_df)} frames, {max(len(hs) - 1, 0)} cycles")

            for condition, perturb_deg in conditions:
                print(f"  computing {condition}, perturb={perturb_deg:+.1f} deg")
                length_df = compute_lengths(imu_df, condition, perturb_deg)
                if condition == "legacy_hip3d" and perturb_deg == 0:
                    repro_rows.extend(compare_to_existing(length_df, subject, trial))
                norm_df = normalize_cycles(imu_df, length_df, subject, trial, condition, perturb_deg)
                all_norm.append(norm_df)

    all_norm_df = pd.concat(all_norm, ignore_index=True)
    metrics = trial_metrics(all_norm_df)
    sensitivity_legacy = compare_waveforms(all_norm_df, "sagittal", "legacy_hip3d")
    sensitivity = compare_waveforms(all_norm_df, "sagittal", "expanded_3d")
    uncertainty = uncertainty_summary(metrics)
    benchmark = summarize_metrics(metrics)
    hamstrings = metrics[
        (metrics["condition"] == "sagittal")
        & (metrics["perturb_deg"] == 0)
        & (metrics["muscle"].isin(HAMSTRING_MUSCLES))
    ].copy()
    repro = pd.DataFrame(repro_rows)
    scope = pd.DataFrame(scope_rows)

    all_norm_df.to_csv(OUT_DIR / "normalized_mtu_cycles.csv", index=False)
    metrics.to_csv(OUT_DIR / "mtu_trial_metrics.csv", index=False)
    sensitivity_legacy.to_csv(OUT_DIR / "sagittal_vs_legacy_hip3d_sensitivity.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "sagittal_vs_expanded_3d_sensitivity.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "sagittal_vs_3d_sensitivity.csv", index=False)
    uncertainty.to_csv(OUT_DIR / "uncertainty_joint_offset_summary.csv", index=False)
    benchmark.to_csv(OUT_DIR / "reproducible_benchmark_summary.csv", index=False)
    hamstrings.to_csv(OUT_DIR / "hamstring_trial_metrics.csv", index=False)
    repro.to_csv(OUT_DIR / "existing_output_reproducibility.csv", index=False)
    scope.to_csv(OUT_DIR / "dataset_scope.csv", index=False)

    make_plots(all_norm_df, sensitivity, uncertainty)

    summary = {
        "subjects": scope["subject"].nunique(),
        "trials": len(scope),
        "cycles": int(scope["n_cycles"].sum()),
        "conditions": sorted(metrics["condition"].unique().tolist()),
        "muscles_total": len(ALL_MUSCLES),
        "primary_muscles": len(PRIMARY_MUSCLES),
        "hamstring_muscles": len(HAMSTRING_MUSCLES),
        "mean_legacy_hip3d_nrmse_pct_rom": float(sensitivity_legacy["nrmse_pct_of_sagittal_rom"].mean()),
        "max_legacy_hip3d_nrmse_pct_rom": float(sensitivity_legacy["nrmse_pct_of_sagittal_rom"].max()),
        "mean_3d_nrmse_pct_rom": float(sensitivity["nrmse_pct_of_sagittal_rom"].mean()),
        "max_3d_nrmse_pct_rom": float(sensitivity["nrmse_pct_of_sagittal_rom"].max()),
        "max_uncertainty_rom_delta_abs_pct": float(uncertainty["rom_delta_absmax_pct"].max()),
        "max_existing_repro_rmse_m": float(repro["rmse_m"].max()) if not repro.empty else np.nan,
    }
    pd.Series(summary).to_json(OUT_DIR / "top_tier_experiment_summary.json", indent=2)
    print("\nSummary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
