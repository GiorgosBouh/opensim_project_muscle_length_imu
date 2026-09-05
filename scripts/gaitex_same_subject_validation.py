from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opensim


BASE_DIR = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
GAITEX_DIR = Path(os.environ.get("GAITEX_DATA_DIR", BASE_DIR / "data" / "gaitex")).expanduser()
GAIT2392 = Path(os.environ.get("GAIT2392_MODEL", BASE_DIR / "models" / "gait2392_simbody.osim")).expanduser()
OUT_DIR = BASE_DIR / "gaitex_validation"
FIG_DIR = OUT_DIR / "figures"
IK_DIR = OUT_DIR / "marker_ik"
MUSCLE_MODEL_DIR = OUT_DIR / "subject_scaled_muscle_models"

N_PHASE = 101
WINDOW_SECONDS = 1.5
MAX_SUBJECTS = 18
MAX_WINDOWS_PER_SUBJECT = 3

MUSCLES = [
    "glut_med1_r",
    "rect_fem_r",
    "vas_lat_r",
    "tib_ant_r",
    "soleus_r",
    "med_gas_r",
    "semimem_r",
    "semiten_r",
    "bifemlh_r",
    "bifemsh_r",
]

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

LOWER_BODY_MARKERS = [
    "R_IAS",
    "L_IAS",
    "R_IPS",
    "L_IPS",
    "R_FLE",
    "R_FME",
    "R_FAX",
    "R_FAL",
    "R_FCC",
    "R_TTC",
    "R_TAM",
    "R_FOOT1",
    "R_FOOT2",
    "L_FLE",
    "L_FME",
    "L_FAX",
    "L_FAL",
    "L_FCC",
    "L_TTC",
    "L_TAM",
    "L_FOOT1",
    "L_FOOT2",
    "SGL",
    "SJN",
    "MAI",
]

KINEMATIC_COLUMNS = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
]


def ensure_dirs():
    for path in [OUT_DIR, FIG_DIR, IK_DIR, MUSCLE_MODEL_DIR]:
        path.mkdir(exist_ok=True)


def build_subject_scaled_muscle_model(row):
    """Apply GAITEX subject segment scales to the muscle-containing Gait2392 model."""
    subject = row["subject"]
    MUSCLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MUSCLE_MODEL_DIR / f"{subject}_ng_subject_scaled_gait2392_muscles.osim"
    if out.exists():
        return out

    opensim.Logger.setLevelString("error")
    source_xml = Path(row["model"]).read_text()
    scaled = opensim.Model(str(GAIT2392))
    state = scaled.initSystem()
    scale_set = opensim.ScaleSet()
    target_bodies = [
        "pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l",
        "talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l",
    ]
    factors = {}
    for body_name in target_bodies:
        match = re.search(r'<Body name="' + re.escape(body_name) + r'">(.*?)</Body>', source_xml, re.S)
        if not match:
            continue
        mesh = re.search(r'<Mesh name=.*?</Mesh>', match.group(1), re.S)
        value = re.search(r'<scale_factors>([^<]+)</scale_factors>', mesh.group(0)) if mesh else None
        if not value:
            continue
        factor = float(value.group(1).split()[0])
        factors[body_name] = factor
        scale = opensim.Scale()
        scale.setSegmentName(body_name)
        scale.setScaleFactors(opensim.Vec3(factor, factor, factor))
        scale_set.cloneAndAppend(scale)
    if not factors:
        raise RuntimeError(f"No GAITEX body scale factors found in {row['model']}")
    if not scaled.scale(state, scale_set, False, 75.1646):
        raise RuntimeError(f"OpenSim scaling failed for {subject}")
    scaled.finalizeConnections()
    if scaled.getMuscles().getSize() != 92:
        raise RuntimeError(f"Expected 92 Gait2392 muscles, got {scaled.getMuscles().getSize()}")
    scaled.printToXML(str(out))
    return out


def read_mot(path):
    with open(path) as handle:
        for idx, line in enumerate(handle):
            if line.strip() == "endheader":
                return pd.read_csv(path, sep="\t", skiprows=idx + 1)
    raise ValueError(f"No endheader found in {path}")


def discover_ng_subjects():
    rows = []
    for subject_dir in sorted(GAITEX_DIR.iterdir()):
        trial_dir = subject_dir / "ng"
        if not trial_dir.is_dir():
            continue
        subject = subject_dir.name
        ik_root = trial_dir / "ik_imus"
        model = ik_root / "models" / f"scaled_model_{subject}_ng.osim"
        trc = ik_root / f"marker_data_osim_format_{subject}_ng.trc"
        imu_ik = ik_root / "results_imu_ik" / f"ik_segment_registered_imu_data_{subject}_ng.mot"
        timestamps = trial_dir / f"timestamps_{subject}_ng.csv"
        rows.append(
            {
                "subject": subject,
                "task": "ng",
                "model": str(model),
                "trc": str(trc),
                "imu_ik": str(imu_ik),
                "timestamps": str(timestamps),
                "has_model": model.exists(),
                "has_trc": trc.exists(),
                "has_imu_ik": imu_ik.exists(),
                "has_timestamps": timestamps.exists(),
            }
        )
    return pd.DataFrame(rows)


def subject_windows(row):
    timestamps = pd.read_csv(row["timestamps"])
    windows = []
    for _, segment in timestamps.head(MAX_WINDOWS_PER_SUBJECT).iterrows():
        start = float(segment["temporal_segment_start_[s]"])
        segment_end = float(segment["temporal_segment_end_[s]"])
        end = min(segment_end, start + WINDOW_SECONDS)
        if end > start:
            windows.append((str(segment["label"]), start, end, float(segment["velocities_[km_h]"])))
    return windows


def run_marker_ik(row, segment_label, start, end):
    opensim.Logger.setLevelString("error")
    subject = row["subject"]
    safe_label = segment_label.replace("/", "_")
    out = IK_DIR / f"{subject}_ng_{safe_label}_marker_ik_{start:.2f}_{end:.2f}.mot"
    if out.exists():
        return out

    model = opensim.Model(row["model"])
    tasks = opensim.IKTaskSet()
    for marker_name in LOWER_BODY_MARKERS:
        if model.getMarkerSet().contains(marker_name):
            task = opensim.IKMarkerTask()
            task.setName(marker_name)
            task.setApply(True)
            task.setWeight(1.0)
            tasks.adoptAndAppend(task)

    tool = opensim.InverseKinematicsTool()
    tool.set_model_file(row["model"])
    tool.set_marker_file(row["trc"])
    tool.set_coordinate_file(row["imu_ik"])
    tool.set_output_motion_file(str(out))
    tool.set_results_directory(str(IK_DIR))
    tool.set_time_range(0, start)
    tool.set_time_range(1, end)
    tool.set_IKTaskSet(tasks)
    tool.set_accuracy(1e-4)
    tool.set_report_errors(True)
    tool.run()
    return out


def align_streams(imu, omc):
    aligned = pd.DataFrame({"time": omc["time"].to_numpy(dtype=float)})
    for col in omc.columns:
        if col == "time" or col not in imu.columns:
            continue
        aligned[col] = np.interp(aligned["time"], imu["time"], imu[col])
    return aligned, omc.reset_index(drop=True)


def waveform_metrics(a, b, scale):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rom = float(np.max(a) - np.min(a))
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
    return {
        f"rmse_{scale}": rmse,
        f"nrmse_pct_of_imu_rom": float(rmse / rom * 100) if rom else np.nan,
        "corr": corr,
        "imu_rom": rom,
        "omc_rom": float(np.max(b) - np.min(b)),
        "mean_bias": float(np.mean(b - a)),
    }


def kinematic_agreement(subject, segment_label, velocity_kmh, imu, omc):
    rows = []
    for col in KINEMATIC_COLUMNS:
        if col not in imu.columns or col not in omc.columns:
            continue
        metric = waveform_metrics(imu[col], omc[col], "deg")
        rows.append(
            {
                "subject": subject,
                "segment_label": segment_label,
                "velocity_kmh": velocity_kmh,
                "coordinate": col,
                **metric,
            }
        )
    return rows


def resample_motion(df):
    phase = np.linspace(0, 100, N_PHASE)
    out = pd.DataFrame({"phase": phase})
    x_old = np.linspace(0, 100, len(df))
    for col in df.columns:
        if col == "time":
            continue
        out[col] = np.interp(phase, x_old, df[col].to_numpy(dtype=float))
    return out


def gait2392_lengths(motion, model_path=GAIT2392):
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    coords = model.getCoordinateSet()
    muscles = model.getMuscles()
    available_muscles = {muscles.get(i).getName() for i in range(muscles.getSize())}
    missing = sorted(set(MUSCLES) - available_muscles)
    if missing:
        raise RuntimeError(f"Muscles missing from {model_path}: {missing}")
    model_coord_names = [coords.get(i).getName() for i in range(coords.getSize())]
    usable = [name for name in model_coord_names if name in motion.columns]
    data = {"phase": motion["phase"].to_numpy(dtype=float)}
    for muscle in MUSCLES:
        data[muscle] = []

    for _, row in motion.iterrows():
        state.setTime(float(row["phase"]) / 100.0)
        for coord_name in usable:
            value = float(row[coord_name])
            if coord_name in {"pelvis_tx", "pelvis_ty", "pelvis_tz"}:
                coords.get(coord_name).setValue(state, value, False)
            else:
                coords.get(coord_name).setValue(state, np.deg2rad(value), False)
        model.realizePosition(state)
        for muscle in MUSCLES:
            data[muscle].append(float(muscles.get(muscle).getLength(state)))
    return pd.DataFrame(data)


def mtu_agreement(subject, segment_label, velocity_kmh, imu, omc, model_path=GAIT2392):
    imu_phase = resample_motion(imu)
    omc_phase = resample_motion(omc)
    imu_len = gait2392_lengths(imu_phase, model_path)
    omc_len = gait2392_lengths(omc_phase, model_path)
    rows = []
    wave_rows = []
    for muscle in MUSCLES:
        metric = waveform_metrics(imu_len[muscle], omc_len[muscle], "m")
        rows.append(
            {
                "subject": subject,
                "segment_label": segment_label,
                "velocity_kmh": velocity_kmh,
                "muscle": muscle,
                "muscle_label": MUSCLE_LABELS[muscle],
                **metric,
            }
        )
        for phase, a, b in zip(imu_len["phase"], imu_len[muscle], omc_len[muscle]):
            wave_rows.append(
                {
                    "subject": subject,
                    "segment_label": segment_label,
                    "velocity_kmh": velocity_kmh,
                    "phase": float(phase),
                    "muscle": muscle,
                    "muscle_label": MUSCLE_LABELS[muscle],
                    "imu_length_m": float(a),
                    "omc_length_m": float(b),
                }
            )
    return rows, wave_rows


def make_plots(kinematics, mtu):
    kin = kinematics[kinematics["coordinate"].isin(["hip_flexion_r", "knee_angle_r", "ankle_angle_r"])]
    kin_summary = kin.groupby("coordinate")["corr"].median().reset_index()
    plt.figure(figsize=(6, 3.5))
    plt.bar(kin_summary["coordinate"], kin_summary["corr"], color="#4C78A8")
    plt.ylim(-1, 1)
    plt.ylabel("Median IMU-vs-OMC correlation")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gaitex_kinematic_correlations.png", dpi=300)
    plt.close()

    mtu_summary = mtu.groupby(["muscle", "muscle_label"])["corr"].median().reset_index()
    mtu_summary = mtu_summary.sort_values("corr")
    plt.figure(figsize=(9, 4))
    plt.bar(mtu_summary["muscle_label"], mtu_summary["corr"], color="#E45756")
    plt.ylim(-1, 1)
    plt.ylabel("Median IMU-vs-OMC MTU correlation")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gaitex_mtu_correlations.png", dpi=300)
    plt.close()

    nrmse_summary = mtu.groupby(["muscle", "muscle_label"])["nrmse_pct_of_imu_rom"].median().reset_index()
    nrmse_summary = nrmse_summary.sort_values("nrmse_pct_of_imu_rom", ascending=False)
    plt.figure(figsize=(9, 4))
    plt.bar(nrmse_summary["muscle_label"], nrmse_summary["nrmse_pct_of_imu_rom"], color="#2A9D8F")
    plt.ylabel("Median NRMSE (% IMU ROM)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gaitex_mtu_nrmse.png", dpi=300)
    plt.close()


def main():
    ensure_dirs()
    inventory = discover_ng_subjects()
    inventory.to_csv(OUT_DIR / "gaitex_subject_task_inventory.csv", index=False)
    candidates = inventory[
        inventory["has_model"] & inventory["has_trc"] & inventory["has_imu_ik"] & inventory["has_timestamps"]
    ].head(MAX_SUBJECTS)

    status_rows = []
    kin_rows = []
    mtu_rows = []
    wave_rows = []
    for _, row in candidates.iterrows():
        subject = row["subject"]
        subject_muscle_model = build_subject_scaled_muscle_model(row)
        for segment_label, start, end, velocity_kmh in subject_windows(row):
            print(f"{subject} {segment_label}: marker IK {start:.2f}-{end:.2f} s")
            status = {
                "subject": subject,
                "segment_label": segment_label,
                "velocity_kmh": velocity_kmh,
                "start_s": start,
                "end_s": end,
                "status": "started",
            }
            try:
                marker_ik = run_marker_ik(row, segment_label, start, end)
                imu = read_mot(row["imu_ik"])
                omc = read_mot(marker_ik)
                imu = imu[(imu["time"] >= start) & (imu["time"] <= end)].reset_index(drop=True)
                imu_aligned, omc_aligned = align_streams(imu, omc)
                kin_rows.extend(kinematic_agreement(subject, segment_label, velocity_kmh, imu_aligned, omc_aligned))
                m_rows, w_rows = mtu_agreement(
                    subject, segment_label, velocity_kmh, imu_aligned, omc_aligned, subject_muscle_model
                )
                mtu_rows.extend(m_rows)
                wave_rows.extend(w_rows)
                status.update({"status": "ok", "marker_ik": str(marker_ik), "n_frames": int(len(omc_aligned))})
            except Exception as exc:
                status.update({"status": "failed", "error": repr(exc)})
            status_rows.append(status)

    status_df = pd.DataFrame(status_rows)
    kin_df = pd.DataFrame(kin_rows)
    mtu_df = pd.DataFrame(mtu_rows)
    wave_df = pd.DataFrame(wave_rows)

    qc_df = pd.DataFrame()
    if not kin_df.empty:
        key_coords = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
        key = kin_df[kin_df["coordinate"].isin(key_coords)]
        wide = key.pivot_table(
            index=["subject", "segment_label", "velocity_kmh"],
            columns="coordinate",
            values="rmse_deg",
        ).reset_index()
        qc_rows = []
        for threshold in [10, 15, 20, 30, 50]:
            available = [coord for coord in key_coords if coord in wide.columns]
            pass_mask = (wide[available] <= threshold).all(axis=1) if available else pd.Series([], dtype=bool)
            qc_rows.append(
                {
                    "criterion": f"all_right_hip_knee_ankle_rmse_le_{threshold}_deg",
                    "threshold_deg": threshold,
                    "n_pass_windows": int(pass_mask.sum()),
                    "n_total_windows": int(len(wide)),
                    "pass_pct": float(pass_mask.mean() * 100) if len(wide) else np.nan,
                }
            )
        qc_df = pd.DataFrame(qc_rows)
    status_df.to_csv(OUT_DIR / "gaitex_marker_ik_status.csv", index=False)
    kin_df.to_csv(OUT_DIR / "gaitex_same_subject_kinematic_metrics.csv", index=False)
    mtu_df.to_csv(OUT_DIR / "gaitex_same_subject_mtu_metrics.csv", index=False)
    wave_df.to_csv(OUT_DIR / "gaitex_same_subject_mtu_waveforms.csv", index=False)
    qc_df.to_csv(OUT_DIR / "gaitex_key_joint_qc_summary.csv", index=False)
    if not kin_df.empty and not mtu_df.empty:
        make_plots(kin_df, mtu_df)

    summary = {
        "dataset": "GAITEX local mirror",
        "source_path": str(GAITEX_DIR),
        "ng_subjects_discovered": int(inventory.shape[0]),
        "subjects_attempted": int(candidates.shape[0]),
        "windows_attempted": int(status_df.shape[0]),
        "windows_completed": int((status_df["status"] == "ok").sum()) if not status_df.empty else 0,
        "subjects_completed": int(status_df.loc[status_df["status"] == "ok", "subject"].nunique()) if not status_df.empty else 0,
        "window_seconds_per_subject": WINDOW_SECONDS,
        "max_windows_per_subject": MAX_WINDOWS_PER_SUBJECT,
        "gaitex_original_models_have_muscles": False,
        "subject_scaled_muscle_models_have_muscles": True,
        "subject_scaled_muscle_model_method": "GAITEX scaled skeleton plus Gait2392 muscle actuators",
        "subject_scaled_muscle_models": int(len(list(MUSCLE_MODEL_DIR.glob("*.osim")))),
        "downstream_mtu_model": str(GAIT2392),
        "median_right_knee_angle_corr": float(
            kin_df.loc[kin_df["coordinate"] == "knee_angle_r", "corr"].median()
        )
        if not kin_df.empty
        else np.nan,
        "median_right_ankle_angle_corr": float(
            kin_df.loc[kin_df["coordinate"] == "ankle_angle_r", "corr"].median()
        )
        if not kin_df.empty
        else np.nan,
        "median_mtu_corr": float(mtu_df["corr"].median()) if not mtu_df.empty else np.nan,
        "median_mtu_nrmse_pct_imu_rom": float(mtu_df["nrmse_pct_of_imu_rom"].median())
        if not mtu_df.empty
        else np.nan,
        "key_joint_windows_all_rmse_le_10_deg": int(qc_df.loc[qc_df["threshold_deg"] == 10, "n_pass_windows"].iloc[0])
        if not qc_df.empty
        else 0,
        "key_joint_windows_all_rmse_le_20_deg": int(qc_df.loc[qc_df["threshold_deg"] == 20, "n_pass_windows"].iloc[0])
        if not qc_df.empty
        else 0,
        "key_joint_windows_all_rmse_le_50_deg": int(qc_df.loc[qc_df["threshold_deg"] == 50, "n_pass_windows"].iloc[0])
        if not qc_df.empty
        else 0,
    }
    (OUT_DIR / "gaitex_same_subject_validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
