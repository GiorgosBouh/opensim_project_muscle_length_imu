from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import opensim


PROJECT_DIR = Path(__file__).resolve().parents[1]
GAITEX_DIR = Path("/home/ilab/project3/Project_3.0/data")
GAIT2392 = Path("/home/ilab/opensim_project/models/gait2392_simbody.osim")
OUT_DIR = PROJECT_DIR / "gaitex_extra_tasks"
FIG_DIR = OUT_DIR / "figures"
IK_DIR = OUT_DIR / "marker_ik"
MUSCLE_MODEL_DIR = OUT_DIR / "subject_scaled_muscle_models"

N_PHASE = 101
WINDOW_SECONDS = 1.5

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GAITEX extra task MTU validation for natural gait, obstacle gait, and instructed variants."
    )
    parser.add_argument("--gaitex-dir", type=Path, default=GAITEX_DIR)
    parser.add_argument("--model", type=Path, default=GAIT2392)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tasks", nargs="+", default=["ng", "gwo", "rd", "rgs"])
    parser.add_argument("--max-subjects", type=int, default=18)
    parser.add_argument("--segments-per-label", type=int, default=1)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> tuple[Path, Path, Path]:
    fig_dir = out_dir / "figures"
    ik_dir = out_dir / "marker_ik"
    muscle_model_dir = out_dir / "subject_scaled_muscle_models"
    for path in [out_dir, fig_dir, ik_dir, muscle_model_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return fig_dir, ik_dir, muscle_model_dir


def read_mot(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if line.strip() == "endheader":
                return pd.read_csv(path, sep="\t", skiprows=idx + 1)
    raise ValueError(f"No endheader found in {path}")


def discover_subject_tasks(gaitex_dir: Path, tasks: list[str]) -> pd.DataFrame:
    rows = []
    for subject_dir in sorted(gaitex_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        for task in tasks:
            task_dir = subject_dir / task
            if not task_dir.is_dir():
                continue
            ik_root = task_dir / "ik_imus"
            model = ik_root / "models" / f"scaled_model_{subject}_{task}.osim"
            trc = ik_root / f"marker_data_osim_format_{subject}_{task}.trc"
            imu_ik = ik_root / "results_imu_ik" / f"ik_segment_registered_imu_data_{subject}_{task}.mot"
            timestamps = task_dir / f"timestamps_{subject}_{task}.csv"
            rows.append(
                {
                    "subject": subject,
                    "task": task,
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


def selected_segments(row: pd.Series, segments_per_label: int, window_seconds: float) -> list[dict]:
    timestamps = pd.read_csv(row["timestamps"])
    label_col = "label"
    start_col = "temporal_segment_start_[s]"
    end_col = "temporal_segment_end_[s]"
    velocity_col = "velocities_[km_h]" if "velocities_[km_h]" in timestamps.columns else None
    segments = []
    for label, group in timestamps.groupby(label_col, sort=False):
        for _, segment in group.head(segments_per_label).iterrows():
            start = float(segment[start_col])
            raw_end = float(segment[end_col])
            end = min(raw_end, start + window_seconds)
            if end <= start:
                continue
            segments.append(
                {
                    "segment_label": str(label),
                    "start_s": start,
                    "end_s": end,
                    "velocity_kmh": float(segment[velocity_col]) if velocity_col else np.nan,
                }
            )
    return segments


def build_subject_scaled_muscle_model(row: pd.Series, generic_model: Path, model_dir: Path) -> Path:
    subject = row["subject"]
    task = row["task"]
    out = model_dir / f"{subject}_{task}_subject_scaled_gait2392_muscles.osim"
    if out.exists():
        return out

    opensim.Logger.setLevelString("error")
    source_xml = Path(row["model"]).read_text()
    scaled = opensim.Model(str(generic_model))
    state = scaled.initSystem()
    scale_set = opensim.ScaleSet()
    target_bodies = [
        "pelvis",
        "femur_r",
        "femur_l",
        "tibia_r",
        "tibia_l",
        "talus_r",
        "talus_l",
        "calcn_r",
        "calcn_l",
        "toes_r",
        "toes_l",
    ]
    factors = {}
    for body_name in target_bodies:
        match = re.search(r'<Body name="' + re.escape(body_name) + r'">(.*?)</Body>', source_xml, re.S)
        if not match:
            continue
        mesh = re.search(r"<Mesh name=.*?</Mesh>", match.group(1), re.S)
        value = re.search(r"<scale_factors>([^<]+)</scale_factors>", mesh.group(0)) if mesh else None
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
        raise RuntimeError(f"OpenSim scaling failed for {subject} {task}")
    scaled.finalizeConnections()
    if scaled.getMuscles().getSize() != 92:
        raise RuntimeError(f"Expected 92 Gait2392 muscles, got {scaled.getMuscles().getSize()}")
    scaled.printToXML(str(out))
    return out


def run_marker_ik(row: pd.Series, segment: dict, ik_dir: Path) -> Path:
    opensim.Logger.setLevelString("error")
    subject = row["subject"]
    task_name = row["task"]
    safe_label = str(segment["segment_label"]).replace("/", "_")
    start = float(segment["start_s"])
    end = float(segment["end_s"])
    out = ik_dir / f"{subject}_{task_name}_{safe_label}_marker_ik_{start:.2f}_{end:.2f}.mot"
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
    tool.set_results_directory(str(ik_dir))
    tool.set_time_range(0, start)
    tool.set_time_range(1, end)
    tool.set_IKTaskSet(tasks)
    tool.set_accuracy(1e-4)
    tool.set_report_errors(True)
    tool.run()
    return out


def align_streams(imu: pd.DataFrame, omc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.DataFrame({"time": omc["time"].to_numpy(dtype=float)})
    for col in omc.columns:
        if col == "time" or col not in imu.columns:
            continue
        aligned[col] = np.interp(aligned["time"], imu["time"], imu[col])
    return aligned, omc.reset_index(drop=True)


def resample_motion(df: pd.DataFrame) -> pd.DataFrame:
    phase = np.linspace(0, 100, N_PHASE)
    out = pd.DataFrame({"phase": phase})
    x_old = np.linspace(0, 100, len(df))
    for col in df.columns:
        if col == "time":
            continue
        out[col] = np.interp(phase, x_old, df[col].to_numpy(dtype=float))
    return out


def waveform_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rom = float(np.max(a) - np.min(a))
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
    return {
        "rmse": rmse,
        "nrmse_pct_of_imu_rom": float(rmse / rom * 100) if rom else np.nan,
        "corr": corr,
        "imu_rom": rom,
        "omc_rom": float(np.max(b) - np.min(b)),
        "mean_bias": float(np.mean(b - a)),
    }


def gait2392_lengths(motion: pd.DataFrame, model_path: Path) -> pd.DataFrame:
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


def kinematic_metrics(row: pd.Series, segment: dict, imu: pd.DataFrame, omc: pd.DataFrame) -> list[dict]:
    rows = []
    for col in KINEMATIC_COLUMNS:
        if col not in imu.columns or col not in omc.columns:
            continue
        metric = waveform_metrics(imu[col], omc[col])
        rows.append(
            {
                "subject": row["subject"],
                "task": row["task"],
                **segment,
                "coordinate": col,
                "rmse_deg": metric["rmse"],
                "nrmse_pct_of_imu_rom": metric["nrmse_pct_of_imu_rom"],
                "corr": metric["corr"],
                "imu_rom_deg": metric["imu_rom"],
                "omc_rom_deg": metric["omc_rom"],
                "mean_bias_deg": metric["mean_bias"],
            }
        )
    return rows


def mtu_metrics(row: pd.Series, segment: dict, imu: pd.DataFrame, omc: pd.DataFrame, model_path: Path) -> tuple[list[dict], list[dict]]:
    imu_len = gait2392_lengths(resample_motion(imu), model_path)
    omc_len = gait2392_lengths(resample_motion(omc), model_path)
    rows = []
    waveform_rows = []
    for muscle in MUSCLES:
        metric = waveform_metrics(imu_len[muscle], omc_len[muscle])
        base = {
            "subject": row["subject"],
            "task": row["task"],
            **segment,
            "muscle": muscle,
            "muscle_label": MUSCLE_LABELS[muscle],
        }
        rows.append(
            {
                **base,
                "rmse_m": metric["rmse"],
                "nrmse_pct_of_imu_rom": metric["nrmse_pct_of_imu_rom"],
                "corr": metric["corr"],
                "imu_rom_m": metric["imu_rom"],
                "omc_rom_m": metric["omc_rom"],
                "mean_bias_m": metric["mean_bias"],
                "peak_phase_imu": float(imu_len.loc[imu_len[muscle].idxmax(), "phase"]),
                "peak_phase_omc": float(omc_len.loc[omc_len[muscle].idxmax(), "phase"]),
            }
        )
        for phase, imu_value, omc_value in zip(imu_len["phase"], imu_len[muscle], omc_len[muscle]):
            waveform_rows.append(
                {
                    **base,
                    "phase": float(phase),
                    "imu_length_m": float(imu_value),
                    "omc_length_m": float(omc_value),
                }
            )
    return rows, waveform_rows


def task_condition_summary(mtu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if mtu.empty:
        return pd.DataFrame()
    for (task_name, label, muscle, muscle_label), group in mtu.groupby(["task", "segment_label", "muscle", "muscle_label"]):
        rows.append(
            {
                "task": task_name,
                "segment_label": label,
                "muscle": muscle,
                "muscle_label": muscle_label,
                "n_segments": int(group.shape[0]),
                "median_imu_rom_mm": float(group["imu_rom_m"].median() * 1000),
                "median_omc_rom_mm": float(group["omc_rom_m"].median() * 1000),
                "median_mtu_corr": float(group["corr"].median()),
                "median_mtu_nrmse_pct": float(group["nrmse_pct_of_imu_rom"].median()),
                "median_peak_phase_difference_pct_gait": float(
                    (group["peak_phase_imu"] - group["peak_phase_omc"]).abs().median()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_plots(mtu: pd.DataFrame, kin: pd.DataFrame, fig_dir: Path) -> None:
    if not mtu.empty:
        summary = mtu.groupby(["task", "segment_label"])["corr"].median().reset_index()
        summary["label"] = summary["task"] + ": " + summary["segment_label"]
        plt.figure(figsize=(10, 4.8))
        plt.bar(summary["label"], summary["corr"], color="#3A6EA5")
        plt.ylim(-1, 1)
        plt.ylabel("Median IMU vs optical MTU correlation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "gaitex_extra_task_mtu_correlation_by_label.png", dpi=300)
        plt.close()

        rom = (
            mtu.groupby(["task", "segment_label"])["imu_rom_m"]
            .median()
            .mul(1000)
            .reset_index(name="median_imu_rom_mm")
        )
        rom["label"] = rom["task"] + ": " + rom["segment_label"]
        plt.figure(figsize=(10, 4.8))
        plt.bar(rom["label"], rom["median_imu_rom_mm"], color="#7A9E3F")
        plt.ylabel("Median IMU derived MTU ROM (mm)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "gaitex_extra_task_mtu_rom_by_label.png", dpi=300)
        plt.close()

    if not kin.empty:
        key = kin[kin["coordinate"].isin(["hip_flexion_r", "knee_angle_r", "ankle_angle_r"])]
        if not key.empty:
            key_summary = key.groupby(["task", "segment_label"])["corr"].median().reset_index()
            key_summary["label"] = key_summary["task"] + ": " + key_summary["segment_label"]
            plt.figure(figsize=(10, 4.8))
            plt.bar(key_summary["label"], key_summary["corr"], color="#C45A4D")
            plt.ylim(-1, 1)
            plt.ylabel("Median key joint correlation")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "gaitex_extra_task_key_joint_correlation_by_label.png", dpi=300)
            plt.close()


def main() -> None:
    args = parse_args()
    fig_dir, ik_dir, muscle_model_dir = ensure_dirs(args.out_dir)
    opensim.Logger.setLevelString("error")

    inventory = discover_subject_tasks(args.gaitex_dir, args.tasks)
    inventory.to_csv(args.out_dir / "gaitex_extra_task_inventory.csv", index=False)
    complete = inventory[inventory["has_model"] & inventory["has_trc"] & inventory["has_imu_ik"] & inventory["has_timestamps"]]
    subjects = complete["subject"].drop_duplicates().head(args.max_subjects).tolist()
    complete = complete[complete["subject"].isin(subjects)].copy()

    status_rows = []
    kin_rows = []
    mtu_rows = []
    waveform_rows = []
    for _, row in complete.iterrows():
        try:
            muscle_model = build_subject_scaled_muscle_model(row, args.model, muscle_model_dir)
        except Exception as exc:
            status_rows.append(
                {
                    "subject": row["subject"],
                    "task": row["task"],
                    "segment_label": "",
                    "status": "failed_model_scaling",
                    "error": repr(exc),
                }
            )
            continue
        for segment in selected_segments(row, args.segments_per_label, args.window_seconds):
            print(f"{row['subject']} {row['task']} {segment['segment_label']}: {segment['start_s']:.2f}-{segment['end_s']:.2f}s")
            status = {
                "subject": row["subject"],
                "task": row["task"],
                **segment,
                "status": "started",
            }
            try:
                marker_ik = run_marker_ik(row, segment, ik_dir)
                imu = read_mot(Path(row["imu_ik"]))
                omc = read_mot(marker_ik)
                imu = imu[(imu["time"] >= segment["start_s"]) & (imu["time"] <= segment["end_s"])].reset_index(drop=True)
                imu_aligned, omc_aligned = align_streams(imu, omc)
                kin_rows.extend(kinematic_metrics(row, segment, imu_aligned, omc_aligned))
                m_rows, w_rows = mtu_metrics(row, segment, imu_aligned, omc_aligned, muscle_model)
                mtu_rows.extend(m_rows)
                waveform_rows.extend(w_rows)
                status.update({"status": "ok", "marker_ik": str(marker_ik), "n_frames": int(len(omc_aligned))})
            except Exception as exc:
                status.update({"status": "failed", "error": repr(exc)})
            status_rows.append(status)

    status_df = pd.DataFrame(status_rows)
    kin_df = pd.DataFrame(kin_rows)
    mtu_df = pd.DataFrame(mtu_rows)
    waveform_df = pd.DataFrame(waveform_rows)
    condition_df = task_condition_summary(mtu_df)

    status_df.to_csv(args.out_dir / "gaitex_extra_task_status.csv", index=False)
    kin_df.to_csv(args.out_dir / "gaitex_extra_task_kinematic_metrics.csv", index=False)
    mtu_df.to_csv(args.out_dir / "gaitex_extra_task_mtu_metrics.csv", index=False)
    waveform_df.to_csv(args.out_dir / "gaitex_extra_task_mtu_waveforms.csv", index=False)
    condition_df.to_csv(args.out_dir / "gaitex_extra_task_condition_summary.csv", index=False)
    write_plots(mtu_df, kin_df, fig_dir)

    summary = {
        "analysis": "GAITEX extra task MTU validation",
        "gaitex_dir": str(args.gaitex_dir),
        "tasks_requested": args.tasks,
        "subjects_attempted": int(len(subjects)),
        "segments_per_label": int(args.segments_per_label),
        "window_seconds": float(args.window_seconds),
        "complete_subject_task_rows": int(complete.shape[0]),
        "segments_attempted": int(status_df.shape[0]),
        "segments_completed": int((status_df["status"] == "ok").sum()) if not status_df.empty else 0,
        "tasks_completed": sorted(status_df.loc[status_df["status"] == "ok", "task"].dropna().unique().tolist())
        if not status_df.empty
        else [],
        "median_key_joint_corr": float(
            kin_df.loc[kin_df["coordinate"].isin(["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]), "corr"].median()
        )
        if not kin_df.empty
        else np.nan,
        "median_mtu_corr": float(mtu_df["corr"].median()) if not mtu_df.empty else np.nan,
        "median_mtu_nrmse_pct_of_imu_rom": float(mtu_df["nrmse_pct_of_imu_rom"].median()) if not mtu_df.empty else np.nan,
        "outputs": {
            "status": str(args.out_dir / "gaitex_extra_task_status.csv"),
            "kinematic_metrics": str(args.out_dir / "gaitex_extra_task_kinematic_metrics.csv"),
            "mtu_metrics": str(args.out_dir / "gaitex_extra_task_mtu_metrics.csv"),
            "condition_summary": str(args.out_dir / "gaitex_extra_task_condition_summary.csv"),
            "waveforms": str(args.out_dir / "gaitex_extra_task_mtu_waveforms.csv"),
        },
    }
    (args.out_dir / "gaitex_extra_task_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
