from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "gaitex_extra_tasks"


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    status = pd.read_csv(OUT_DIR / "gaitex_extra_task_status.csv")
    kin = pd.read_csv(OUT_DIR / "gaitex_extra_task_kinematic_metrics.csv")
    mtu = pd.read_csv(OUT_DIR / "gaitex_extra_task_mtu_metrics.csv")

    key_kin = kin[kin["coordinate"].isin(["hip_flexion_r", "knee_angle_r", "ankle_angle_r"])]
    task_rows = []
    for task, group in mtu.groupby("task"):
        key_group = key_kin[key_kin["task"] == task]
        task_rows.append(
            {
                "task": task,
                "n_segments": int(status[(status["task"] == task) & (status["status"] == "ok")].shape[0]),
                "n_subjects": int(status[(status["task"] == task) & (status["status"] == "ok")]["subject"].nunique()),
                "median_key_joint_corr": key_group["corr"].median(),
                "median_mtu_corr": group["corr"].median(),
                "median_mtu_nrmse_pct": group["nrmse_pct_of_imu_rom"].median(),
                "median_mtu_rom_mm": group["imu_rom_m"].median() * 1000,
            }
        )
    task_summary = pd.DataFrame(task_rows).sort_values("task")
    task_summary.to_csv(OUT_DIR / "gaitex_extra_task_task_summary.csv", index=False)

    label_rows = []
    for (task, label), group in mtu.groupby(["task", "segment_label"]):
        key_group = key_kin[(key_kin["task"] == task) & (key_kin["segment_label"] == label)]
        label_rows.append(
            {
                "task": task,
                "segment_label": label,
                "n_segments": int(status[(status["task"] == task) & (status["segment_label"] == label) & (status["status"] == "ok")].shape[0]),
                "n_subjects": int(status[(status["task"] == task) & (status["segment_label"] == label) & (status["status"] == "ok")]["subject"].nunique()),
                "median_key_joint_corr": key_group["corr"].median(),
                "median_mtu_corr": group["corr"].median(),
                "median_mtu_nrmse_pct": group["nrmse_pct_of_imu_rom"].median(),
                "median_mtu_rom_mm": group["imu_rom_m"].median() * 1000,
            }
        )
    label_summary = pd.DataFrame(label_rows).sort_values(["task", "segment_label"])
    label_summary.to_csv(OUT_DIR / "gaitex_extra_task_label_summary.csv", index=False)

    primary_muscles = ["tib_ant_r", "soleus_r", "med_gas_r", "rect_fem_r", "vas_lat_r", "glut_med1_r"]
    rom = (
        mtu[mtu["muscle"].isin(primary_muscles)]
        .groupby(["task", "segment_label", "muscle_label"])["imu_rom_m"]
        .median()
        .mul(1000)
        .reset_index(name="median_imu_rom_mm")
    )
    rom.to_csv(OUT_DIR / "gaitex_extra_task_primary_muscle_rom_summary.csv", index=False)

    lines = [
        "# GAITEX Extra Task MTU Validation Summary",
        "",
        "This analysis extends the paired IMU versus optical GAITEX validation beyond natural gait.",
        "It includes natural gait (`ng`), gait with obstacle (`gwo`), and instructed movement variants (`rd`, `rgs`).",
        "",
        "## Run Summary",
        "",
        f"- Completed segments: {(status['status'] == 'ok').sum()} / {len(status)}",
        f"- Subjects represented: {status.loc[status['status'] == 'ok', 'subject'].nunique()}",
        f"- Overall median key joint correlation: {fmt(key_kin['corr'].median())}",
        f"- Overall median MTU waveform correlation: {fmt(mtu['corr'].median())}",
        f"- Overall median MTU normalized RMSE: {fmt(mtu['nrmse_pct_of_imu_rom'].median(), 1)}% of IMU derived ROM",
        "",
        "## Task Level Summary",
        "",
        "| Task | Segments | Subjects | Key joint corr | MTU corr | MTU NRMSE (%) | MTU ROM (mm) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in task_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(int(row["n_segments"])),
                    str(int(row["n_subjects"])),
                    fmt(row["median_key_joint_corr"]),
                    fmt(row["median_mtu_corr"]),
                    fmt(row["median_mtu_nrmse_pct"], 1),
                    fmt(row["median_mtu_rom_mm"], 1),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Segment Label Summary",
            "",
            "| Task | Label | Segments | Subjects | Key joint corr | MTU corr | MTU NRMSE (%) | MTU ROM (mm) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in label_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["segment_label"]),
                    str(int(row["n_segments"])),
                    str(int(row["n_subjects"])),
                    fmt(row["median_key_joint_corr"]),
                    fmt(row["median_mtu_corr"]),
                    fmt(row["median_mtu_nrmse_pct"], 1),
                    fmt(row["median_mtu_rom_mm"], 1),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The results support use of GAITEX as a controlled robustness extension rather than as a clinical pathology cohort.",
            "The additional tasks test whether IMU driven MTU trajectories remain comparable with optical motion capture during obstacle gait and instructed movement variants.",
            "The data should be described as healthy participants performing controlled task variants, not as patient validation.",
        ]
    )
    (OUT_DIR / "gaitex_extra_task_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
