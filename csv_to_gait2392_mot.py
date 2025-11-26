import sys
import os
import pandas as pd

# Mapping from NONAN column names -> Gait2392 coordinate names
COLUMN_MAP = {
    # LEFT leg
    "Hip Flexion LT (deg)": "hip_flexion_l",
    "Hip Abduction LT (deg)": "hip_adduction_l",
    "Hip Rotation Ext LT (deg)": "hip_rotation_l",
    "Knee Flexion LT (deg)": "knee_angle_l",
    "Ankle Dorsiflexion LT (deg)": "ankle_angle_l",

    # RIGHT leg
    "Hip Flexion RT (deg)": "hip_flexion_r",
    "Hip Abduction RT (deg)": "hip_adduction_r",
    "Hip Rotation Ext RT (deg)": "hip_rotation_r",
    "Knee Flexion RT (deg)": "knee_angle_r",
    "Ankle Dorsiflexion RT (deg)": "ankle_angle_r",
}


def write_opensim_mot(df, out_path):
    """
    Γράφει ένα OpenSim .mot file με κλασικό Storage format:
    name ...
    datacolumns N
    datarows M
    range t0 t1
    endheader
    time col2 col3 ...
    ...
    """
    n_rows, n_cols = df.shape
    t0 = float(df["time"].iloc[0])
    t1 = float(df["time"].iloc[-1])

    name = os.path.splitext(os.path.basename(out_path))[0]

    with open(out_path, "w") as f:
        f.write(f"name {name}\n")
        f.write(f"datarows {n_rows}\n")
        f.write(f"datacolumns {n_cols}\n")
        f.write(f"range {t0:.6f} {t1:.6f}\n")
        f.write("endheader\n")
        # header line with column names
        f.write("\t".join(df.columns) + "\n")
        # data rows
        for _, row in df.iterrows():
            f.write("\t".join(f"{float(val):.6f}" for val in row.values) + "\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python csv_to_gait2392_mot.py <input_csv> <output_mot>")
        sys.exit(1)

    csv_path = sys.argv[1]
    mot_path = sys.argv[2]

    if not os.path.isfile(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    print(f"📄 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        print("❌ Column 'time' not found in CSV.")
        sys.exit(1)

    # φτιάχνουμε dict με time + mapped joint angles
    data = {"time": df["time"].values}
    missing = []

    for src_col, gait_col in COLUMN_MAP.items():
        if src_col in df.columns:
            data[gait_col] = df[src_col].values
        else:
            missing.append(src_col)

    if missing:
        print("⚠ Warning: The following expected columns were NOT found in the CSV:")
        for m in missing:
            print("  -", m)
        print("Θα συνεχίσουμε μόνο με τις διαθέσιμες στήλες.\n")

    mot_df = pd.DataFrame(data)

    print("➡ Output columns in .mot:")
    print(list(mot_df.columns))
    print("➡ Shape:", mot_df.shape)

    write_opensim_mot(mot_df, mot_path)
    print(f"\n✅ Wrote OpenSim .mot file to: {mot_path}")


if __name__ == "__main__":
    main()
