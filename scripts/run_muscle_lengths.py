import sys
import os
import numpy as np
import pandas as pd
import opensim

# Mapping από NONAN στήλες -> ονόματα συντεταγμένων στο Gait2392
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

# Μύες που θα υπολογίσουμε (μπορούμε να προσθέσουμε κι άλλους αργότερα)
MUSCLES = [
    "med_gas_r",
    "soleus_r",
    "tib_ant_r",
    "vas_lat_r",
    "rect_fem_r",
    "glut_med1_r",
]


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_muscle_lengths.py <model.osim> <input_csv> <output_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    out_csv = sys.argv[3]

    if not os.path.isfile(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    if not os.path.isfile(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    print(f"📄 Model: {model_path}")
    print(f"📄 Input CSV: {csv_path}")
    print(f"📄 Output CSV: {out_csv}")

    # Διαβάζουμε το NONAN CSV
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        print("❌ Column 'time' not found in CSV.")
        sys.exit(1)

    # Ελέγχουμε ποιες στήλες γωνιών υπάρχουν
    available_map = {}
    missing = []
    for src_col, coord_name in COLUMN_MAP.items():
        if src_col in df.columns:
            available_map[src_col] = coord_name
        else:
            missing.append(src_col)

    if missing:
        print("⚠ Warning: Missing expected angle columns:")
        for m in missing:
            print("  -", m)
        print("Θα συνεχίσουμε με όσες στήλες βρέθηκαν.\n")

    # Φορτώνουμε το μοντέλο
    model = opensim.Model(model_path)
    state = model.initSystem()
    coord_set = model.getCoordinateSet()
    muscles = model.getMuscles()

    print(f"✅ Loaded model with {muscles.getSize()} muscles.")

    n_rows = len(df)
    print(f"➡ Frames: {n_rows}")

    # Προετοιμάζουμε results dict
    results = {"time": df["time"].values.copy()}
    for m_name in MUSCLES:
        results[m_name + "_length"] = np.zeros(n_rows)

    # Βρόχος στο χρόνο
    for i, row in df.iterrows():
        # Set coordinates for όσες γωνίες έχουμε
        for src_col, coord_name in available_map.items():
            angle_deg = row[src_col]
            angle_rad = np.deg2rad(angle_deg)
            coord = coord_set.get(coord_name)
            coord.setValue(state, float(angle_rad), False)

        # Οι υπόλοιπες συντεταγμένες μένουν στις default τιμές του μοντέλου
        model.realizePosition(state)

        # Υπολογισμός μυϊκού μήκους
        for m_name in MUSCLES:
            m = muscles.get(m_name)
            results[m_name + "_length"][i] = m.getLength(state)

        if i % 1000 == 0 and i > 0:
            print(f"  ... processed {i}/{n_rows} frames")

    # Αποθήκευση σε CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved muscle lengths to: {out_csv}")
    print("   Columns:", list(out_df.columns))


if __name__ == "__main__":
    main()
