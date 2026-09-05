import sys
import pandas as pd
import numpy as np

def detect_heel_strikes(contact_rt: pd.Series, time: pd.Series, thresh=0.5):
    """
    Heel strikes: μεταβάσεις 0 -> 1 στο Contact RT.
    contact_rt: 0/1 ή ποσοστό επαφής
    """
    # Αν δεν είναι ακριβώς 0/1, κάνε threshold
    c = (contact_rt > thresh).astype(int)
    hs_indices = np.where((c.shift(1, fill_value=0) == 0) & (c == 1))[0]
    hs_times = time.iloc[hs_indices].values
    return hs_indices, hs_times

def resample_cycle(df_cycle, n_points=101, time_col="time", value_cols=None):
    """Resample ενός κύκλου σε 0–100% (n_points)."""
    if value_cols is None:
        value_cols = [c for c in df_cycle.columns if c != time_col]

    t = df_cycle[time_col].values
    t_norm = (t - t[0]) / (t[-1] - t[0])  # 0–1

    phases = np.linspace(0, 1, n_points)
    out = {"phase": phases * 100}  # σε %

    for col in value_cols:
        y = df_cycle[col].values
        out[col] = np.interp(phases, t_norm, y)

    return pd.DataFrame(out)

def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python gait_cycle_muscle_lengths.py "
              "<original_csv> <muscle_lengths_csv> <output_csv>")
        sys.exit(1)

    original_csv = sys.argv[1]
    muscle_csv = sys.argv[2]
    out_csv = sys.argv[3]

    print(f"📄 Original CSV: {original_csv}")
    print(f"📄 Muscle lengths CSV: {muscle_csv}")
    print(f"📄 Output (normalized cycles): {out_csv}")

    # 1. Load data
    df_orig = pd.read_csv(original_csv)
    df_musc = pd.read_csv(muscle_csv)

    # Sanity: πρέπει να έχουν κοινό 'time'
    if "time" not in df_orig.columns or "time" not in df_musc.columns:
        raise ValueError("Both CSVs must contain a 'time' column.")

    # 2. Merge on time (inner join)
    df = pd.merge(df_orig, df_musc, on="time", how="inner")
    print(f"➡ Merged shape: {df.shape}")

    # 3. Heel strikes από Contact RT
    if "Contact RT" not in df.columns:
        raise ValueError("Column 'Contact RT' not found in original CSV.")

    hs_idx, hs_times = detect_heel_strikes(df["Contact RT"], df["time"])
    print(f"➡ Detected {len(hs_idx)} heel strikes (right).")

    if len(hs_idx) < 3:
        raise ValueError("Not enough heel strikes to form at least 2 full gait cycles.")

    # θα πάρουμε τους 2 πρώτους πλήρεις κύκλους
    n_cycles = 2
    n_cycles = min(n_cycles, len(hs_idx) - 1)

    # Μυες που μας ενδιαφέρουν (μπορείς να προσθέσεις κι άλλους)
    muscles = [
        "med_gas_r_length",
        "soleus_r_length",
        "tib_ant_r_length",
        "vas_lat_r_length",
        "rect_fem_r_length",
        "glut_med1_r_length",
    ]

    value_cols = muscles  # μόνο αυτοί στο output

    all_cycles = []

    for i in range(n_cycles):
        start = hs_idx[i]
        end = hs_idx[i + 1]
        df_cycle = df.iloc[start:end+1].copy()
        if len(df_cycle) < 10:
            continue

        df_resampled = resample_cycle(
            df_cycle[["time"] + value_cols].reset_index(drop=True),
            n_points=101,
            time_col="time",
            value_cols=value_cols,
        )
        df_resampled["cycle"] = i + 1
        all_cycles.append(df_resampled)

    if not all_cycles:
        raise ValueError("No valid cycles resampled.")

    df_out = pd.concat(all_cycles, ignore_index=True)
    # Ταξινόμηση στη λογική σειρά: cycle, phase, muscles…
    cols = ["cycle", "phase"] + value_cols
    df_out = df_out[cols]

    df_out.to_csv(out_csv, index=False)
    print(f"✅ Saved normalized muscle-length cycles to: {out_csv}")
    print(f"   Shape: {df_out.shape}")
    print("   Columns:", df_out.columns.tolist())

if __name__ == "__main__":
    main()
