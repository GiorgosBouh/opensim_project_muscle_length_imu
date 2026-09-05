import pandas as pd
import numpy as np
import glob

# αν ποτέ έχεις ; αντί για , βάζεις: READ_KWARGS = {"sep": ";"}
READ_KWARGS = {}

# χάρτης: όνομα στήλης -> ωραίο όνομα για πίνακα
MUSCLES = {
    "glut_med1_r_length": "Glut. medius",
    "rect_fem_r_length": "Rect. femoris",
    "soleus_r_length": "Soleus",
    "med_gas_r_length": "Med. gastrocnemius",
    "tib_ant_r_length": "Tib. anterior",
    "vas_lat_r_length": "Vast. lateralis",
}

# χρησιμοποιούμε τα 3 trials D01_B01_T01–03 όπως στο paper
subjects = {
    "S135": sorted(glob.glob("S135_G03_D01_B01_T0*_muscles.csv")),
    "S146": sorted(glob.glob("S146_G03_D01_B01_T0*_muscles.csv")),
}

rows = []

for subj, files in subjects.items():
    print(f"\nSubject {subj}, files: {files}")
    for col, nice_name in MUSCLES.items():
        peaks = []
        mins = []
        roms = []
        for f in files:
            df = pd.read_csv(f, **READ_KWARGS)
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in {f}. Columns: {df.columns}")
            x = df[col].to_numpy()
            peak = float(np.max(x))
            minval = float(np.min(x))
            rom = peak - minval
            peaks.append(peak)
            mins.append(minval)
            roms.append(rom)

        # mean, sd, cv
        peak_mean = float(np.mean(peaks))
        peak_sd   = float(np.std(peaks, ddof=1))
        peak_cv   = (peak_sd / peak_mean * 100) if peak_mean != 0 else np.nan

        min_mean = float(np.mean(mins))
        min_sd   = float(np.std(mins, ddof=1))
        min_cv   = (min_sd / min_mean * 100) if min_mean != 0 else np.nan

        rom_mean = float(np.mean(roms))
        rom_sd   = float(np.std(roms, ddof=1))
        rom_cv   = (rom_sd / rom_mean * 100) if rom_mean != 0 else np.nan

        rows.append({
            "Muscle": nice_name,
            "Subject": subj,
            "Peak_mean": peak_mean,
            "Peak_SD": peak_sd,
            "Peak_CV_%": peak_cv,
            "Min_mean": min_mean,
            "Min_SD": min_sd,
            "Min_CV_%": min_cv,
            "ROM_mean": rom_mean,
            "ROM_SD": rom_sd,
            "ROM_CV_%": rom_cv,
        })

out = pd.DataFrame(rows)
out.to_csv("mtu_repeatability_summary.csv", index=False)
print("\nSaved mtu_repeatability_summary.csv")
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
