{\rtf1\ansi\ansicpg1253\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .SFNS-Semibold;}
{\colortbl;\red255\green255\blue255;\red14\green14\blue14;}
{\*\expandedcolortbl;;\cssrgb\c6700\c6700\c6700;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\sl324\slmult1\pardirnatural\partightenfactor0

\f0\b\fs30 \cf2 import os\
import numpy as np\
import pandas as pd\
import matplotlib.pyplot as plt\
\
\
# -------------------------------------------------------\
# CONFIG\
# -------------------------------------------------------\
\
# Trials we want per subject\
TRIALS = \{\
    "S135": [\
        "S135_G03_D01_B01_T01",\
        "S135_G03_D01_B01_T02",\
        "S135_G03_D01_B01_T03",\
    ],\
    "S146": [\
        "S146_G03_D01_B01_T01",\
        "S146_G03_D01_B01_T02",\
        "S146_G03_D01_B01_T03",\
    ],\
\}\
\
# Muscles we keep (\uc0\u972 \u960 \u969 \u962  \u963 \u964 \u945  *_muscles.csv)\
MUSCLES = [\
    "med_gas_r_length",\
    "soleus_r_length",\
    "tib_ant_r_length",\
    "vas_lat_r_length",\
    "rect_fem_r_length",\
    "glut_med1_r_length",\
]\
\
# Output directory for plots\
OUTPUT_DIR = "all_muscle_plots"\
\
# Number of points in normalized gait cycle (0\'96100%)\
N_PHASE_POINTS = 101\
\
\
# -------------------------------------------------------\
# HELPER FUNCTIONS\
# -------------------------------------------------------\
\
def detect_heel_strikes(df):\
    """\
    Detect right heel strikes from the original IMU CSV.\
    1) If 'Contact RT' column exists (0/1), use transitions 0\uc0\u8594 1.\
    2) Else, use local minima of 'Noraxon MyoMotion-Trajectories-Heel RT-y (mm)'.\
\
    Returns:\
        indices (list of int): indices in df corresponding to heel strikes\
    """\
    if "Contact RT" in df.columns:\
        contact = df["Contact RT"].values\
        hs = []\
        for i in range(1, len(contact)):\
            if contact[i - 1] < 0.5 and contact[i] >= 0.5:\
                hs.append(i)\
        return hs\
\
    col_name = "Noraxon MyoMotion-Trajectories-Heel RT-y (mm)"\
    if col_name not in df.columns:\
        raise ValueError(\
            "Could not find 'Contact RT' or "\
            "'Noraxon MyoMotion-Trajectories-Heel RT-y (mm)' in IMU CSV."\
        )\
\
    y = df[col_name].values\
    hs = []\
    min_distance = 50  # minimum samples between heel strikes (\uc0\u957 \u945  \u956 \u951 \u957  \u946 \u961 \u943 \u963 \u954 \u949 \u953  \u948 \u953 \u960 \u955 \u940  minima)\
\
    for i in range(1, len(y) - 1):\
        if y[i] < y[i - 1] and y[i] < y[i + 1]:\
            if not hs or (i - hs[-1]) > min_distance:\
                hs.append(i)\
    return hs\
\
\
def normalize_cycles(imu_df, muscles_df, subject, trial):\
    """\
    Create normalized gait cycles (0\'96100%) for all cycles of a trial.\
\
    imu_df: original IMU dataframe (with time & contact/heel trajectory)\
    muscles_df: dataframe with columns ['time', MUSCLES...]\
    subject: 'S135' or 'S146'\
    trial: trial id string\
\
    Returns:\
        norm_df: DataFrame with columns:\
            ['subject', 'trial', 'cycle', 'phase'] + MUSCLES\
    """\
    # Make sure time alignment is consistent (assume row-wise sync)\
    if len(imu_df) != len(muscles_df):\
        raise ValueError(\
            f"IMU rows (\{len(imu_df)\}) != muscle rows (\{len(muscles_df)\}) "\
            f"for trial \{trial\}. They must be synchronized."\
        )\
\
    hs_indices = detect_heel_strikes(imu_df)\
\
    if len(hs_indices) < 2:\
        raise ValueError(\
            f"Not enough heel strikes detected in trial \{trial\} "\
            f"(found \{len(hs_indices)\})."\
        )\
\
    records = []\
\
    for c_idx in range(len(hs_indices) - 1):\
        start = hs_indices[c_idx]\
        end = hs_indices[c_idx + 1]\
\
        if end <= start + 5:\
            # \uc0\u960 \u959 \u955 \u973  \u956 \u953 \u954 \u961 \u972  \u948 \u953 \u940 \u963 \u964 \u951 \u956 \u945 , \u956 \u940 \u955 \u955 \u959 \u957  artefact\
            continue\
\
        # Original index domain\
        idx_range = np.arange(start, end + 1)\
        phase_new = np.linspace(0, 100, N_PHASE_POINTS)\
\
        for muscle in MUSCLES:\
            # Muscle length values in this cycle\
            y = muscles_df[muscle].values[idx_range]\
            x = np.linspace(0, 1, len(idx_range))  # normalized 0\'961 domain\
            x_new = np.linspace(0, 1, N_PHASE_POINTS)\
            # Interpolate\
            y_new = np.interp(x_new, x, y)\
\
            if muscle == MUSCLES[0]:\
                # create base rows once per cycle\
                for k in range(N_PHASE_POINTS):\
                    records.append(\{\
                        "subject": subject,\
                        "trial": trial,\
                        "cycle": c_idx,\
                        "phase": phase_new[k],\
                        muscle: y_new[k],\
                    \})\
            else:\
                # fill muscle column in the same rows\
                base_idx = len(records) - N_PHASE_POINTS\
                for k in range(N_PHASE_POINTS):\
                    records[base_idx + k][muscle] = y_new[k]\
\
    norm_df = pd.DataFrame.from_records(records)\
    return norm_df\
\
\
def ensure_dir(path):\
    if not os.path.exists(path):\
        os.makedirs(path)\
\
\
# -------------------------------------------------------\
# MAIN ANALYSIS\
# -------------------------------------------------------\
\
def main():\
    base_dir = os.path.dirname(os.path.abspath(__file__))\
    ensure_dir(os.path.join(base_dir, OUTPUT_DIR))\
\
    all_cycles_list = []\
\
    # 1) Build normalized cycles for all subject/trial combinations\
    for subject, trial_list in TRIALS.items():\
        for trial in trial_list:\
            imu_path = os.path.join(base_dir, subject, trial + ".csv")\
            muscles_path = os.path.join(base_dir, trial + "_muscles.csv")\
\
            if not os.path.isfile(imu_path):\
                print(f"[WARNING] Missing IMU CSV: \{imu_path\}")\
                continue\
            if not os.path.isfile(muscles_path):\
                print(f"[WARNING] Missing muscles CSV: \{muscles_path\}")\
                continue\
\
            print(f"\\n\uc0\u55357 \u56516  Processing \{subject\} - \{trial\}")\
            print(f"   IMU:     \{imu_path\}")\
            print(f"   Muscles: \{muscles_path\}")\
\
            imu_df = pd.read_csv(imu_path)\
            muscles_df = pd.read_csv(muscles_path)\
\
            # ensure only needed columns from muscles_df\
            keep_cols = ["time"] + MUSCLES\
            muscles_df = muscles_df[keep_cols]\
\
            norm_df = normalize_cycles(imu_df, muscles_df, subject, trial)\
            print(f"   \uc0\u10145  Normalized cycles shape: \{norm_df.shape\}")\
\
            all_cycles_list.append(norm_df)\
\
    if not all_cycles_list:\
        print("\uc0\u10060  No valid data found. Check paths/TRIALS.")\
        return\
\
    all_cycles = pd.concat(all_cycles_list, ignore_index=True)\
\
    # 2) Melt for easier grouping\
    long_df = all_cycles.melt(\
        id_vars=["subject", "trial", "cycle", "phase"],\
        value_vars=MUSCLES,\
        var_name="muscle",\
        value_name="length",\
    )\
\
    # 3) Compute statistics per subject\'96muscle\'96phase\
    stats = (\
        long_df\
        .groupby(["subject", "muscle", "phase"])\
        .agg(\
            mean_length=("length", "mean"),\
            sd_length=("length", "std"),\
            n_cycles=("length", "count"),\
        )\
        .reset_index()\
    )\
\
    # -------------------------------------------------------\
    # PLOTS\
    # -------------------------------------------------------\
\
    # A) Per-subject plots (mean + SD band) for each muscle separately\
    for muscle in MUSCLES:\
        for subject in TRIALS.keys():\
            sub_stats = stats[(stats["subject"] == subject) &\
                              (stats["muscle"] == muscle)]\
\
            if sub_stats.empty:\
                continue\
\
            fig, ax = plt.subplots(figsize=(6, 4))\
            phase = sub_stats["phase"].values\
            mean = sub_stats["mean_length"].values\
            sd = sub_stats["sd_length"].values\
\
            ax.plot(phase, mean, label=f"\{subject\} mean")\
            ax.fill_between(phase, mean - sd, mean + sd,\
                            alpha=0.3, label=f"\{subject\} \'b1 SD")\
\
            ax.set_title(f"\{muscle\} \'96 \{subject\}")\
            ax.set_xlabel("Gait cycle (%)")\
            ax.set_ylabel("Muscle length (norm.)")\
            ax.legend()\
            ax.grid(True, alpha=0.3)\
\
            out_name = f"\{muscle\}_\{subject\}_mean_sd.png"\
            out_path = os.path.join(base_dir, OUTPUT_DIR, out_name)\
            fig.tight_layout()\
            fig.savefig(out_path, dpi=300)\
            plt.close(fig)\
            print(f"\uc0\u9989  Saved: \{out_path\}")\
\
    # B) Subject comparison plots (S135 vs S146) per muscle\
    subjects = list(TRIALS.keys())\
    if len(subjects) >= 2:\
        subj1, subj2 = subjects[0], subjects[1]\
        for muscle in MUSCLES:\
            fig, ax = plt.subplots(figsize=(6, 4))\
\
            for subject, color, alpha_fill in [\
                (subj1, "tab:blue", 0.2),\
                (subj2, "tab:orange", 0.2),\
            ]:\
                sub_stats = stats[(stats["subject"] == subject) &\
                                  (stats["muscle"] == muscle)]\
                if sub_stats.empty:\
                    continue\
\
                phase = sub_stats["phase"].values\
                mean = sub_stats["mean_length"].values\
                sd = sub_stats["sd_length"].values\
\
                ax.plot(phase, mean, label=f"\{subject\} mean")\
                ax.fill_between(phase, mean - sd, mean + sd,\
                                alpha=alpha_fill)\
\
            ax.set_title(f"\{muscle\} \'96 \{subj1\} vs \{subj2\}")\
            ax.set_xlabel("Gait cycle (%)")\
            ax.set_ylabel("Muscle length (norm.)")\
            ax.legend()\
            ax.grid(True, alpha=0.3)\
\
            out_name = f"\{muscle\}_\{subj1\}_vs_\{subj2\}.png"\
            out_path = os.path.join(base_dir, OUTPUT_DIR, out_name)\
            fig.tight_layout()\
            fig.savefig(out_path, dpi=300)\
            plt.close(fig)\
            print(f"\uc0\u9989  Saved: \{out_path\}")\
\
    print("\\n\uc0\u55356 \u57225  Done \'96 all subject/muscle comparison plots generated.")\
\
\
if __name__ == "__main__":\
    main()}