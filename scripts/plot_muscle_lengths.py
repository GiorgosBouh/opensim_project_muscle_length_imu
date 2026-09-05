#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_muscle_lengths.py <normcycles_csv> <out_dir>")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    print(f"📄 Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Περιμένουμε στήλες: cycle, phase, και _length για τους μύες
    base_cols = {"cycle", "phase", "time", "gait_pct", "gait_cycle_pct"}
    muscle_cols = [c for c in df.columns if c.endswith("_length") and c not in base_cols]

    print("➡ Muscles found:")
    for m in muscle_cols:
        print(f"  - {m}")

    # Για κάθε μύα φτιάχνουμε 4 plots
    for muscle in muscle_cols:
        mdf = df[["cycle", muscle]].copy()

        # Βρίσκουμε πόσα cycles έχουμε
        cycles = sorted(mdf["cycle"].unique())
        n_cycles = len(cycles)

        # Υποθέτουμε ότι κάθε cycle έχει ίδιο αριθμό σημείων (normalized)
        # Παίρνουμε το πρώτο cycle για να βρούμε N
        N = mdf[mdf["cycle"] == cycles[0]].shape[0]
        x = np.linspace(0, 100, N)

        # Φτιάχνουμε πίνακα (n_cycles x N) με τα lengths
        data = np.zeros((n_cycles, N))
        for i, c in enumerate(cycles):
            vals = mdf[mdf["cycle"] == c][muscle].values
            if len(vals) != N:
                # Σε περίπτωση μικρής απόκλισης κάνουμε απλή παρεμβολή
                old_x = np.linspace(0, 100, len(vals))
                vals = np.interp(x, old_x, vals)
            data[i, :] = vals

        mean_len = data.mean(axis=0)
        std_len = data.std(axis=0)

        # --------- 4-plot figure (2x2) ----------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Muscle: {muscle}", fontsize=16)

        # 1) All cycles + mean
        ax1 = axes[0, 0]
        for i in range(n_cycles):
            ax1.plot(x, data[i, :], alpha=0.4)
        ax1.plot(x, mean_len, color="green", linewidth=2.0, label="Mean")
        ax1.set_title(f"{muscle}\nAll cycles + mean")
        ax1.set_xlabel("Gait cycle (%)")
        ax1.set_ylabel("Length (m)")
        ax1.legend(loc="best")

        # 2) Mean ± SD
        ax2 = axes[0, 1]
        ax2.plot(x, mean_len, linewidth=2.0)
        ax2.fill_between(
            x,
            mean_len - std_len,
            mean_len + std_len,
            alpha=0.3,
        )
        ax2.set_title(f"{muscle}\nMean ± SD")
        ax2.set_xlabel("Gait cycle (%)")
        ax2.set_ylabel("Length (m)")

        # 3) Cycle-wise Peak / Min / ROM (με σωστή κλίμακα)
        ax3 = axes[1, 0]
        peaks = data.max(axis=1)
        mins = data.min(axis=1)
        roms = peaks - mins

        # Θα δείξουμε τα **μέσα** + SD για Peak/Min/ROM
        labels = ["Peak", "Min", "ROM"]
        vals_mean = [peaks.mean(), mins.mean(), roms.mean()]
        vals_std = [peaks.std(), mins.std(), roms.std()]
        xpos = np.arange(3)

        ax3.bar(xpos, vals_mean, yerr=vals_std, capsize=5)
        ax3.set_xticks(xpos)
        ax3.set_xticklabels(labels)
        ax3.set_title(f"{muscle}\nCycle-wise Peak / Min / ROM")
        ax3.set_ylabel("Length (m)")

        # Δυναμική κλίμακα Υ για να φαίνονται καθαρά οι μπάρες
        all_vals = np.concatenate([peaks, mins, roms])
        y_min = all_vals.min()
        y_max = all_vals.max()
        if y_max == y_min:
            # safety: σε περίπτωση σχεδόν σταθερού σήματος
            margin = 0.01
        else:
            margin = 0.1 * (y_max - y_min)
        ax3.set_ylim(y_min - margin, y_max + margin)

        # 4) Heatmap (cycles x gait %)
        ax4 = axes[1, 1]
        im = ax4.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[0, 100, 1, n_cycles + 1],
        )
        ax4.set_title(f"{muscle}\nHeatmap (cycles × gait %)")
        ax4.set_xlabel("Gait cycle (%)")
        ax4.set_ylabel("Cycle index")

        cbar = fig.colorbar(im, ax=ax4)
        cbar.set_label("Length (m)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = os.path.join(out_dir, f"{muscle}_plots_simple.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved plots for {muscle} to {out_path}")

    print("🎉 Done.")


if __name__ == "__main__":
    main()
