#!/usr/bin/env bash
set -e

# Φάκελος εξόδου για τα νέα plots
OUT_DIR="all_muscle_plots_new"
mkdir -p "$OUT_DIR"

echo "▶ Generating plots for S135..."
python plot_muscle_lengths.py S135_G03_D01_B01_T01_muscle_lengths_normcycles.csv "$OUT_DIR"

echo "▶ Generating plots for S146..."
python plot_muscle_lengths.py S146_G03_D01_B01_T01_muscle_lengths_normcycles.csv "$OUT_DIR"

echo "✅ All plots saved in: $OUT_DIR"
