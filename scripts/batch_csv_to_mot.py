import os
import glob
import subprocess

# φάκελοι με trials
SUBJECT_FOLDERS = ["S135", "S146"]

def main():
    for subj in SUBJECT_FOLDERS:
        if not os.path.isdir(subj):
            print(f"❌ Folder not found: {subj}")
            continue

        pattern = os.path.join(subj, "*.csv")
        csv_files = sorted(glob.glob(pattern))

        if not csv_files:
            print(f"⚠ No CSV files found in {subj}")
            continue

        print(f"\n📂 Processing subject folder: {subj}")
        for csv_path in csv_files:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            mot_name = f"{base}_gait2392.mot"
            mot_path = os.path.join(subj, mot_name)

            if os.path.isfile(mot_path):
                print(f"  ➝ Skipping (already exists): {mot_path}")
                continue

            cmd = [
                "python",
                "csv_to_gait2392_mot.py",
                csv_path,
                mot_path
            ]
            print(f"  ➝ Converting: {csv_path}  →  {mot_path}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Error converting {csv_path}: {e}")

    print("\n✅ Batch conversion finished.")

if __name__ == "__main__":
    main()
