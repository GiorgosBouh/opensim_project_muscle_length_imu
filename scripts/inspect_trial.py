import sys
import os
import pandas as pd

def main():
    # Check argument
    if len(sys.argv) < 2:
        print("Usage: python inspect_trial.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Check if file exists
    if not os.path.isfile(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    print(f"\n📄 Reading file: {csv_path}\n")

    # Try loading with comma separator first
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # If comma fails, try semicolon
        df = pd.read_csv(csv_path, sep=';')

    print("➡ Headers (columns):")
    print(list(df.columns))

    print("\n➡ Shape (rows, columns):", df.shape)

    print("\n➡ First 5 rows:")
    print(df.head())

    print("\n➡ Column info:")
    print(df.info())

if __name__ == "__main__":
    main()
