import os
import pandas as pd

def convert_parquet_to_csv(path):
    # Read the .parquet file
    df = pd.read_parquet(path)
    
    # Replace the .parquet extension with .csv and save
    csv_path = path.replace('.parquet', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Converted {path} to {csv_path}")

def main():
    # Walk through all files in the current directory and subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            # Check if the file is a .parquet file
            if file.endswith(".parquet"):
                path = os.path.join(root, file)
                convert_parquet_to_csv(path)

if __name__ == "__main__":
    main()
