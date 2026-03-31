import json
import os
from pathlib import Path
import pandas as pd

def calculate_folder_duration(folder_path):
    total_seconds = 0.0
    
    # Check for raw packet stream parquets
    parquet_files = list(folder_path.glob("training_packets_run*.parquet"))
    
    if parquet_files:
        for pf in parquet_files:
            try:
                # Load parquet and use floating point 'ts' (or 'relative_time') array
                df = pd.read_parquet(pf, engine='pyarrow')
                
                if "ts" in df.columns:
                    duration = df["ts"].max() - df["ts"].min()
                elif "relative_time" in df.columns:
                    duration = df["relative_time"].max() - df["relative_time"].min()
                else:
                    duration = 0.0
                    
                total_seconds += float(duration)
            except Exception as e:
                print(f"[WARN] Failed to parse Parquet {pf}: {e}")
                
        return total_seconds
        
    return 0.0

def main():
    base_dir = Path(r"c:\Users\User\Desktop\workspace\Dataset-for-NWDAF\POC place")
    results = {}
    
    # Iterate through all items in POC place
    for item in base_dir.iterdir():
        if item.is_dir():
            dur = calculate_folder_duration(item)
            if dur > 0:
                results[item.name] = dur
                
    # Include the large combined packet file if it exists
    combined_pf = base_dir / "combined_packetbypacket.parquet"
    if combined_pf.exists():
        try:
            df = pd.read_parquet(combined_pf, engine='pyarrow')
            if "ts" in df.columns:
                dur = df["ts"].max() - df["ts"].min()
                results["combined_packetbypacket"] = float(dur)
        except Exception as e:
            print(f"Warning on combined file: {e}")

    out_file = base_dir / "meta.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Calculated RAW PACKET exact durations for {len(results)} datasets. Wrote to {out_file.name}")
    for k, v in results.items():
        print(f" - {k}: {v} seconds")

if __name__ == "__main__":
    main()
