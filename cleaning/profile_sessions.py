#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET_DIR = Path(r"C:\Users\lcc04\Desktop\workspace\dataset\Combined_Dataset")
OUTPUT_CSV = Path(r"C:\Users\lcc04\Desktop\workspace\dataset\cleaning\session_profiles.csv")

def process_file(f_path):
    try:
        try:
            df = pd.read_parquet(f_path, columns=['relative_time', 'pkt_len'])
        except ValueError:
            df = pd.read_parquet(f_path)
            
        if df.empty:
            return None
            
        if 'relative_time' not in df.columns or 'pkt_len' not in df.columns:
            return {
                'file_path': str(f_path.relative_to(DATASET_DIR)),
                'action': f_path.parent.parent.name,
                'duration_sec': 0.0,
                'total_bytes': 0,
                'mean_mbps': 0.0,
                'cv': 0.0,
                'idle_ratio': 1.0,
                'status': 'Missing Schema Fields'
            }
        
        df['relative_time'] = pd.to_numeric(df['relative_time'], errors='coerce')
        df['pkt_len'] = pd.to_numeric(df['pkt_len'], errors='coerce').fillna(0)
        
        df = df.dropna(subset=['relative_time'])
        if df.empty:
            return None
            
        dur = df['relative_time'].max()
        
        if pd.isna(dur) or dur < 2.0:
            return {
                'file_path': str(f_path.relative_to(DATASET_DIR)),
                'action': f_path.parent.parent.name,
                'duration_sec': dur if pd.notna(dur) else 0.0,
                'total_bytes': df['pkt_len'].sum(),
                'mean_mbps': 0.0,
                'cv': 0.0,
                'idle_ratio': 1.0,
                'status': 'Too Short (< 2 sec)'
            }
            
        df['sec'] = np.floor(df['relative_time']).astype(int)
        grouped = df.groupby('sec')['pkt_len'].sum()
        
        max_sec = int(df['sec'].max())
        idx = pd.Index(range(max_sec + 1), name='sec')
        grouped = grouped.reindex(idx, fill_value=0.0)
        
        mbps_arr = (grouped.values * 8) / 1_000_000
        
        total_bytes = df['pkt_len'].sum()
        mean_mbps = np.mean(mbps_arr) if len(mbps_arr) > 0 else 0
        std_mbps = np.std(mbps_arr) if len(mbps_arr) > 0 else 0
        
        cv = (std_mbps / mean_mbps) if mean_mbps > 0 else 0.0
        
        zeros_cnt = np.sum(mbps_arr == 0)
        idle_ratio = zeros_cnt / len(mbps_arr) if len(mbps_arr) > 0 else 1.0
        
        return {
            'file_path': str(f_path.relative_to(DATASET_DIR)),
            'action': f_path.parent.parent.name,
            'duration_sec': round(dur, 2),
            'total_bytes': total_bytes,
            'mean_mbps': round(mean_mbps, 3),
            'cv': round(cv, 3),
            'idle_ratio': round(idle_ratio, 3),
            'status': 'Valid'
        }
    except Exception as e:
        return {
            'file_path': str(f_path.relative_to(DATASET_DIR)),
            'action': f_path.parent.parent.name,
            'duration_sec': 0.0,
            'total_bytes': 0,
            'mean_mbps': 0.0,
            'cv': 0.0,
            'idle_ratio': 1.0,
            'status': f'Error: {str(e)}'
        }

def main():
    start = time.time()
    print("Initiating Fast Combined Dataset Analysis Pipeline...")
    all_files = list(DATASET_DIR.glob("*/*/*.parquet"))
    total = len(all_files)
    print(f"Discovered {total} unique session parity files.")
    print("Spawning process pool for multi-core processing...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            
            completed += 1
            if completed % 1000 == 0:
                print(f"Processed {completed}/{total} files... ({(completed)/total*100:.1f}%)")
                
    print(f"\nFinalizing data extraction, writing to {OUTPUT_CSV}...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)
    
    elapsed = time.time() - start
    print(f"\nComplete! Profiled {len(results)} sessions in {elapsed:.1f} seconds.")
    
    print("=" * 60)
    print("Quick Summary by Category (Valid Sessions Only):")
    valid_df = df_res[df_res['status'] == 'Valid']
    summary = valid_df.groupby('action').agg({
        'file_path': 'count',
        'mean_mbps': 'mean',
        'cv': 'mean',
        'idle_ratio': 'mean'
    }).rename(columns={'file_path': 'count'}).round(3)
    print(summary)
    print("=" * 60)
    print(f"Report saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
