#!/usr/bin/env python3
"""
generate_cat123_training.py
=========================
End-to-end dataset generator producing STRICTLY compliant EES JSON files
alongside a separate labels parquet file for ML training.

Features:
1. Excludes 'background'.
2. Generates exactly 3 UEs.
3. Randomizes phase transitions and total duration per run for diversity.
4. Simulates Category 1 (Real-time), Category 2 (Bursty), and Category 3 (Bulk data) traffic.
5. Outputs multiple files (runs) to provide varied datasets (default: 15 runs).

Usage:
    python generate_cat123_training.py --runs 15 --min-duration 200 --max-duration 400
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
# Adjusted DATASET_PATH to point to the main folder from 'POC place'
DATASET_PATH = (SCRIPT_DIR / ".." / "Combined_Dataset").resolve()

# ── Config ────────────────────────────────────────────────────────────
CLEAN_REGISTRY_PATH = (SCRIPT_DIR / ".." / "cleaning" / "clean_sessions_registry.json").resolve()
with open(CLEAN_REGISTRY_PATH, 'r') as f:
    CLEAN_REGISTRY = json.load(f)

# We define Cat1, Cat2, Cat3 actions based on user requirements.
CAT1_ACTIONS = {"videocall", "audiocall"}
CAT2_ACTIONS = {"chat", "browsing", "search", "social-post", "open-email", "directions", "gaming-online", "video-streaming", "music-streaming"}
CAT3_ACTIONS = {"download", "upload"}


def parse_args():
    p = argparse.ArgumentParser(description="Cat1/Cat2/Cat3 Training Dataset Generator")
    p.add_argument("--interval", type=float, default=5.0, help="Reporting interval (seconds)")
    p.add_argument("--min-duration", type=float, default=150.0, help="Minimum total simulation duration (seconds)")
    p.add_argument("--max-duration", type=float, default=450.0, help="Maximum total simulation duration (seconds)")
    p.add_argument("--runs", type=int, default=15, help="Number of distinct output runs to generate")
    p.add_argument("--base-time", type=str, default=None, help="Base time ISO 8601")
    p.add_argument("--output-dir", type=str, default="cat123_training_data", help="Output directory")
    return p.parse_args()


def iso_format(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


# ── Packet Reading Logic ──────────────────────────────────────────────
def find_session_files(action: str) -> list:
    """Find pure session Parquet files using the pre-computed clean registry."""
    if action in CLEAN_REGISTRY:
        # Resolve to absolute paths based on DATASET_PATH
        return [(DATASET_PATH / Path(p)).resolve() for p in CLEAN_REGISTRY[action]]
    return []


def read_session_packets(file_path: Path) -> tuple:
    df = pd.read_parquet(file_path)
    header = df.columns.tolist()
    
    # Convert all columns to string, then fill NA with empty string
    df = df.astype(str)
    # pandas astype(str) often converts NaN to the literal string "nan" or "<NA>"
    df = df.replace({"nan": "", "<NA>": "None", "None": "", "NaN": ""}) 
    rows = df.values.tolist()
    return header, rows


def parse_timestamp(value: str) -> float:
    try:
        return float(value.strip().strip('"'))
    except ValueError:
        return 0.0


def build_flow_packets(action: str, start: float, duration: float, ue_ip: str) -> list:
    """Read dataset Parquet files and build packet rows mapping src/dst IPs."""
    session_files = find_session_files(action)
    if not session_files:
        print(f"[WARN] No session files found for action: {action}")
        return []
        
    random.shuffle(session_files)
    flow_end = start + duration
    collected = []
    
    current_time = start
    file_idx = 0
    unified_header = None
    
    while current_time < flow_end:
        sf = session_files[file_idx % len(session_files)]
        file_idx += 1
        
        try:
            header, rows = read_session_packets(sf)
        except Exception as e:
            print(f"[ERR] Failed to read {sf}: {e}")
            continue

        if not rows:
            continue
            
        if unified_header is None:
            unified_header = header
            
        try:
            ts_col = header.index("relative_time")
            dir_col = header.index("direction")
            len_col = header.index("pkt_len")
        except ValueError:
            continue
            
        delta = current_time
        last_included_rel_ts = None
        session_exhausted = True
        
        for row in rows:
            src_rel_ts = parse_timestamp(row[ts_col])
            adj_ts = src_rel_ts + delta
            
            if adj_ts > flow_end:
                session_exhausted = False
                break

            direction = row[dir_col].strip()
            pkt_len = int(float(row[len_col])) if row[len_col] else 0

            # Keep track of from which parquet this data came
            session_file_name = sf.name

            # Store minimum info required for EES bucketing: timestamp, direction, len, action
            collected.append({
                "ts": adj_ts,
                "direction": direction,
                "len": pkt_len,
                "action": action, # Retaining original action for labels
                "ue_ip": ue_ip,
                "session_file": session_file_name
            })
            
            last_included_rel_ts = src_rel_ts
            
        if not rows or last_included_rel_ts is None:
            break
            
        if session_exhausted:
            current_time = last_included_rel_ts + delta + 1e-9
        else:
            break
            
    return collected


# ── Scenario Generation & Main Logic ──────────────────────────────────

def get_available_actions():
    """Scan Combined_Dataset."""
    available = []
    if not DATASET_PATH.exists():
        return available
    for act_dir in DATASET_PATH.iterdir():
        if act_dir.is_dir():
            available.append(act_dir.name)
    return available


def main():
    args = parse_args()
    
    available_actions = get_available_actions()
    if not available_actions:
        print(f"[ERR] No valid datasets found in {DATASET_PATH}")
        sys.exit(1)
        
    cat1_pool = list(CAT1_ACTIONS.intersection(available_actions))
    cat2_pool = list(CAT2_ACTIONS.intersection(available_actions))
    cat3_pool = list(CAT3_ACTIONS.intersection(available_actions))
    
    # Validation checks
    if not cat1_pool:
        print("[ERR] No Cat1 actions found in dataset.")
        sys.exit(1)
    if not cat2_pool:
        print("[ERR] No Cat2 actions found in dataset.")
        sys.exit(1)
    if not cat3_pool:
        print("[ERR] No Cat3 actions found in dataset.")
        sys.exit(1)

    print("=" * 60)
    print("Cat1/Cat2/Cat3 Training Dataset Generator")
    print(f"  Runs requested  : {args.runs}")
    print(f"  Duration Range  : {args.min_duration}s - {args.max_duration}s")
    print(f"  Reporting Intv  : {args.interval}s")
    print(f"  Cat1 Pool (Real-time): {cat1_pool}")
    print(f"  Cat2 Pool (Bursty)   : {cat2_pool}")
    print(f"  Cat3 Pool (Bulk)     : {cat3_pool}")
    print("=" * 60)

    # Base Time
    base_time = datetime.fromisoformat(args.base_time.replace("Z", "+00:00")) if args.base_time else datetime.now(timezone.utc)
    out_dir = Path(SCRIPT_DIR / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for run_id in range(1, args.runs + 1):
        # Randomize the total simulation length for this dataset combination
        run_duration = random.uniform(args.min_duration, args.max_duration)
        correlation_id = f"training_{int(base_time.timestamp())}_run{run_id:03d}"

        # Strictly 3 UEs
        ues = [
            {"id": "UE_1", "ip": "10.0.0.1"},
            {"id": "UE_2", "ip": "10.0.0.2"},
            {"id": "UE_3", "ip": "10.0.0.3"}
        ]
        
        all_packets = []
        
        for ue in ues:
            ip = ue["ip"]
            
            # For each UE, we split their timeline into 2 or 3 phases to simulate transitions.
            num_phases = random.choice([2, 3])
            
            mode_choices = ["Cat1", "Cat2", "Cat3"]

            if num_phases == 2:
                cut1 = random.uniform(0.3, 0.7)
                dur_p1 = run_duration * cut1
                dur_p2 = run_duration - dur_p1
                
                # Pick two distinct modes
                m1 = random.choice(mode_choices)
                m2 = random.choice([m for m in mode_choices if m != m1])
                modes = [m1, m2]
                
                t_starts = [0.0, dur_p1]
                durs = [dur_p1, dur_p2]
                
            else: # 3 phases
                cut1 = random.uniform(0.2, 0.45)
                cut2 = random.uniform(0.2, 0.45)
                dur_p1 = run_duration * cut1
                dur_p2 = run_duration * cut2
                dur_p3 = run_duration - dur_p1 - dur_p2
                
                m1 = random.choice(mode_choices)
                m2 = random.choice([m for m in mode_choices if m != m1])
                m3 = random.choice([m for m in mode_choices if m != m2])
                modes = [m1, m2, m3]
                
                t_starts = [0.0, dur_p1, dur_p1 + dur_p2]
                durs = [dur_p1, dur_p2, dur_p3]
            
            for phase_idx in range(num_phases):
                mode = modes[phase_idx]
                t_start = t_starts[phase_idx]
                p_dur = durs[phase_idx]
                
                if mode == "Cat1":
                    action = random.choice(cat1_pool)
                elif mode == "Cat2":
                    action = random.choice(cat2_pool)
                else:
                    action = random.choice(cat3_pool)
                
                # Add packets for this phase
                pkts = build_flow_packets(action, t_start, p_dur, ip)
                all_packets.extend(pkts)
                
                # Optionally add some background noise (a concurrent action from a different category)
                if random.random() < 0.2:
                    noise_mode = random.choice([m for m in mode_choices if m != mode])
                    if noise_mode == "Cat1":
                        noise_pool = cat1_pool
                    elif noise_mode == "Cat2":
                        noise_pool = cat2_pool
                    else:
                        noise_pool = cat3_pool
                    noise_action = random.choice(noise_pool)
                    pkts_noise = build_flow_packets(noise_action, t_start, p_dur, ip)
                    all_packets.extend(pkts_noise)
            
        all_packets.sort(key=lambda x: x["ts"])
        print(f"Run {run_id:02d} ({run_duration:5.1f}s): Generated {len(all_packets)} packets.")
        
        # ── EES Bucketing ─────────────────────────────────────────────────
        buckets = {}
        for pkt in all_packets:
            win_idx = int(pkt["ts"] // args.interval)
            ue_ip = pkt["ue_ip"]
            action = pkt["action"]
            key = (win_idx, ue_ip)
            
            if key not in buckets:
                buckets[key] = {
                    "ue_ip": ue_ip,
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                    "labels": set()
                }
                
            b = buckets[key]
            b["labels"].add(action)
            
            if pkt["direction"] == "0":
                b["ul_vol"] += pkt["len"]
                b["ul_pkts"] += 1
            elif pkt["direction"] == "1":
                b["dl_vol"] += pkt["len"]
                b["dl_pkts"] += 1

        if not buckets:
            print(f"[WARN] Run {run_id:02d}: No traffic generated. Skipping.")
            continue

        max_win_idx = max(k[0] for k in buckets.keys())
        windows = {}
        
        for win_idx in range(max_win_idx + 1):
            windows[win_idx] = []
            for ue in ues:
                # Zero-filling for empty intervals
                b = buckets.get((win_idx, ue["ip"]), {
                    "ue_ip": ue["ip"],
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                    "labels": {"idle"} # Empty interval = idle
                })
                windows[win_idx].append((ue["ip"], b))

        # ── Output EES JSON & Labels Parquet ──────────────────────────────
        all_notifications = []
        labels_rows = [["window_index", "start_time", "end_time", "ue_ip", "ground_truth_labels"]]

        for win_idx in sorted(windows.keys()):
            win_start = base_time + timedelta(seconds=win_idx * args.interval)
            win_end = base_time + timedelta(seconds=(win_idx + 1) * args.interval)
            notification_items = []

            for ue_ip, b in sorted(windows[win_idx], key=lambda x: x[0]):
                total_vol = b["ul_vol"] + b["dl_vol"]
                total_pkts = b["ul_pkts"] + b["dl_pkts"]

                ul_throughput = (b["ul_vol"] * 8) / args.interval
                dl_throughput = (b["dl_vol"] * 8) / args.interval
                ul_pkt_throughput = b["ul_pkts"] / args.interval
                dl_pkt_throughput = b["dl_pkts"] / args.interval

                item = {
                    "eventType": "USER_DATA_USAGE_MEASURES",
                    "timeStamp": iso_format(win_end),
                    "ueIpv4Addr": ue_ip,
                    "startTime": iso_format(win_start),
                    "userDataUsageMeasurements": [
                        {
                            "volumeMeasurement": {
                                "totalVolume": total_vol,
                                "ulVolume": b["ul_vol"],
                                "dlVolume": b["dl_vol"],
                                "totalNbOfPackets": total_pkts,
                                "ulNbOfPackets": b["ul_pkts"],
                                "dlNbOfPackets": b["dl_pkts"],
                            },
                            "throughputMeasurement": {
                                "ulThroughput": f"{ul_throughput:.0f} bps",
                                "dlThroughput": f"{dl_throughput:.0f} bps",
                                "ulPacketThroughput": f"{ul_pkt_throughput:.2f} pps",
                                "dlPacketThroughput": f"{dl_pkt_throughput:.2f} pps",
                            },
                        }
                    ],
                }
                notification_items.append(item)
                
                # Record label maintaining original actions, like 'videocall|download'
                labels_str = "|".join(sorted(list(b["labels"])))
                labels_rows.append([
                    win_idx,
                    iso_format(win_start),
                    iso_format(win_end),
                    ue_ip,
                    labels_str
                ])

            notification = {
                "notificationItems": notification_items,
                "correlationId": correlation_id,
            }
            all_notifications.append(notification)

        # Write Combined JSON for this run
        combined_path = out_dir / f"training_notifications_run{run_id:03d}.json"
        with open(combined_path, "w", encoding="utf-8") as fout:
            json.dump(all_notifications, fout, indent=2, ensure_ascii=False)

        # Write Labels Parquet for this run
        labels_path = out_dir / f"training_labels_run{run_id:03d}.parquet"
        df_labels = pd.DataFrame(labels_rows[1:], columns=labels_rows[0])
        df_labels.to_parquet(labels_path, index=False, compression="snappy")

        # Write Packet-by-Packet Parquet for this run
        packets_path = out_dir / f"training_packets_run{run_id:03d}.parquet"
        df_packets = pd.DataFrame(all_packets)
        df_packets.to_parquet(packets_path, index=False, compression="snappy")

    print(f"\nSaved {args.runs} separate runs to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
