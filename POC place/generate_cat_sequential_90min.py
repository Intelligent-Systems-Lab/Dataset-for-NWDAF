#!/usr/bin/env python3
import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (SCRIPT_DIR / ".." / "Combined_Dataset").resolve()
CLEAN_REGISTRY_PATH = (SCRIPT_DIR / ".." / "cleaning" / "clean_sessions_registry.json").resolve()
with open(CLEAN_REGISTRY_PATH, 'r') as f:
    CLEAN_REGISTRY = json.load(f)

CAT1_ACTIONS = {"videocall", "audiocall"}
CAT2_ACTIONS = {"chat", "browsing", "search", "social-post", "open-email", "directions", "gaming-online", "video-streaming", "music-streaming"}
CAT3_ACTIONS = {"download", "upload"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--base-time", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="cat_sequential_90min_data")
    return p.parse_args()

def iso_format(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

def find_session_files(action: str) -> list:
    if action in CLEAN_REGISTRY:
        return [(DATASET_PATH / Path(p)).resolve() for p in CLEAN_REGISTRY[action]]
    return []

def read_session_packets(file_path: Path) -> tuple:
    df = pd.read_parquet(file_path)
    header = df.columns.tolist()
    df = df.astype(str)
    df = df.replace({"nan": "", "<NA>": "None", "None": "", "NaN": ""}) 
    rows = df.values.tolist()
    return header, rows

def parse_timestamp(value: str) -> float:
    try:
        return float(value.strip().strip('"'))
    except ValueError:
        return 0.0

def build_flow_packets(action: str, start: float, duration: float, ue_ip: str) -> list:
    session_files = find_session_files(action)
    if not session_files:
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
            try:
                pkt_len = int(float(row[len_col])) if row[len_col] else 0
            except ValueError:
                pkt_len = 0

            collected.append({
                "ts": adj_ts,
                "direction": direction,
                "len": pkt_len,
                "action": action,
                "ue_ip": ue_ip,
                "session_file": sf.name
            })
            
            last_included_rel_ts = src_rel_ts
            
        if not rows or last_included_rel_ts is None:
            break
            
        if session_exhausted:
            current_time = last_included_rel_ts + delta + 1e-9
        else:
            break
            
    return collected

def get_available_actions():
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
        sys.exit(1)
        
    cat1_pool = list(CAT1_ACTIONS.intersection(available_actions))
    cat2_pool = list(CAT2_ACTIONS.intersection(available_actions))
    cat3_pool = list(CAT3_ACTIONS.intersection(available_actions))
    
    base_time = datetime.fromisoformat(args.base_time.replace("Z", "+00:00")) if args.base_time else datetime.now(timezone.utc)
    out_dir = Path(SCRIPT_DIR / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = 1
    correlation_id = f"training_{int(base_time.timestamp())}_sequential_90m"

    ues = [
        {"id": "UE_1", "ip": "10.0.0.1", "offset": 0.0},
        {"id": "UE_2", "ip": "10.0.0.2", "offset": 15.0},
        {"id": "UE_3", "ip": "10.0.0.3", "offset": 30.0}
    ]
    
    all_packets = []
    phase_duration = 1800.0 # 30 mins per phase
    
    print("Generating CAT1 -> CAT2 -> CAT3 sequence (1800s each) for 3 UEs...")
    
    for ue in ues:
        ip = ue["ip"]
        offset = ue["offset"]
        
        # Phase 1: CAT1
        action1 = random.choice(cat1_pool)
        print(f"[{ip}] Phase 1 (CAT1): {action1} at start={offset}s")
        pkts1 = build_flow_packets(action1, offset, phase_duration, ip)
        all_packets.extend(pkts1)
        
        # Phase 2: CAT2
        action2 = random.choice(cat2_pool)
        t_start2 = offset + phase_duration
        print(f"[{ip}] Phase 2 (CAT2): {action2} at start={t_start2}s")
        pkts2 = build_flow_packets(action2, t_start2, phase_duration, ip)
        all_packets.extend(pkts2)
        
        # Phase 3: CAT3
        action3 = random.choice(cat3_pool)
        t_start3 = offset + 2 * phase_duration
        print(f"[{ip}] Phase 3 (CAT3): {action3} at start={t_start3}s")
        pkts3 = build_flow_packets(action3, t_start3, phase_duration, ip)
        all_packets.extend(pkts3)
        
    all_packets.sort(key=lambda x: x["ts"])
    print(f"Total {len(all_packets)} packets generated. Bucketing into intervals...")
    
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
        print("[WARN] No traffic generated.")
        return

    max_win_idx = max(k[0] for k in buckets.keys())
    windows = {}
    
    for win_idx in range(max_win_idx + 1):
        windows[win_idx] = []
        for ue in ues:
            b = buckets.get((win_idx, ue["ip"]), {
                "ue_ip": ue["ip"],
                "ul_vol": 0, "dl_vol": 0,
                "ul_pkts": 0, "dl_pkts": 0,
                "labels": {"idle"}
            })
            windows[win_idx].append((ue["ip"], b))

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

    print("Writing files...")
    combined_path = out_dir / f"training_notifications_run001.json"
    with open(combined_path, "w", encoding="utf-8") as fout:
        json.dump(all_notifications, fout, indent=2, ensure_ascii=False)

    labels_path = out_dir / f"training_labels_run001.parquet"
    df_labels = pd.DataFrame(labels_rows[1:], columns=labels_rows[0])
    df_labels.to_parquet(labels_path, index=False, compression="snappy")

    packets_path = out_dir / f"training_packets_run001.parquet"
    df_packets = pd.DataFrame(all_packets)
    df_packets.to_parquet(packets_path, index=False, compression="snappy")

    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
