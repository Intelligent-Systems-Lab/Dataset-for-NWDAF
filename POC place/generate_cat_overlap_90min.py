#!/usr/bin/env python3
import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import os

# 使用最簡單的相對路徑，相對於專案根目錄
DATASET_PATH = Path("Combined_Dataset")
CLEAN_REGISTRY_PATH = Path("cleaning/clean_sessions_registry.json")

# 載入註冊表
with open(CLEAN_REGISTRY_PATH, 'r') as f:
    CLEAN_REGISTRY = json.load(f)

CAT1_ACTIONS = {"videocall", "audiocall"}
CAT2_ACTIONS = {"chat", "browsing", "search", "social-post", "open-email", "directions", "gaming-online", "video-streaming", "music-streaming"}
CAT3_ACTIONS = {"download", "upload"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--base-time", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="POC place/cat_overlap_90min_data")
    return p.parse_args()

def iso_format(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

def find_session_files(action: str) -> list:
    if action in CLEAN_REGISTRY:
        # 確保路徑拼接正確
        return [DATASET_PATH / p for p in CLEAN_REGISTRY[action]]
    return []

def read_session_packets(file_path: Path) -> tuple:
    try:
        df = pd.read_parquet(file_path)
        header = df.columns.tolist()
        df = df.astype(str)
        df = df.replace({"nan": "", "<NA>": "None", "None": "", "NaN": ""}) 
        rows = df.values.tolist()
        return header, rows
    except Exception:
        return None, None

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
    
    while current_time < flow_end:
        sf = session_files[file_idx % len(session_files)]
        file_idx += 1
        
        header, rows = read_session_packets(sf)
        if not rows:
            continue
            
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
                "ue_ip": ue_ip
            })
            last_included_rel_ts = src_rel_ts
            
        if session_exhausted:
            current_time = last_included_rel_ts + delta + 1e-9
        else:
            break
            
    return collected

def main():
    args = parse_args()
    
    # 直接從資料夾清單檢查可用 Action
    available_actions = [d.name for d in DATASET_PATH.iterdir() if d.is_dir()]
    
    cat1_pool = list(CAT1_ACTIONS.intersection(available_actions))
    cat2_pool = list(CAT2_ACTIONS.intersection(available_actions))
    cat3_pool = list(CAT3_ACTIONS.intersection(available_actions))
    
    base_time = datetime.fromisoformat(args.base_time.replace("Z", "+00:00")) if args.base_time else datetime.now(timezone.utc)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ues = [
        {"ip": "10.0.0.1", "offset": 0.0},
        {"ip": "10.0.0.2", "offset": 15.0},
        {"ip": "10.0.0.3", "offset": 30.0}
    ]
    
    all_packets = []
    phase_dur = 1800.0
    overlap_dur = 120.0
    
    print(f"Generating 90min Overlap Dataset...")
    
    for ue in ues:
        ip = ue["ip"]
        off = ue["offset"]
        
        # Phase 1: CAT1
        a1 = random.choice(cat1_pool)
        t1_start, t1_dur = off, phase_dur + (overlap_dur / 2.0)
        all_packets.extend(build_flow_packets(a1, t1_start, t1_dur, ip))
        
        # Phase 2: CAT2
        a2 = random.choice(cat2_pool)
        t2_start, t2_dur = off + phase_dur - (overlap_dur / 2.0), phase_dur + overlap_dur
        all_packets.extend(build_flow_packets(a2, t2_start, t2_dur, ip))
        
        # Phase 3: CAT3
        a3 = random.choice(cat3_pool)
        t3_start, t3_dur = off + (2 * phase_dur) - (overlap_dur / 2.0), phase_dur + (overlap_dur / 2.0)
        all_packets.extend(build_flow_packets(a3, t3_start, t3_dur, ip))
        
    all_packets.sort(key=lambda x: x["ts"])
    print(f"Total {len(all_packets)} packets. Bucketing...")
    
    buckets = {}
    for pkt in all_packets:
        win_idx = int(pkt["ts"] // args.interval)
        ue_ip = pkt["ue_ip"]
        key = (win_idx, ue_ip)
        if key not in buckets:
            buckets[key] = {"ul_vol": 0, "dl_vol": 0, "ul_pkts": 0, "dl_pkts": 0}
        b = buckets[key]
        if pkt["direction"] == "0":
            b["ul_vol"] += pkt["len"]; b["ul_pkts"] += 1
        elif pkt["direction"] == "1":
            b["dl_vol"] += pkt["len"]; b["dl_pkts"] += 1

    max_win_idx = max(k[0] for k in buckets.keys())
    all_notifications = []
    correlation_id = f"training_{int(base_time.timestamp())}_overlap_90m"

    for win_idx in range(max_win_idx + 1):
        win_start = base_time + timedelta(seconds=win_idx * args.interval)
        win_end = base_time + timedelta(seconds=(win_idx + 1) * args.interval)
        notification_items = []
        for ue in ues:
            b = buckets.get((win_idx, ue["ip"]), {"ul_vol": 0, "dl_vol": 0, "ul_pkts": 0, "dl_pkts": 0})
            ul_t, dl_t = (b["ul_vol"] * 8) / args.interval, (b["dl_vol"] * 8) / args.interval
            notification_items.append({
                "eventType": "USER_DATA_USAGE_MEASURES",
                "timeStamp": iso_format(win_end),
                "ueIpv4Addr": ue["ip"],
                "startTime": iso_format(win_start),
                "userDataUsageMeasurements": [{
                    "volumeMeasurement": {
                        "totalVolume": b["ul_vol"] + b["dl_vol"],
                        "ulVolume": b["ul_vol"], "dlVolume": b["dl_vol"],
                        "totalNbOfPackets": b["ul_pkts"] + b["dl_pkts"],
                        "ulNbOfPackets": b["ul_pkts"], "dlNbOfPackets": b["dl_pkts"],
                    },
                    "throughputMeasurement": {
                        "ulThroughput": f"{ul_t:.0f} bps", "dlThroughput": f"{dl_t:.0f} bps",
                        "ulPacketThroughput": f"{b['ul_pkts']/args.interval:.2f} pps",
                        "dlPacketThroughput": f"{b['dl_pkts']/args.interval:.2f} pps",
                    },
                }],
            })
        all_notifications.append({"notificationItems": notification_items, "correlationId": correlation_id})

    print("Writing output...")
    with open(out_dir / "training_notifications_run001.json", "w") as f:
        json.dump(all_notifications, f, indent=2)
    pd.DataFrame(all_packets).to_parquet(out_dir / "training_packets_run001.parquet", index=False)
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
