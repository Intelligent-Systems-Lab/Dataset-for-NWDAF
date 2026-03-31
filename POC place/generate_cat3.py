#!/usr/bin/env python3
"""
generate_cat3.py
================
Traffic generator for Category 3 (Bulk Data Transfer) using only
the 'download' action (longest total session duration in Cat3).

Outputs:
  - EES JSON files (training_notifications_runXXX.json)
  - Packet parquet files (training_packets_runXXX.parquet)

No labels are generated.

Usage:
    python generate_cat3.py --runs 15 --min-duration 150 --max-duration 450
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
DATASET_PATH = (SCRIPT_DIR / ".." / "Combined_Dataset").resolve()

# ── Config ────────────────────────────────────────────────────────────
CLEAN_REGISTRY_PATH = (SCRIPT_DIR / ".." / "cleaning" / "clean_sessions_registry.json").resolve()
with open(CLEAN_REGISTRY_PATH, 'r') as f:
    CLEAN_REGISTRY = json.load(f)

# Fixed action for Cat3: download (longest total session duration)
FIXED_ACTION = "download"


def parse_args():
    p = argparse.ArgumentParser(description="Cat3 Traffic Generator (download only)")
    p.add_argument("--interval", type=float, default=5.0, help="Reporting interval (seconds)")
    p.add_argument("--min-duration", type=float, default=150.0, help="Minimum total simulation duration (seconds)")
    p.add_argument("--max-duration", type=float, default=450.0, help="Maximum total simulation duration (seconds)")
    p.add_argument("--runs", type=int, default=15, help="Number of distinct output runs to generate")
    p.add_argument("--base-time", type=str, default=None, help="Base time ISO 8601")
    p.add_argument("--output-dir", type=str, default="cat3", help="Output directory")
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
    """Read dataset Parquet files and build packet rows."""
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

            session_file_name = sf.name

            collected.append({
                "ts": adj_ts,
                "direction": direction,
                "len": pkt_len,
                "action": action,
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


# ── Main Logic ────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Verify the fixed action exists
    session_files = find_session_files(FIXED_ACTION)
    if not session_files:
        print(f"[ERR] No session files found for '{FIXED_ACTION}' in {DATASET_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("Cat3 Traffic Generator (download only)")
    print(f"  Fixed Action    : {FIXED_ACTION}")
    print(f"  Runs requested  : {args.runs}")
    print(f"  Duration Range  : {args.min_duration}s - {args.max_duration}s")
    print(f"  Reporting Intv  : {args.interval}s")
    print(f"  Session files   : {len(session_files)}")
    print("=" * 60)

    base_time = datetime.fromisoformat(args.base_time.replace("Z", "+00:00")) if args.base_time else datetime.now(timezone.utc)
    out_dir = Path(SCRIPT_DIR / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for run_id in range(1, args.runs + 1):
        run_duration = random.uniform(args.min_duration, args.max_duration)
        correlation_id = f"cat3_{int(base_time.timestamp())}_run{run_id:03d}"

        # Strictly 3 UEs
        ues = [
            {"id": "UE_1", "ip": "10.0.0.1"},
            {"id": "UE_2", "ip": "10.0.0.2"},
            {"id": "UE_3", "ip": "10.0.0.3"}
        ]

        all_packets = []

        for ue in ues:
            ip = ue["ip"]
            # No phase transitions — single action for the entire duration
            pkts = build_flow_packets(FIXED_ACTION, 0.0, run_duration, ip)
            all_packets.extend(pkts)

        all_packets.sort(key=lambda x: x["ts"])
        print(f"Run {run_id:02d} ({run_duration:5.1f}s): Generated {len(all_packets)} packets.")

        # ── EES Bucketing ─────────────────────────────────────────────────
        buckets = {}
        for pkt in all_packets:
            win_idx = int(pkt["ts"] // args.interval)
            ue_ip = pkt["ue_ip"]
            key = (win_idx, ue_ip)

            if key not in buckets:
                buckets[key] = {
                    "ue_ip": ue_ip,
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                }

            b = buckets[key]

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
                b = buckets.get((win_idx, ue["ip"]), {
                    "ue_ip": ue["ip"],
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                })
                windows[win_idx].append((ue["ip"], b))

        # ── Output EES JSON ───────────────────────────────────────────────
        all_notifications = []

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

            notification = {
                "notificationItems": notification_items,
                "correlationId": correlation_id,
            }
            all_notifications.append(notification)

        # Write EES JSON
        combined_path = out_dir / f"training_notifications_run{run_id:03d}.json"
        with open(combined_path, "w", encoding="utf-8") as fout:
            json.dump(all_notifications, fout, indent=2, ensure_ascii=False)

        # Write Packet Parquet
        packets_path = out_dir / f"training_packets_run{run_id:03d}.parquet"
        df_packets = pd.DataFrame(all_packets)
        df_packets.to_parquet(packets_path, index=False, compression="snappy")

    print(f"\nSaved {args.runs} separate runs to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
