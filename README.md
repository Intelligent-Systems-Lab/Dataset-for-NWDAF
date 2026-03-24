# Network Traffic Dataset & UE Simulator for NWDAF

A unified, packet-level network traffic dataset and simulation toolkit for NWDAF (Network Data Analytics Function) research.
We harmonize **MIRAGE-AppAct2024** and **UTMobileNet2021** into a single schema, then provide tools to simulate multi-UE traffic scenarios and package them as UPF-EES notifications.

---

## Table of Contents

1. [Key Features](#1-key-features)
2. [Dataset Structure](#2-dataset-structure)
3. [Action Unification](#3-action-unification)
4. [Workflow](#4-workflow)
5. [Simulation Configuration](#5-simulation-configuration)
6. [UPF-EES Notification Packager](#6-upf-ees-notification-packager)
7. [Known Limitations](#7-known-limitations)

---

## 1. Key Features

| Feature | Description |
| :--- | :--- |
| **Unified Schema** | Both datasets mapped to 16 standardized columns |
| **Action-based Hierarchy** | Files organized as `<action>/<genre>/` for behavior-centric analysis |
| **Microsecond Precision** | `relative_time` and `iat` preserve 6–9 decimal places |
| **Packet-Level Granularity** | Every packet preserved as a row — no aggregation |
| **Session Alignment** | All sessions rebased to `relative_time = 0.0` |
| **Multi-UE Simulation** | Compose realistic UPF-scale traffic from real session data |
| **UPF-EES Export** | Convert simulated traffic to 3GPP-compliant notification JSON |

### Source Datasets

| Dataset | Apps | Sessions | Description |
| :--- | :---: | ---: | :--- |
| **MIRAGE-AppAct2024** | ~20 | 106,145 | Discord, Zoom, Twitch, etc. — rich per-flow metadata from JSON |
| **UTMobileNet2021** | ~27 | 3,400 | YouTube, Netflix, Spotify, etc. — controlled mobile traces from CSV |

---

## 2. Dataset Structure

### Directory Layout

```
Combined_Dataset/
├── videocall/VoIP/                  # 31,774 sessions (2,025.9h)
├── audiocall/VoIP/                  # 25,665 sessions (1,142.0h)
├── background/
│   ├── VoIP/                        # 27,343 sessions (389.2h)
│   ├── Video_Streaming/             #  1,339 sessions (5.6h)
│   └── Gaming/                      #    142 sessions (0.3h)
├── video-streaming/
│   ├── Video_Streaming/             # 10,845 sessions (360.5h)
│   └── VoIP/                        #  2,500 sessions (6.8h)
├── chat/VoIP/                       #  6,858 sessions (343.1h)
├── browsing/
│   ├── Social_Media/                #    656 sessions (5.4h)
│   ├── Video_Streaming/             #    233 sessions (2.3h)
│   └── Navigation/                  #    209 sessions (0.6h)
├── gaming-online/Gaming/            #    309 sessions (6.6h)
├── social-post/Social_Media/        #    276 sessions (46.5m)
├── upload/File_Transfer/            #    251 sessions (1.6h)
├── download/
│   ├── File_Transfer/               #    250 sessions (2.9h)
│   └── Navigation/                  #    126 sessions (0.1h)
├── directions/Navigation/           #    142 sessions (1.9h)
├── search/
│   ├── Social_Media/                #    132 sessions (1.6h)
│   ├── Music_Streaming/             #    126 sessions (1.7h)
│   └── Video_Streaming/             #    115 sessions (0.3h)
├── open-email/Email/                #    127 sessions (28.2m)
└── music-streaming/Music_Streaming/ #    127 sessions (23.1m)
```

**Total: 109,545 sessions ~4,330 hours** (MIRAGE: 106,145 / UTMobileNet: 3,400)

### Unified Column Format

Every Parquet file contains exactly these 16 standardized columns:

| # | Column | Description |
| ---: | :--- | :--- |
| 1 | `source_dataset` | `MIRAGE` or `UTMobileNet` |
| 2 | `app_name` | e.g. `Discord`, `youtube` |
| 3 | `category` | Genre — e.g. `VoIP`, `Video_Streaming` |
| 4 | `activity` | **Unified action** — e.g. `videocall`, `browsing` |
| 5 | `session_id` | Unique identifier for the flow/session |
| 6 | `session_duration` | Total duration of the session (seconds) |
| 7 | `relative_time` | Timestamp relative to session start (start = 0.0) |
| 8 | `pkt_len` | Packet length in bytes |
| 9 | `l4_proto` | Transport protocol (6 = TCP, 17 = UDP) |
| 10 | `src_ip` | Source IP address |
| 11 | `dst_ip` | Destination IP address |
| 12 | `src_port` | Source port |
| 13 | `dst_port` | Destination port |
| 14 | `tcp_flags` | TCP flags string (e.g. `PA`, `S`, `F`) |
| 15 | `direction` | `0` = Uplink (UE → server), `1` = Downlink (server → UE) |
| 16 | `iat` | Inter-arrival time (seconds since previous packet) |

---

## 3. Action Unification

Both datasets' raw action labels are mapped to **14 unified actions** via `ACTION_UNIFICATION_MAP` in `unify_datasets.py`.

### 3.1 MIRAGE (from `BF_activity` field)

Compound labels (e.g. `videocall-audiocall`) use the **primary-first rule**: the first activity determines classification.

| Unified Action | Raw Labels | Justification |
| :--- | :--- | :--- |
| `videocall` | `videocall`, `chat-videocall`, `videocall-chat`, `videocall-audiocall` | Large-packet ratio 11–17% (video frames >1 KB) |
| `audiocall` | `audiocall`, `chat-audiocall`, `audiocall-chat`, `audiocall-videocall` | Large-packet ratio 7–10% (voice codecs <500 B) |
| `video-streaming` | `video-streaming`, `video on-demand` | UL byte% ~30%, high large-pkt ratio, IAT <20 ms |
| `chat` | `chat` | Bidirectional small packets, high IAT (~0.46 s) |
| `gaming-online` | `gaming-online` | UL byte% ~22%, very large packets |
| `background` | `None`, `Unknown` | No labeled activity |

> **Key distinction**: `videocall-audiocall` → `videocall` (large-pkt rate 11.5% > pure audiocall 10.3%), while `audiocall-videocall` → `audiocall` (large-pkt rate 10.3% ≈ pure audiocall). Similarity scores: 88.8%–97.4%.

### 3.2 UTMobileNet (from filename parsing)

| Unified Action | Raw Labels | Apps |
| :--- | :--- | :--- |
| `browsing` | `scroll-newsfeed`, `browse`, `browse-home`, `scroll-home`, `scroll-feed`, `IgSearchBrowse`, `tap-board`, `explore` | facebook, reddit, hulu, netflix, instagram, pinterest, google-maps |
| `video-streaming` | `watch-video`, `play-video` | hulu, netflix, youtube |
| `chat` | `send-message`, `hangout` | messenger, hangout |
| `search` | `search-page`, `search-music`, `catSearch` | facebook, spotify, youtube |
| `download` | `download`, `download-map` | dropbox, google-drive, google-maps |
| `upload` | `upload` | dropbox, google-drive |
| `social-post` | `post`, `post-tweet` | reddit, twitter |
| `music-streaming` | `play-music` | spotify |
| `open-email` | `open-email` | gmail |
| `directions` | `directions` | google-maps |

### 3.3 Cross-Dataset Validation

Two unified actions contain data from **both** datasets, confirming classification consistency:

- **`video-streaming/Video_Streaming`**: MIRAGE 10,466 + UTMobileNet 379 sessions
- **`chat/VoIP`**: MIRAGE 6,607 + UTMobileNet 251 sessions

---

## 4. Workflow

### Step 1 — Generate Unified Dataset

```bash
python unify_datasets.py
```

Processes raw MIRAGE JSONs and UTMobileNet CSVs into the `Combined_Dataset/` folder with the standardized column format.

### Step 2 — Simulate Multi-UE Traffic

```bash
python simulate_ue.py scenario_config.json
```

Composes realistic multi-UE traffic from real session data based on a scenario configuration. Outputs a merged CSV and a JSON metadata file.

### Step 3 — Package as UPF-EES Notifications *(optional)*

```bash
python upf_ees_packager.py simulated_traffic.parquet --interval 30 --base-time 2026-01-14T12:00:00Z
```

Aggregates the simulated traffic into 3GPP-compliant UPF-EES notification JSON files with per-UE volume and throughput measurements.

---

## 5. Simulation Configuration (`scenario_config.json`)

```json
{
    "correlation_id": "sim_001",
    "output_file": "simulated_traffic.parquet",
    "UEs": [
        {
            "UE_ID": "UE_001",
            "IP_address": "10.10.0.1",
            "flows": [
                {
                    "action": "videocall",
                    "genre": "VoIP",
                    "app": "Discord",
                    "start": 0.0,
                    "duration": 60.0
                },
                {
                    "action": "browsing",
                    "genre": "Social_Media",
                    "start": 30.0,
                    "duration": 120.0
                }
            ]
        },
        {
            "UE_ID": "UE_002",
            "IP_address": "10.10.0.2",
            "flows": [
                {
                    "action": "video-streaming",
                    "genre": "Video_Streaming",
                    "start": 10.0,
                    "duration": 200.0
                }
            ]
        }
    ]
}
```

### Configuration Fields

| Field | Required | Description |
| :--- | :---: | :--- |
| `correlation_id` | No | Identifier for this simulation run |
| `output_file` | No | Output Parquet filename (default: `simulated_ue_traffic.parquet`) |
| `UEs[].UE_ID` | Yes | Unique identifier for the UE |
| `UEs[].IP_address` | Yes | UE IPv4 address (overwrites client-side IPs in packets) |
| `flows[].action` | Yes | Must match an `<action>` folder in `Combined_Dataset/` |
| `flows[].genre` | Yes | Must match a `<genre>` folder under the action |
| `flows[].app` | No | Filter sessions by app name (case-insensitive substring match) |
| `flows[].start` | Yes | Flow start time in seconds |
| `flows[].duration` | Yes | Flow duration in seconds |
| `flows[].file_path` | No | Pinned session files for deterministic replay (see below) |

### How It Works

1. **Session Selection** — For each flow, session files are randomly selected from `Combined_Dataset/<action>/<genre>/`.
2. **Automatic Stitching** — If a session is shorter than the requested duration, additional sessions are concatenated end-to-end until the duration is filled.
3. **IP Replacement** — The configured `IP_address` replaces the UE-side IP in every packet:
   - Uplink (`direction=0`): `src_ip` → UE's IP
   - Downlink (`direction=1`): `dst_ip` → UE's IP
4. **Timestamp Rebasing** — Each flow's packets are shifted to start at the configured `start` time.
5. **Merging** — All UEs' packets are merged into a single time-sorted Parquet.

### Output Files

| File | Description |
| :--- | :--- |
| `simulated_traffic.parquet` | Merged packet-level Parquet with `ue_id`, `ue_ip`, `flow_id`, `adjusted_timestamp` columns prepended to the standard 16 columns |
| `simulated_traffic.meta.json` | Session provenance — records which source files were used for each flow. **Can be used as input for deterministic replay** (see below) |

### Deterministic Replay

The output `meta.json` has the same structure as the input config, with an additional `file_path` array in each flow recording exactly which session files were selected. Feeding the `meta.json` back as input produces **byte-identical** output:

```bash
# 1. Normal run (random session selection)
python simulate_ue.py scenario_config.json
# → simulated_traffic.parquet + simulated_traffic.meta.json

# 2. Replay (deterministic — uses the same sessions in the same order)
python simulate_ue.py simulated_traffic.meta.json
# → identical simulated_traffic.parquet
```

When `file_path` is present, the simulator skips random selection and uses those exact files in order. Flows in replay mode are tagged with `[REPLAY]` in the console output.

---

## 6. UPF-EES Notification Packager

Converts simulated traffic into 3GPP UPF Event Exposure Service (EES) notification format.

```bash
python upf_ees_packager.py simulated_traffic.parquet --interval 30 --base-time 2026-01-14T12:00:00Z
```

| Argument | Default | Description |
| :--- | :--- | :--- |
| `csv_file` | *(required)* | Path to `simulated_traffic.parquet` |
| `--interval` | `30` | Reporting period in seconds |
| `--base-time` | current UTC | Simulation start time (ISO 8601) |
| `--output-dir` | `upf_ees_output/` | Output directory for JSON files |

### Output Format

Each notification window produces a JSON file containing per-UE measurements:

```json
{
  "notificationItems": [
    {
      "eventType": "USER_DATA_USAGE_MEASURES",
      "timeStamp": "2026-01-14T12:00:30Z",
      "ueIpv4Addr": "10.10.0.1",
      "startTime": "2026-01-14T12:00:00Z",
      "userDataUsageMeasurements": [
        {
          "volumeMeasurement": {
            "totalVolume": 1572864,
            "ulVolume": 524288,
            "dlVolume": 1048576,
            "totalNbOfPackets": 2000,
            "ulNbOfPackets": 800,
            "dlNbOfPackets": 1200
          },
          "throughputMeasurement": {
            "ulThroughput": "139810 bps",
            "dlThroughput": "279620 bps",
            "ulPacketThroughput": "26.67 pps",
            "dlPacketThroughput": "40.00 pps"
          }
        }
      ]
    }
  ],
  "correlationId": "sim_001"
}
```

### Measurement Calculations

| Metric | Formula |
| :--- | :--- |
| `ulVolume` / `dlVolume` | Sum of `pkt_len` for UL (`direction=0`) / DL (`direction=1`) packets in window |
| `totalVolume` | `ulVolume + dlVolume` |
| `ulThroughput` | `ulVolume × 8 ÷ interval` (bps) |
| `ulPacketThroughput` | `ulNbOfPackets ÷ interval` (pps) |

---

## 7. Known Limitations

### UTMobileNet TCP Flags

The raw UTMobileNet2021 dataset lacks a full hexadecimal TCP flags column and is missing several flag bits (SYN, ACK, PSH, RST).

- **Current Behavior**: Flags are reconstructed from the available `tcp.flags.fin` and `tcp.flags.ns` columns only.
- **Impact**: UTMobileNet packets will mostly have empty `tcp_flags`, except for those with **FIN (F)** or **NS (N)** set.
- **MIRAGE Data**: Unaffected — contains full TCP flag information (e.g. `SPA`, `A`, `FA`).

### Session Density Variation

Traffic density varies significantly across action types. For example, `videocall` sessions may contain thousands of packets per second, while `chat` sessions average ~1 packet every 3 seconds. This is a faithful reflection of real network behavior and should be considered when configuring flow durations.
