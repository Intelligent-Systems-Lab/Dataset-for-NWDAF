#!/usr/bin/env python3
"""
plot_nwdaf_paper_drift_v2.py
=============================
Visualises the NWDAF paper two-regime AR(1) drift simulation dataset
produced by generate_nwdaf_paper_drift_v2.py.

This version uses the 10.100.0.x subnet (clients 10.100.0.1–3).

Generates two chart files:
  1. nwdaf_paper_drift_overview_v2.png  — 2-row overview (UL / DL) with all
     3 clients overlaid on the same axes, plus a regime-switch marker.
  2. nwdaf_paper_drift_clients_v2.png   — 3-row × 2-col grid showing each
     client individually so per-client heterogeneity and burst patterns
     are clearly visible.

Output PNGs are saved next to this script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── File paths ────────────────────────────────────────────────────────────────
PARQUET_PATH = SCRIPT_DIR / "nwdaf_paper_drift_simulation_v2.parquet"
OVERVIEW_PNG = SCRIPT_DIR / "nwdaf_paper_drift_overview_v2.png"
CLIENTS_PNG = SCRIPT_DIR / "nwdaf_paper_drift_clients_v2.png"

# ── Phase configuration (must match the generator) ───────────────────────────
N_PHASE1 = 30_000
N_PHASE2 = 30_000
N_TOTAL = N_PHASE1 + N_PHASE2

# ── Downsample factor for plotting (60k points per client is heavy) ──────────
# Average every DOWNSAMPLE consecutive points to keep the plot responsive.
DOWNSAMPLE = 50

# ── Colour palette (3 distinct colours for 3 clients) ────────────────────────
CLIENT_COLORS = [
    "#3B82F6",  # blue   — 10.100.0.1
    "#F97316",  # orange — 10.100.0.2
    "#10B981",  # emerald — 10.100.0.3
]

# ── Plot styling (dark theme) ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#CBD5E1",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "text.color": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "legend.facecolor": "#1E293B",
    "legend.edgecolor": "#475569",
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.size": 10,
})


def downsample_series(ts, values, factor):
    """
    Downsample a time series by averaging blocks of `factor` consecutive
    points. Returns truncated arrays (tail remainder is dropped).
    """
    n = len(values) // factor * factor
    ts_ds = ts[:n:factor]  # take every factor-th timestamp as representative
    vals_ds = values[:n].reshape(-1, factor).mean(axis=1)
    return ts_ds, vals_ds


def add_regime_line(ax, drift_ts, label="Regime Switch"):
    """Draw a vertical dashed line at the regime-switch timestamp."""
    ax.axvline(
        drift_ts, color="#EF4444", linestyle="--",
        linewidth=1.4, alpha=0.90, zorder=10, label=label,
    )


def plot_overview(df: pd.DataFrame, clients: list):
    """
    Create a 2-row overview figure (UL Volume / DL Volume) with all
    clients overlaid. Each client is drawn as a thin semi-transparent line.
    """
    fig, axes = plt.subplots(2, 1, figsize=(24, 10), sharex=True)
    titles = ["Uplink Volume (bytes)", "Downlink Volume (bytes)"]
    cols = ["ul_volume", "dl_volume"]

    # Determine regime-switch timestamp from the first client
    c0 = df[df["src_ip"] == clients[0]]
    drift_ts = c0["timestamp"].iloc[N_PHASE1]

    for row, (col, title) in enumerate(zip(cols, titles)):
        ax = axes[row]
        for idx, cid in enumerate(clients):
            cdf = df[df["src_ip"] == cid].reset_index(drop=True)
            ts_arr = cdf["timestamp"].values
            val_arr = cdf[col].values

            ts_ds, val_ds = downsample_series(ts_arr, val_arr, DOWNSAMPLE)

            ax.plot(
                ts_ds, val_ds,
                color=CLIENT_COLORS[idx % len(CLIENT_COLORS)],
                linewidth=0.8, alpha=0.75, label=cid, zorder=3,
            )

        add_regime_line(ax, drift_ts)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, pad=8)
        ax.grid(True, linewidth=0.4)
        ax.legend(loc="upper left", ncol=6, fontsize=7)

        # Format y-axis with comma separators
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        ))

    # X-axis formatting
    axes[1].set_xlabel("Time", fontsize=10)
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.suptitle(
        "NWDAF Paper — Two-Regime AR(1) Drift Simulation (All Clients — 10.100.0.x)",
        fontsize=15, fontweight="bold", y=0.98, color="#F8FAFC",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OVERVIEW_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved overview → {OVERVIEW_PNG}")


def plot_per_client(df: pd.DataFrame, clients: list):
    """
    Create an N-row × 2-col grid: one row per client,
    left column = UL volume, right column = DL volume.
    """
    n_clients = len(clients)
    fig, axes = plt.subplots(
        n_clients, 2, figsize=(24, 3 * n_clients),
        sharex=True,
    )

    # Regime-switch timestamp
    c0 = df[df["src_ip"] == clients[0]]
    drift_ts = c0["timestamp"].iloc[N_PHASE1]

    cols = ["ul_volume", "dl_volume"]
    col_labels = ["UL Volume", "DL Volume"]

    for row, cid in enumerate(clients):
        cdf = df[df["src_ip"] == cid].reset_index(drop=True)
        ts_arr = cdf["timestamp"].values
        color = CLIENT_COLORS[row % len(CLIENT_COLORS)]

        for col_idx, (col, clabel) in enumerate(zip(cols, col_labels)):
            ax = axes[row, col_idx]
            val_arr = cdf[col].values
            ts_ds, val_ds = downsample_series(ts_arr, val_arr, DOWNSAMPLE)

            ax.plot(ts_ds, val_ds, color=color, linewidth=0.7, alpha=0.85, zorder=3)
            ax.fill_between(ts_ds, val_ds, alpha=0.15, color=color, zorder=2)
            add_regime_line(ax, drift_ts, label="")
            ax.grid(True, linewidth=0.3)

            # Y-axis comma formatting
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x:,.0f}"
            ))

            # Row label on the left column
            if col_idx == 0:
                ax.set_ylabel(cid, fontsize=9, fontweight="bold")

            # Column title on the top row
            if row == 0:
                ax.set_title(clabel, pad=8)

    # X-axis label on bottom
    axes[-1, 0].set_xlabel("Time", fontsize=10)
    axes[-1, 1].set_xlabel("Time", fontsize=10)
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.suptitle(
        "NWDAF Paper — Per-Client AR(1) Traffic  (UL / DL Volume — 10.100.0.x)",
        fontsize=15, fontweight="bold", y=1.00, color="#F8FAFC",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(CLIENTS_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved per-client → {CLIENTS_PNG}")


def main():
    print("=" * 70)
    print("NWDAF Paper Drift (v2 — 10.100.0.x) — Chart Generator")
    print("=" * 70)

    if not PARQUET_PATH.exists():
        print(f"[ERROR] Parquet not found: {PARQUET_PATH}")
        print("        Run generate_nwdaf_paper_drift_v2.py first.")
        return

    print(f"\nReading {PARQUET_PATH.name} ...")
    df = pd.read_parquet(PARQUET_PATH)
    clients = sorted(df["src_ip"].unique().tolist())
    print(f"  Loaded {len(df):,} rows — {len(clients)} clients")

    # ── 1. Overview chart (all clients overlaid) ──────────────────────────
    print("\nGenerating overview chart ...")
    plot_overview(df, clients)

    # ── 2. Per-client grid chart ──────────────────────────────────────────
    print("Generating per-client chart ...")
    plot_per_client(df, clients)

    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == "__main__":
    main()
