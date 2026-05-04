#!/usr/bin/env python3
"""
generate_nwdaf_paper_drift.py
=============================
Generates synthetic network traffic data following the two-regime AR(1)
mean-reverting process defined in:

    "A Feasibility Study Toward Trustworthy 3GPP NWDAF Closed-Loop Analytics
     via Distributed MTLF with OpenDaisy"

Mathematical Model
------------------
For each regime s ∈ {0, 1} and client i, the traffic volume X_t is:

    X_t = ρ_{s,i} · X_{t-1} + (1 - ρ_{s,i}) · μ_{s,i} + ε_t

where ε_t ~ N(0, σ_{s,i}²).

Regime 0 (Phase 1, steps 1–50,000): Stable traffic, ρ ∈ [0.90, 0.98].
Regime 1 (Phase 2, steps 50,001–60,000): Concept drift, ρ ∈ [0.15, 0.35],
    with additive log-normal bursts at p_burst = 0.05.

Cross-client heterogeneity is induced by sampling per-client base means
from a normal distribution with coefficient of variation CV = 0.24.

Output: nwdaf_paper_drift_simulation.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "nwdaf_paper_drift_simulation.parquet"

SEED = 42                        # Reproducibility
N_CLIENTS = 3                    # Number of distinct clients / gNBs

# ── Client IP addresses ──────────────────────────────────────────────────────
CLIENT_IPS = ["10.10.0.1", "10.10.0.2", "10.10.0.3"]
DST_IP = "8.8.8.8"               # Destination IP for all packet flows
INTERVAL = "5s"                  # 5-second reporting interval
N_PHASE1 = 30_000               # Regime 0 (normal operation) steps
N_PHASE2 = 30_000               # Regime 1 (concept drift) steps
N_TOTAL = N_PHASE1 + N_PHASE2   # 60,000 total timesteps

# ── Global mean traffic volumes (bytes per interval) ──────────────────────────
GLOBAL_MEAN_UL = 1_000           # μ̄ for uplink
GLOBAL_MEAN_DL = 5_000           # μ̄ for downlink

# ── Non-IID heterogeneity ────────────────────────────────────────────────────
CV = 0.24                        # Coefficient of variation for cross-client skew

# ── AR(1) autoregressive coefficient ranges ──────────────────────────────────
RHO_PHASE1_RANGE = (0.90, 0.98)  # Strong temporal correlation in Regime 0
RHO_PHASE2_RANGE = (0.15, 0.35)  # Weakened correlation in Regime 1

# ── Noise scale (σ as a fraction of the client mean) ─────────────────────────
NOISE_FRAC = 0.02                # σ_{s,i} = NOISE_FRAC × μ_{s,i}

# ── Burst injection parameters (Regime 1 only) ──────────────────────────────
P_BURST = 0.05                   # Probability of burst per timestep
BURST_LOGNORM_MU = 0.5           # Log-normal μ parameter
BURST_LOGNORM_SIGMA = 1.0        # Log-normal σ parameter
BURST_SCALE_FACTOR = 5.0         # Multiplicative scaling so bursts are visible

# ── Regime 1 mean shift factor ───────────────────────────────────────────────
# In Regime 1, the base mean shifts to simulate concept drift.
# A factor of 1.5 means 50% increase in base traffic level.
REGIME1_MEAN_SHIFT = 1.5


def sample_client_means(
    global_mean: float,
    n_clients: int,
    cv: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample heterogeneous per-client base traffic means.

    Each client mean is drawn from N(global_mean, (cv * global_mean)^2)
    and clipped to positive values.

    Parameters
    ----------
    global_mean : float
        The grand mean μ̄ for this traffic direction.
    n_clients : int
        Number of clients.
    cv : float
        Coefficient of variation controlling cross-client spread.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray of shape (n_clients,)
        Positive per-client means.
    """
    std = cv * global_mean
    means = rng.normal(loc=global_mean, scale=std, size=n_clients)
    # Clip to ensure all means are strictly positive
    means = np.clip(means, a_min=global_mean * 0.1, a_max=None)
    return means


def generate_ar1_series(
    n_steps: int,
    rho: float,
    mu: float,
    sigma: float,
    x0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a mean-reverting AR(1) time series.

    X_t = ρ · X_{t-1} + (1 - ρ) · μ + ε_t,  where ε_t ~ N(0, σ²)

    Parameters
    ----------
    n_steps : int
        Number of timesteps to generate.
    rho : float
        Autoregressive coefficient.
    mu : float
        Long-run mean level.
    sigma : float
        Standard deviation of the innovation noise.
    x0 : float
        Initial value (X_0).
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray of shape (n_steps,)
        The generated AR(1) series.
    """
    series = np.empty(n_steps, dtype=np.float64)
    noise = rng.normal(0, sigma, size=n_steps)

    series[0] = rho * x0 + (1 - rho) * mu + noise[0]
    for t in range(1, n_steps):
        series[t] = rho * series[t - 1] + (1 - rho) * mu + noise[t]

    return series


def inject_bursts(
    series: np.ndarray,
    base_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add log-normal bursts to the series with probability P_BURST per step.

    Burst magnitude = BURST_SCALE_FACTOR × base_mean × LogNormal(μ, σ).
    This scaling ensures bursts are visually significant relative to the
    baseline volume.

    Parameters
    ----------
    series : np.ndarray
        Input time series (modified in-place and returned).
    base_mean : float
        The client's base traffic mean (used for scaling).
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray
        The series with bursts injected.
    """
    n = len(series)
    burst_mask = rng.random(n) < P_BURST
    n_bursts = burst_mask.sum()

    if n_bursts > 0:
        burst_magnitudes = (
            BURST_SCALE_FACTOR
            * base_mean
            * rng.lognormal(mean=BURST_LOGNORM_MU, sigma=BURST_LOGNORM_SIGMA, size=n_bursts)
        )
        series[burst_mask] += burst_magnitudes

    return series


def generate_client_traffic(
    client_idx: int,
    mu_ul: float,
    mu_dl: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate the full 60,000-step traffic series for a single client.

    Phase 1 (Regime 0): 50,000 steps of stable, highly correlated traffic.
    Phase 2 (Regime 1): 10,000 steps with weakened correlation, shifted mean,
                         and stochastic burst injection.

    Parameters
    ----------
    client_idx : int
        Client index (0-based), used for client_id formatting.
    mu_ul : float
        Client-specific uplink base mean (Regime 0).
    mu_dl : float
        Client-specific downlink base mean (Regime 0).
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [ul_volume, dl_volume] for this client.
    """
    # ── Sample per-client AR(1) coefficients ──────────────────────────────
    rho_phase1 = rng.uniform(*RHO_PHASE1_RANGE)
    rho_phase2 = rng.uniform(*RHO_PHASE2_RANGE)

    # ── Noise standard deviations ─────────────────────────────────────────
    sigma_ul_p1 = NOISE_FRAC * mu_ul
    sigma_dl_p1 = NOISE_FRAC * mu_dl

    # Regime 1 means (shifted due to concept drift)
    mu_ul_p2 = mu_ul * REGIME1_MEAN_SHIFT
    mu_dl_p2 = mu_dl * REGIME1_MEAN_SHIFT
    sigma_ul_p2 = NOISE_FRAC * mu_ul_p2
    sigma_dl_p2 = NOISE_FRAC * mu_dl_p2

    # ── Phase 1: Regime 0 (Normal Operation) ──────────────────────────────
    ul_p1 = generate_ar1_series(N_PHASE1, rho_phase1, mu_ul, sigma_ul_p1, mu_ul, rng)
    dl_p1 = generate_ar1_series(N_PHASE1, rho_phase1, mu_dl, sigma_dl_p1, mu_dl, rng)

    # ── Phase 2: Regime 1 (Concept Drift) ─────────────────────────────────
    # Initialize Phase 2 from the last value of Phase 1 for continuity
    ul_p2 = generate_ar1_series(N_PHASE2, rho_phase2, mu_ul_p2, sigma_ul_p2, ul_p1[-1], rng)
    dl_p2 = generate_ar1_series(N_PHASE2, rho_phase2, mu_dl_p2, sigma_dl_p2, dl_p1[-1], rng)

    # ── Inject bursts in Regime 1 ─────────────────────────────────────────
    ul_p2 = inject_bursts(ul_p2, mu_ul_p2, rng)
    dl_p2 = inject_bursts(dl_p2, mu_dl_p2, rng)

    # ── Concatenate phases and clamp to non-negative ──────────────────────
    ul_volume = np.clip(np.concatenate([ul_p1, ul_p2]), 0, None)
    dl_volume = np.clip(np.concatenate([dl_p1, dl_p2]), 0, None)

    return pd.DataFrame({
        "ul_volume": ul_volume,
        "dl_volume": dl_volume,
    })


def main():
    rng = np.random.default_rng(SEED)

    # ── Time axis ─────────────────────────────────────────────────────────
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = pd.date_range(
        start=start_time,
        periods=N_TOTAL,
        freq=INTERVAL,
    )

    print("=" * 70)
    print("NWDAF Paper — Two-Regime AR(1) Drift Simulation")
    print(f"  Time range       : {timestamps[0]}  →  {timestamps[-1]}")
    print(f"  Interval         : {INTERVAL}")
    print(f"  Total timesteps  : {N_TOTAL:,}")
    print(f"  Phase 1 (Regime 0): {N_PHASE1:,} steps  (normal operation)")
    print(f"  Phase 2 (Regime 1): {N_PHASE2:,} steps  (concept drift)")
    print(f"  Clients          : {N_CLIENTS}  {CLIENT_IPS}")
    print(f"  Destination IP   : {DST_IP}")
    print(f"  CV (heterogeneity): {CV}")
    print(f"  UL global mean   : {GLOBAL_MEAN_UL}")
    print(f"  DL global mean   : {GLOBAL_MEAN_DL}")
    print("=" * 70)

    # ── Sample heterogeneous per-client base means ────────────────────────
    client_means_ul = sample_client_means(GLOBAL_MEAN_UL, N_CLIENTS, CV, rng)
    client_means_dl = sample_client_means(GLOBAL_MEAN_DL, N_CLIENTS, CV, rng)

    print("\n── Per-Client Base Means (Regime 0) ──")
    print(f"  {'Source IP':<16s}  {'UL Mean':>10s}  {'DL Mean':>10s}")
    print(f"  {'----------':<16s}  {'-------':>10s}  {'-------':>10s}")
    for i in range(N_CLIENTS):
        print(f"  {CLIENT_IPS[i]:<16s}  {client_means_ul[i]:>10.1f}  {client_means_dl[i]:>10.1f}")

    # ── Generate traffic for each client ──────────────────────────────────
    all_frames = []

    for i in range(N_CLIENTS):
        src_ip = CLIENT_IPS[i]
        print(f"\n  Generating traffic for {src_ip} ...")

        client_df = generate_client_traffic(
            client_idx=i,
            mu_ul=client_means_ul[i],
            mu_dl=client_means_dl[i],
            rng=rng,
        )
        client_df.insert(0, "timestamp", timestamps)
        client_df.insert(1, "src_ip", src_ip)
        client_df.insert(2, "dst_ip", DST_IP)
        all_frames.append(client_df)

    # ── Combine all clients ───────────────────────────────────────────────
    df = pd.concat(all_frames, ignore_index=True)

    # Round to 2 decimal places for cleaner output
    df["ul_volume"] = df["ul_volume"].round(2)
    df["dl_volume"] = df["dl_volume"].round(2)

    # ── Save to Parquet ───────────────────────────────────────────────────
    df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    print(f"\n{'=' * 70}")
    print(f"Saved {len(df):,} rows → {OUTPUT_FILE}")
    print(f"  ({N_CLIENTS} clients × {N_TOTAL:,} timesteps = {N_CLIENTS * N_TOTAL:,} rows)")

    # ── Verification: per-regime statistics ───────────────────────────────
    # Each client contributes N_TOTAL rows, so regime boundary per client:
    vol_cols = ["ul_volume", "dl_volume"]

    print(f"\n── Aggregate Statistics (all {N_CLIENTS} clients) ──")

    # Phase 1: first N_PHASE1 rows per client
    phase1_mask = df.groupby("src_ip").cumcount() < N_PHASE1
    phase2_mask = ~phase1_mask

    df_p1 = df.loc[phase1_mask, vol_cols]
    df_p2 = df.loc[phase2_mask, vol_cols]

    print("\n  Regime 0 (Phase 1 — Normal Operation):")
    print(df_p1.describe().to_string())

    print("\n  Regime 1 (Phase 2 — Concept Drift):")
    print(df_p2.describe().to_string())

    # Mean shift ratios
    print("\n── Drift Ratios (Phase 2 mean / Phase 1 mean) ──")
    for c in vol_cols:
        m1 = df_p1[c].mean()
        m2 = df_p2[c].mean()
        s1 = df_p1[c].std()
        s2 = df_p2[c].std()
        print(
            f"  {c:12s}  mean: {m1:>10.1f} → {m2:>10.1f}  "
            f"(×{m2 / m1:.2f})   "
            f"std: {s1:>8.1f} → {s2:>8.1f}  "
            f"(×{s2 / s1:.2f})"
        )

    # Burst count in Phase 2
    # Approximate: values exceeding 3σ above Phase 2 mean
    print("\n── Burst Injection Summary (Phase 2) ──")
    for c in vol_cols:
        p2_vals = df_p2[c]
        threshold = p2_vals.mean() + 3 * p2_vals.std()
        n_extreme = (p2_vals > threshold).sum()
        print(f"  {c:12s}  values > 3σ above mean: {n_extreme:,}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
