"""
main_dynamic.py
===============
Point d'entrée — simulation FSO LEO dynamique et tracé des 4 figures.
"""

import numpy as np
from config import DEFAULT_ATM, DEFAULT_ORBIT, DEFAULT_TERMINAL
from optical_link_budget_paper.link.geometry import slant_distance_km
from dynamic_link_budget.dynamic_link_budget import DynamicLinkBudget
from dynamic_link_budget.plots import plot_snr_and_rate

# ── Configuration ─────────────────────────────────────────────────────

cfg = {
    # Optique FSO
    "lam_um":       DEFAULT_ATM.lam_um,
    "hE_km":        DEFAULT_ATM.hE_km,
    "hS_km":        DEFAULT_ORBIT.hS_km,
    "Dr_m":         DEFAULT_TERMINAL.Dr_m,
    "Theta_T_rad":  DEFAULT_TERMINAL.Theta_T_rad,
    "theta_T_rad":  DEFAULT_TERMINAL.theta_T_rad,
    "theta_R_rad":  DEFAULT_TERMINAL.theta_R_rad,
    "eta_T":        DEFAULT_TERMINAL.eta_T,
    "eta_R":        DEFAULT_TERMINAL.eta_R,
    "Pr_dBm":       DEFAULT_TERMINAL.Pr_dBm,
    # Atmosphère FSO
    "LW":           DEFAULT_ATM.LW,
    "N_droplets":   DEFAULT_ATM.N,
    "hA_km":        DEFAULT_ATM.hA_km,
    "phi":          DEFAULT_ATM.phi,
    # UPA (défini ici, à déplacer dans config.py si souhaité)
    "N_x": 16, "N_y": 16,
    "d_x": 0.5, "d_y": 0.5,
    # Système
    "noise_dBm":        -100.0,
    "bandwidth_hz":     10e9,
    "channel_scenario": "urban",
    "propagation_model": "los",
}

# ── Initialisation ─────────────────────────────────────────────────────

dlb = DynamicLinkBudget(cfg)
dlb.precompute_lut(lut_cache_path="lut_fso.npy")

# ── Simulation passage LEO ─────────────────────────────────────────────

N_ts   = 100    # timesteps
N_user = 50     # utilisateurs simultanés

elevations     = np.concatenate([
    np.linspace(10, 85, N_ts // 2),
    np.linspace(85, 10, N_ts // 2),
])
azimuths_users = np.random.uniform(0, 360, N_user)

all_slant      = []
all_SNR_ideal  = []
all_SNR_real   = []
all_rate_ideal = []
all_rate_real  = []

for ts in range(N_ts):
    el_ts    = np.full(N_user, elevations[ts])
    slant_m  = np.full(
        N_user,
        slant_distance_km(elevations[ts], cfg["hE_km"], cfg["hS_km"]) * 1e3
    )

    res = dlb.compute(ts, slant_m, azimuths_users, el_ts)

    all_slant.append(res["slant_m"] / 1e3)
    all_SNR_ideal.append(res["SNR_dB"])
    all_SNR_real.append(res["SNR_real_dB"])
    all_rate_ideal.append(res["rate_ideal"] / 1e9)
    all_rate_real.append(res["rate_real"]  / 1e9)

# ── Tracé des 4 figures via plots.py ──────────────────────────────────

plot_snr_and_rate(
    slant_km_all    = np.concatenate(all_slant),
    SNR_ideal_dB    = np.concatenate(all_SNR_ideal),
    SNR_real_dB     = np.concatenate(all_SNR_real),
    rate_ideal_gbps = np.concatenate(all_rate_ideal),
    rate_real_gbps  = np.concatenate(all_rate_real),
    save_path       = "dynamic_link_budget/plots/dynamic_link_budget.png",
)