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
    "hE_km":        DEFAULT_ATM.hE_km,   # fallback global si LUT absente
    "hS_km":        DEFAULT_ORBIT.hS_km,
    "Dr_m":         DEFAULT_TERMINAL.Dr_m,
    "Theta_T_rad":  DEFAULT_TERMINAL.Theta_T_rad,
    "theta_T_rad":  DEFAULT_TERMINAL.theta_T_rad,
    "theta_R_rad":  DEFAULT_TERMINAL.theta_R_rad,
    "eta_T":        DEFAULT_TERMINAL.eta_T,
    "eta_R":        DEFAULT_TERMINAL.eta_R,
    "Pr_dBm":       DEFAULT_TERMINAL.Pr_dBm,
    "P_tx":         DEFAULT_TERMINAL.P_tx, # Watt
    "noise_dBm":    DEFAULT_TERMINAL.noise_dBm,  # Noise power (dBm) — convenience parameter for SNR calculations
    "bandwidth_hz": DEFAULT_TERMINAL.bandwidth_hz, #4.4e12,
    # Atmosphère FSO — paramètres globaux (Kim / ITU-R P.1622)
    "LW":           DEFAULT_ATM.LW,
    "N_droplets":   DEFAULT_ATM.N,
    "hA_km":        DEFAULT_ATM.hA_km,
    "phi":          DEFAULT_ATM.phi,
    # Scintillation (ITU-R P.1621-2 / P.1622)
    "Z_m":          DEFAULT_ATM.Z_m,
    "vrms":         DEFAULT_ATM.vrms,
    "C0":           DEFAULT_ATM.C0,
    # Modèle atmosphérique : "mie_geom" | "mie_scin" | "mie_geom_scin"
    "atm_model":    "mie_geom_scin",
    # UPA
    "N_x": DEFAULT_TERMINAL.N_x, 
    "N_y": DEFAULT_TERMINAL.N_y,
    "d_x": DEFAULT_TERMINAL.d_x, 
    "d_y": DEFAULT_TERMINAL.d_y,
    
}

# ── Utilisateurs — paramètres spatiaux ────────────────────────────────
# Chaque utilisateur peut avoir une altitude sol et une hauteur de station
# différentes, ce qui donne un profil atmosphérique propre dans la LUT.

N_user = 50

# hE_km : altitude MSL de chaque utilisateur (plaine → montagne)
user_hE_km = np.random.uniform(0.0, 1.0, N_user)

# h0_m : hauteur au-dessus du sol local (paramètre de scintillation)
user_h0_m  = np.full(N_user, DEFAULT_ATM.h0_m)

# ── Initialisation ─────────────────────────────────────────────────────

dlb = DynamicLinkBudget(cfg)
dlb.precompute_lut(
    user_hE_km,
    user_h0_m,
    lut_cache_path="lut_fso_spatial.npy",
)

# ── Simulation passage LEO ─────────────────────────────────────────────

N_ts = 100

elevations     = np.concatenate([
    np.linspace(10, 85, N_ts // 2, endpoint=False),
    np.linspace(85, 10, N_ts // 2),
])
azimuths_users = np.random.uniform(0, 360, N_user)
user_indices   = np.arange(N_user)

# Each user has a fixed angular offset from the reference satellite pass,
# reflecting their different ground positions within the coverage area.
user_el_offset = np.random.uniform(-20, 20, N_user)   # deg

all_slant      = []
all_SNR_ideal  = []
all_SNR_real   = []
all_rate_ideal = []
all_rate_real  = []

for ts in range(N_ts):
    el_ts   = np.clip(elevations[ts] + user_el_offset, 10, 85)
    slant_m = np.array([
        slant_distance_km(el, hE, cfg["hS_km"]) * 1e3
        for el, hE in zip(el_ts, user_hE_km)
    ])

    res = dlb.compute(ts, slant_m, azimuths_users, el_ts,
                      user_indices=user_indices)

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
    atm_model       = cfg["atm_model"],
    save_path       = "dynamic_link_budget/plots/dynamic_link_budget.png",
)
