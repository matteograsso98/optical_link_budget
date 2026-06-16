"""
main_dynamic.py
===============
Entry point — dynamic FSO LEO simulation and plots.
All parameters are read from config.yaml.
"""

from pathlib import Path

import numpy as np

from config import DEFAULT_SIM, DEFAULT_ATM, DEFAULT_ORBIT, DEFAULT_TERMINAL
from optical_link_budget_paper.link.geometry import slant_distance_km
from dynamic_link_budget.dynamic_link_budget import DynamicLinkBudget
from dynamic_link_budget.plots import plot_snr_and_rate

sim = DEFAULT_SIM
atm = DEFAULT_ATM
orb = DEFAULT_ORBIT
trm = DEFAULT_TERMINAL

# ── LUT cache path (relative to the package root) ─────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LUT_PATH = DATA_DIR / sim.lut_cache_file.split("/")[-1]

# ── Users — spatial parameters (sampled once, held fixed across timesteps) ────

user_hE_km = np.random.uniform(sim.user_alt_min_km, sim.user_alt_max_km, sim.n_users)
user_h0_m  = np.full(sim.n_users, sim.station_height_m)

# ── Initialise and precompute atmospheric LUT ──────────────────────────────────

dlb = DynamicLinkBudget(atm, orb, trm)

elevation_grid = np.arange(
    sim.elev_min_deg,
    sim.elev_max_deg + sim.elev_step_deg,
    sim.elev_step_deg,
    dtype=np.float64,
)

dlb.precompute_lut(
    user_hE_km,
    user_h0_m,
    elevation_grid_deg=elevation_grid,
    lut_cache_path=LUT_PATH,
)

# ── Simulate LEO pass ──────────────────────────────────────────────────────────

azimuths_users = np.random.uniform(0, 360, sim.n_users)
user_indices   = np.arange(sim.n_users)

all_slant      = []
all_SNR_ideal  = []
all_SNR_real   = []
all_rate_ideal = []
all_rate_real  = []

for ts in range(sim.duration):
    # Independent per-user elevation at each timestep, uniform on the LUT range,
    # reproducing natural geographic diversity without explicit orbital mechanics.
    el_ts   = np.random.uniform(sim.elev_min_deg, sim.elev_max_deg - 5, sim.n_users)
    slant_m = np.array([
        slant_distance_km(el_ts[u], user_hE_km[u], orb.hS_km) * 1e3
        for u in range(sim.n_users)
    ])

    res = dlb.compute(ts, slant_m, azimuths_users, el_ts,
                      user_indices=user_indices)

    all_slant.append(res["slant_m"] / 1e3)
    all_SNR_ideal.append(res["SNR_dB"])
    all_SNR_real.append(res["SNR_real_dB"])
    all_rate_ideal.append(res["rate_ideal"] / 1e9)
    all_rate_real.append(res["rate_real"]   / 1e9)

# ── Plot ───────────────────────────────────────────────────────────────────────

plot_snr_and_rate(
    slant_km_all    = np.concatenate(all_slant),
    SNR_ideal_dB    = np.concatenate(all_SNR_ideal),
    SNR_real_dB     = np.concatenate(all_SNR_real),
    rate_ideal_gbps = np.concatenate(all_rate_ideal),
    rate_real_gbps  = np.concatenate(all_rate_real),
    atm_model       = atm.atm_model,
    save_path       = Path(__file__).parent / "plots" / "dynamic_link_budget.png",
)
