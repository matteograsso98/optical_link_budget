"""
main.py — FSO LEO dynamic link budget simulation
=================================================
Runs a LEO satellite pass simulation over a population of ground users,
computes per-timestep SNR and Shannon capacity, then plots the results.

All parameters are read from config.yaml. Run from the repo root:
    python main.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from channel_model.link.geometry import slant_distance_km
from channel_model import OpticalChannel

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

_gen     = _cfg["general"]
_offline = _cfg["offline"]
_online  = _cfg["online"]
_atm_raw = _online["atmosphere"]
_cloud   = _atm_raw["cloud"]
_rx      = _online["ground_station_terminal"]
_tx      = _online["satellite_terminal"]

sim = SimpleNamespace(
    name             = _gen["simulation_name"],
    duration         = _gen["simulation_duration"],
    timestep         = _gen["simulation_timestep"],
    n_users          = _offline["user_deployment"]["total_users"],
    user_alt_min_km  = _offline["user_deployment"]["altitude_min_km"],
    user_alt_max_km  = _offline["user_deployment"]["altitude_max_km"],
    station_height_m = _offline["user_deployment"]["station_height_m"],
    lut_cache_file   = _offline["atmospheric_lut"]["cache_file"],
    elev_min_deg     = _offline["atmospheric_lut"]["elevation_min_deg"],
    elev_max_deg     = _offline["atmospheric_lut"]["elevation_max_deg"],
    elev_step_deg    = _offline["atmospheric_lut"]["elevation_step_deg"],
)

atm = SimpleNamespace(
    atm_model = _online["channel_model"]["active_model"],
    lam_um    = _atm_raw["wavelength_um"],
    hA_km     = _atm_raw["troposphere_top_km"],
    Z_m       = float(_atm_raw["turbulence_ceiling_m"]),
    vrms      = _atm_raw["rms_wind_speed_m_s"],
    C0        = _atm_raw["ground_cn2_m_neg23"],
    LW        = _cloud["liquid_water_content_g_m3"],
    N         = _cloud["droplet_concentration_cm3"],
    phi       = _cloud["kim_phi_coefficient"],
)

orb = SimpleNamespace(
    hS_km = _offline["constellation"]["altitude_km"],
)

trm = SimpleNamespace(
    Dr_m            = _rx["aperture_diameter_m"],
    eta_R           = _rx["optical_efficiency"],
    theta_R_rad     = _rx["pointing_error_rad"],
    noise_dBm_night = _rx["noise_power_dBm"]["night"],
    noise_dBm_day   = _rx["noise_power_dBm"]["day"],
    noise_condition = _online["link"]["noise_condition"],
    Pr_dBm          = _rx["target_rx_power_dBm"],
    P_tx            = _tx["tx_power_W"],
    eta_T           = _tx["optical_efficiency"],
    Theta_T_rad     = _tx["beam_divergence_rad"],
    theta_T_rad     = _tx["pointing_error_rad"],
    bandwidth_hz    = _online["link"]["bandwidth_hz"],
)

# ── Users ─────────────────────────────────────────────────────────────────────

user_hE_km = np.random.uniform(sim.user_alt_min_km, sim.user_alt_max_km, sim.n_users)
user_h0_m  = np.full(sim.n_users, sim.station_height_m)

# ── Channel init + atmospheric LUT ───────────────────────────────────────────

channel = OpticalChannel(atm, orb, trm)

elevation_grid = np.arange(
    sim.elev_min_deg,
    sim.elev_max_deg + sim.elev_step_deg,
    sim.elev_step_deg,
    dtype=np.float64,
)

LUT_PATH = Path(__file__).resolve().parent / sim.lut_cache_file
channel.precompute_lut(
    user_hE_km,
    user_h0_m,
    elevation_grid_deg=elevation_grid,
    lut_cache_path=LUT_PATH,
)

# ── Simulation loop ───────────────────────────────────────────────────────────

user_indices = np.arange(sim.n_users)

all_slant      = []
all_SNR_ideal  = []
all_SNR_real   = []
all_rate_ideal = []
all_rate_real  = []

for ts in range(sim.duration):
    el_ts   = np.random.uniform(sim.elev_min_deg, sim.elev_max_deg - 5, sim.n_users)
    slant_m = np.array([
        slant_distance_km(el_ts[u], user_hE_km[u], orb.hS_km) * 1e3
        for u in range(sim.n_users)
    ])

    res = channel.compute(ts, slant_m, el_ts, user_indices=user_indices)

    all_slant.append(res["slant_m"] / 1e3)
    all_SNR_ideal.append(res["SNR_dB"])
    all_SNR_real.append(res["SNR_real_dB"])
    all_rate_ideal.append(res["rate_ideal"] / 1e9)
    all_rate_real.append(res["rate_real"]   / 1e9)

slant_km    = np.concatenate(all_slant)
SNR_ideal   = np.concatenate(all_SNR_ideal)
SNR_real    = np.concatenate(all_SNR_real)
rate_ideal  = np.concatenate(all_rate_ideal)
rate_real   = np.concatenate(all_rate_real)

# ── Plots ─────────────────────────────────────────────────────────────────────

label_map = {
    "mie_geom":      "Mie + Geometric",
    "mie_scin":      "Mie + Scintillation",
    "mie_geom_scin": "Mie + Geometric + Scintillation",
}
atm_label    = label_map.get(atm.atm_model, atm.atm_model)
label_ideal  = "Ideal Channel"
label_real   = f"Real Channel ({atm_label})"
c_ideal      = "steelblue"
c_real       = "red"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"FSO LEO Dynamic Link Budget\nReal Channel : {atm_label} (ITU-R P.1622)",
    fontsize=13, fontweight="bold",
)

ax = axes[0, 0]
ax.scatter(slant_km, SNR_ideal, s=4, alpha=0.5,  color=c_ideal, label=label_ideal)
ax.scatter(slant_km, SNR_real,  s=4, alpha=0.3,  color=c_real,  label=label_real)
ax.set_xlabel("Slant Range [km]")
ax.set_ylabel("SNR [dB]")
ax.set_title("SNR vs Slant Range")
ax.legend(markerscale=3, fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(slant_km, rate_ideal, s=4, alpha=0.5, color=c_ideal, label=label_ideal)
ax.scatter(slant_km, rate_real,  s=4, alpha=0.3, color=c_real,  label=label_real)
ax.set_xlabel("Slant Range [km]")
ax.set_ylabel("Shannon Rate [Gbps]")
ax.set_title("Rate vs Slant Range")
ax.legend(markerscale=3, fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.hist(SNR_ideal, bins=60, alpha=0.6, density=True, color=c_ideal, label=label_ideal)
ax.hist(SNR_real,  bins=60, alpha=0.6, density=True, color=c_real,  label=label_real)
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("Probability Density")
ax.set_title("SNR Distribution")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.hist(rate_ideal, bins=60, alpha=0.6, density=True, color=c_ideal, label=label_ideal)
ax.hist(rate_real,  bins=60, alpha=0.6, density=True, color=c_real,  label=label_real)
ax.set_xlabel("Shannon Rate [Gbps]")
ax.set_ylabel("Probability Density")
ax.set_title("Rate Distribution")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = Path(__file__).resolve().parent / "plots" / "dynamic_link_budget.png"
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_path}")
