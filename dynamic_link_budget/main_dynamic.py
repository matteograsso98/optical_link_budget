"""
main_dynamic.py
===============
Point d'entrée — simulation FSO LEO dynamique et tracés.
Tous les paramètres sont lus directement depuis config.yaml (pas de config.py).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from optical_link_budget_paper.link.geometry import slant_distance_km
from dynamic_link_budget.dynamic_link_budget import DynamicLinkBudget
from dynamic_link_budget.plots import plot_snr_and_rate

# ── Chargement de config.yaml ──────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

_gen     = _cfg["general"]
_offline = _cfg["offline"]
_online  = _cfg["online"]
_atm_raw = _online["atmosphere"]
_cloud   = _atm_raw["cloud"]
_rx      = _online["ground_station_terminal"]
_tx      = _online["satellite_terminal"]

# ── Paramètres de simulation ───────────────────────────────────────────────────

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

# ── Paramètres atmosphériques ──────────────────────────────────────────────────

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

# ── Paramètres orbitaux ────────────────────────────────────────────────────────

orb = SimpleNamespace(
    hS_km = _offline["constellation"]["altitude_km"],
)

# ── Paramètres terminaux (émetteur + récepteur) ────────────────────────────────

trm = SimpleNamespace(
    # Récepteur (station sol)
    Dr_m        = _rx["aperture_diameter_m"],
    eta_R       = _rx["optical_efficiency"],
    theta_R_rad = _rx["pointing_error_rad"],
    noise_dBm   = _rx["noise_power_dBm"],
    Pr_dBm      = _rx["target_rx_power_dBm"],
    # Émetteur (satellite)
    P_tx        = _tx["tx_power_W"],
    eta_T       = _tx["optical_efficiency"],
    Theta_T_rad = _tx["beam_divergence_rad"],
    theta_T_rad = _tx["pointing_error_rad"],
    # Lien
    bandwidth_hz = _online["link"]["bandwidth_hz"],
)

# ── Chemin de cache LUT ────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LUT_PATH = DATA_DIR / sim.lut_cache_file.split("/")[-1]

# ── Utilisateurs — paramètres spatiaux (tirés une fois, fixes sur toute la sim) ─

user_hE_km = np.random.uniform(sim.user_alt_min_km, sim.user_alt_max_km, sim.n_users)
user_h0_m  = np.full(sim.n_users, sim.station_height_m)

# ── Initialisation et précalcul de la LUT atmosphérique ───────────────────────

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

# ── Simulation du passage LEO ──────────────────────────────────────────────────

azimuths_users = np.random.uniform(0, 360, sim.n_users)
user_indices   = np.arange(sim.n_users)

all_slant      = []
all_SNR_ideal  = []
all_SNR_real   = []
all_rate_ideal = []
all_rate_real  = []

for ts in range(sim.duration):
    # Élévation per-user indépendante à chaque pas de temps (diversité géographique)
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

# ── Tracé ──────────────────────────────────────────────────────────────────────

plot_snr_and_rate(
    slant_km_all    = np.concatenate(all_slant),
    SNR_ideal_dB    = np.concatenate(all_SNR_ideal),
    SNR_real_dB     = np.concatenate(all_SNR_real),
    rate_ideal_gbps = np.concatenate(all_rate_ideal),
    rate_real_gbps  = np.concatenate(all_rate_real),
    atm_model       = atm.atm_model,
    save_path       = Path(__file__).parent / "plots" / "dynamic_link_budget.png",
)
