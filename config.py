"""
Scenario configuration — loaded from config.yaml
=================================================
All simulation parameters live in config.yaml.
Edit values there; nothing else needs to change.

Module-level constants are populated at import time by reading config.yaml
from the same directory.  Falls back to hard-coded defaults if the file is
missing or PyYAML is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses  (schema — do not put values here, use config.yaml)
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Top-level simulation control parameters."""
    name:               str   = "fso_leo_dynamic_link_budget"
    duration:           int   = 100     # number of timesteps
    timestep:           int   = 1       # seconds per timestep
    n_users:            int   = 50
    user_alt_min_km:    float = 0.0     # MSL altitude sampling range (uniform)
    user_alt_max_km:    float = 1.0
    station_height_m:   float = 5.5     # height above local ground (scintillation)
    lut_cache_file:     str   = "data/lut_fso_spatial.npy"
    elev_min_deg:       float = 10.0
    elev_max_deg:       float = 90.0
    elev_step_deg:      float = 0.5


@dataclass
class AtmosphereConfig:
    """ITU-R P.1621-2 / P.1622-0 atmosphere parameters."""
    # Atmospheric model selection
    atm_model: str   = "mie_geom_scin"  # mie_geom | mie_scin | mie_geom_scin

    # Wavelength
    lam_um:    float = 1.550        # operating wavelength (µm) — C-band, 193.41 THz

    # Geometry
    hE_km:     float = 1.0          # ground station altitude above MSL (km)
    hA_km:     float = 20.0         # troposphere top height for geometric scattering (km)
    h0_m:      float = 5.5          # station height above local ground (m) — scintillation
    Z_m:       float = 20_000.0     # turbulence ceiling (m)

    # Turbulence / scintillation
    vrms:      float = 21.0         # RMS wind speed (m/s) — vg ≈ 2.3 m/s
    C0:        float = 1.7e-14      # ground-level Cn² (m^{-2/3})

    # Geometric scattering (fog / clouds) — Kim model
    LW:        float = 3.128e-4     # liquid water content (g m^{-3})
    N:         float = 0.5          # cloud droplet concentration (cm^{-3})
    phi:       float = 1.6          # Kim model particle-size coefficient


@dataclass
class OrbitConfig:
    """Satellite orbit parameters."""
    hS_km: float = 550.0            # satellite altitude above MSL (km)


@dataclass
class TerminalConfig:
    """Transmit / receive terminal parameters."""
    # Receiver
    Dr_m:        float = 80e-3      # receiver telescope diameter (m)
    eta_R:       float = 0.8        # receiver optical efficiency (0–1)
    theta_R_rad: float = 1e-6       # receiver pointing error (rad)
    noise_dBm:   float = -100.0     # noise power (dBm)
    Pr_dBm:      float = -32.5      # target received power (dBm)

    # Transmitter
    P_tx:        float = 1.0        # transmit power (W)
    eta_T:       float = 0.8        # transmitter optical efficiency (0–1)
    Theta_T_rad: float = 15e-6      # full beam divergence angle (rad)
    theta_T_rad: float = 1e-6       # transmitter pointing error (rad)

    # Link
    bandwidth_hz: float = 10e9      # bandwidth (Hz)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(
    path: str | Path,
) -> tuple[SimulationConfig, AtmosphereConfig, OrbitConfig, TerminalConfig]:
    """
    Parse config.yaml and return typed configuration objects.

    Returns
    -------
    (SimulationConfig, AtmosphereConfig, OrbitConfig, TerminalConfig)
    """
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load config.yaml. "
            "Install with:  pip install pyyaml"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    gen     = raw.get("general", {})
    offline = raw.get("offline",  {})
    online  = raw.get("online",   {})

    # offline sub-sections
    constellation = offline.get("constellation",    {})
    deployment    = offline.get("user_deployment",  {})
    lut_cfg       = offline.get("atmospheric_lut",  {})

    # online sub-sections
    ch_model  = online.get("channel_model",           {})
    atm_raw   = online.get("atmosphere",              {})
    cloud     = atm_raw.get("cloud",                  {})
    rx        = online.get("ground_station_terminal", {})
    tx        = online.get("satellite_terminal",      {})
    lnk       = online.get("link",                    {})

    sim = SimulationConfig(
        name             = gen.get("simulation_name",      "fso_leo_dynamic_link_budget"),
        duration         = gen.get("simulation_duration",  100),
        timestep         = gen.get("simulation_timestep",  1),
        n_users          = deployment.get("total_users",       50),
        user_alt_min_km  = deployment.get("altitude_min_km",   0.0),
        user_alt_max_km  = deployment.get("altitude_max_km",   1.0),
        station_height_m = deployment.get("station_height_m",  5.5),
        lut_cache_file   = lut_cfg.get("cache_file",           "data/lut_fso_spatial.npy"),
        elev_min_deg     = lut_cfg.get("elevation_min_deg",    10.0),
        elev_max_deg     = lut_cfg.get("elevation_max_deg",    90.0),
        elev_step_deg    = lut_cfg.get("elevation_step_deg",   0.5),
    )

    atm = AtmosphereConfig(
        atm_model = ch_model.get("active_model",            "mie_geom_scin"),
        lam_um    = atm_raw.get("wavelength_um",            1.550),
        hE_km     = deployment.get("altitude_min_km",       1.0),  # fallback global
        hA_km     = atm_raw.get("troposphere_top_km",       20.0),
        h0_m      = deployment.get("station_height_m",      5.5),
        Z_m       = atm_raw.get("turbulence_ceiling_m",     20_000.0),
        vrms      = atm_raw.get("rms_wind_speed_m_s",       21.0),
        C0        = atm_raw.get("ground_cn2_m_neg23",       1.7e-14),
        LW        = cloud.get("liquid_water_content_g_m3",  3.128e-4),
        N         = cloud.get("droplet_concentration_cm3",  0.5),
        phi       = cloud.get("kim_phi_coefficient",        1.6),
    )

    orb = OrbitConfig(
        hS_km = constellation.get("altitude_km", 550.0),
    )

    trm = TerminalConfig(
        Dr_m        = rx.get("aperture_diameter_m",  80e-3),
        eta_R       = rx.get("optical_efficiency",   0.8),
        theta_R_rad = rx.get("pointing_error_rad",   1e-6),
        noise_dBm   = rx.get("noise_power_dBm",      -100.0),
        Pr_dBm      = rx.get("target_rx_power_dBm",  -32.5),
        P_tx        = tx.get("tx_power_W",            1.0),
        eta_T       = tx.get("optical_efficiency",   0.8),
        Theta_T_rad = tx.get("beam_divergence_rad",  15e-6),
        theta_T_rad = tx.get("pointing_error_rad",   1e-6),
        bandwidth_hz = lnk.get("bandwidth_hz",       10e9),
    )

    return sim, atm, orb, trm


# ---------------------------------------------------------------------------
# Module-level defaults — loaded from config.yaml at import time
# ---------------------------------------------------------------------------

_CONFIG_YAML = Path(__file__).with_name("config.yaml")

try:
    DEFAULT_SIM, DEFAULT_ATM, DEFAULT_ORBIT, DEFAULT_TERMINAL = load_config(_CONFIG_YAML)
except (FileNotFoundError, ImportError):
    DEFAULT_SIM      = SimulationConfig()
    DEFAULT_ATM      = AtmosphereConfig()
    DEFAULT_ORBIT    = OrbitConfig()
    DEFAULT_TERMINAL = TerminalConfig()
