#!/usr/bin/env python3
"""
test_channel_model_new.py — updated to use new code structure:
  - geometry.py for UV, theta/phi, off-nadir, elevation, slant range
  - Antenna class (antenna.py) for steering vector and element pattern
  - Channel class (channel.py) for channel computation
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

#from simulator.core.rrm.temporary.channel_matrix import initialize_channel
from simulator.satviz.backend import config

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml

from simulator.auxiliary.utils.geometry import (
    compute_uv,
    compute_theta_phi,
    compute_eta_from_uv,
    compute_elevation_from_uv,
    compute_slant_range_from_uv,
)
from simulator.auxiliary.scenario_builder.antenna import Antenna
from simulator.auxiliary.scenario_builder.channel_model.channel import Channel

# Set modern plot style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['legend.frameon'] = True


# -----------------------------------------------------------------------------
# CONFIG LOADER
# -----------------------------------------------------------------------------
def load_raw_config(path='simulator/simulation_configuration.yaml'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_antennas(raw: dict) -> tuple[Antenna, Antenna]:
    """
    Build satellite TX and user terminal RX Antenna objects from raw yaml.
    """
    online      = raw['online']
    payload     = online['satellite_payload']
    service_cfg = online['service']['user_link']

    # Satellite TX: tx_antenna params + payload-level eirp/fov fields
    tx_config = {
        **payload['tx_antenna'],
        'eirp_density_dBW_per_mhz': payload['eirp_density_dBW_per_mhz'],
        'array_field_of_view_deg':   payload.get('array_field_of_view_deg'),
    }
    ant_tx = Antenna(tx_config=tx_config, rx_config={}, service_config=service_cfg)

    # User terminal RX: rx params only
    ant_rx = Antenna(tx_config={}, rx_config=online['user_terminal'], service_config=service_cfg)

    return ant_tx, ant_rx

# -----------------------------------------------------------------------------
# HELPERS  (unchanged from original test_channel_model.py)
# -----------------------------------------------------------------------------
def calculate_footprint_radius_km(alt_km, fov_deg):
    R = 6378.0
    eta = np.radians(fov_deg)
    sin_beta = ((R + alt_km) / R) * np.sin(eta)
    if sin_beta > 1:
        sin_beta = 1
    beta = np.arcsin(sin_beta)
    lambda_rad = np.pi - eta - (np.pi - beta)
    return lambda_rad * R


def destination_point_sphere_deg(lat0_deg, lon0_deg, bearing_deg, central_angle_deg):
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    brg  = np.radians(bearing_deg)
    d    = np.radians(central_angle_deg)

    lat = np.arcsin(np.sin(lat0) * np.cos(d) + np.cos(lat0) * np.sin(d) * np.cos(brg))
    lon = lon0 + np.arctan2(
        np.sin(brg) * np.sin(d) * np.cos(lat0),
        np.cos(d) - np.sin(lat0) * np.sin(lat),
    )

    lat_deg = np.degrees(lat)
    lon_deg = (np.degrees(lon) + 540.0) % 360.0 - 180.0
    return lat_deg, lon_deg


def sample_users_uniform_in_footprint(sat_lat_deg, sat_lon_deg, alt_km, fov_deg,
                                       n_users, seed=0):
    rng = np.random.default_rng(seed)
    R = 6378.0
    radius_km   = calculate_footprint_radius_km(alt_km, fov_deg)
    lam_max_rad = radius_km / R
    lam_max_deg = float(np.degrees(lam_max_rad))

    u       = rng.uniform(np.cos(lam_max_rad), 1.0, size=n_users)
    lam_deg = np.degrees(np.arccos(np.clip(u, -1.0, 1.0)))
    brg_deg = rng.uniform(0.0, 360.0, size=n_users)

    lat, lon = destination_point_sphere_deg(sat_lat_deg, sat_lon_deg, brg_deg, lam_deg)
    users = {
        "lat":     lat.astype(float),
        "lon":     lon.astype(float),
        "lam_deg": lam_deg.astype(float),
    }
    return users, lam_max_deg, radius_km


def plot_users_and_footprint(users, sat_lat_deg, sat_lon_deg, lam_max_deg, title):
    bearings = np.linspace(0.0, 360.0, 721)
    lam = np.full_like(bearings, lam_max_deg, dtype=float)
    fp_lat, fp_lon = destination_point_sphere_deg(sat_lat_deg, sat_lon_deg, bearings, lam)
    plt.figure(figsize=(9, 7))
    plt.scatter(users["lon"], users["lat"], s=6, alpha=0.35, label="Users")
    plt.plot(fp_lon, fp_lat, linewidth=2, label="Footprint boundary")
    plt.scatter([sat_lon_deg], [sat_lat_deg], s=90, marker="*", label="Sub-sat point")
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_rate(bw_mhz, snr_linear):
    rate_bps = bw_mhz * 1e6 * np.log2(1 + snr_linear)
    return rate_bps / 1e9


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    sat_lat        = 48.8566   # Paris
    sat_lon        = 2.3522
    sat_alt_km     = 310.0
    fov_half_angle = 50.0
    N              = 2000

    raw    = load_raw_config()
    online = raw['online']

    ant_tx, ant_rx = build_antennas(raw)
    channel        = Channel(online['channel_model'])

    bw_mhz = online['satellite_payload']['bandwidth_mhz']

    users, lam_max_deg, radius_km = sample_users_uniform_in_footprint(
        sat_lat_deg=sat_lat,
        sat_lon_deg=sat_lon,
        alt_km=sat_alt_km,
        fov_deg=fov_half_angle,
        n_users=N,
        seed=0,
    )

    print(f"Footprint surface radius ≈ {radius_km:.2f} km")
    print(f"Max Earth central angle λ_max ≈ {lam_max_deg:.3f} deg")

    plot_users_and_footprint(
        users, sat_lat, sat_lon, lam_max_deg,
        title="Uniform users within satellite footprint (Paris sub-sat point)"
    )

    # Build (K, 3) geodetic array [lat, lon, alt_m] for geometry functions
    sat_alt_m     = sat_alt_km * 1e3
    sat_geodetic  = (sat_lat, sat_lon, sat_alt_m)        # (lat, lon, alt_m)
    user_geodetic = np.column_stack([
        users["lat"], users["lon"], np.zeros(N)           # users at ground level
    ])                                                     # (N, 3)

    # Satellite's geometric nadir half-angle rho = arcsin(RE/(RE+h))
    # This is what the geometry functions need for elevation and slant range —
    # NOT fov_half_angle (which is just the footprint-sampling coverage angle).
    R_E = 6378137.0  # [m], matches geodedic2satellite_u_v
    sat_nadir_deg = np.degrees(np.arcsin(R_E / (R_E + sat_alt_m)))

    # UV coordinates via geometry.py
    uv = compute_uv(user_geodetic, sat_geodetic)          # (N, 2)
    u, v = uv[:, 0], uv[:, 1]

    # Antenna frame angles via geometry.py
    theta, phi = compute_theta_phi(u, v)                  # (N,) each [deg]
    thetaphi   = np.column_stack([theta, phi])            # (N, 2)

    # Off-nadir, elevation, slant range via geometry.py
    off_nadir   = compute_eta_from_uv(uv)                 # (N,) [deg]
    elevation   = compute_elevation_from_uv(uv, sat_nadir_deg)       # (N,) [deg]
    slant_range = compute_slant_range_from_uv(uv, sat_nadir_deg)     # (N,) [m]

    # Channel computation via Channel class
    # Channel.compute() builds the radiation pattern internally via ant_tx

    h_ideal, h_realistic = channel.compute(
        ts_index=0,
        tx=ant_tx,
        rx=ant_rx,
        user_terminals_geodetic=user_geodetic,
        slant_range=slant_range,
        uv=uv,
        thetaphi=thetaphi,
        elevation=elevation,
    )

    H_lb_norm = np.linalg.norm(h_ideal,    ord=2, axis=1)
    H_tx_norm = np.linalg.norm(h_realistic, ord=2, axis=1)

    rate_ideal    = get_rate(bw_mhz, H_lb_norm)
    vals_db_ideal = 20.0 * np.log10(H_lb_norm)

    rate_real     = get_rate(bw_mhz, H_tx_norm)
    vals_db_real  = 20.0 * np.log10(H_tx_norm)

    slant_range_km = slant_range / 1000.0

    # ---------- Plots ----------
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    axs[0, 0].scatter(slant_range_km, vals_db_ideal, c='dodgerblue', s=10, alpha=0.8,  label='Ideal Channel')
    axs[0, 0].scatter(slant_range_km, vals_db_real,  c='red',        s=7,  alpha=0.35, label='Real Channel (LOS; ITU-618 + Shadowing)')
    axs[0, 0].set_title('SNR vs Real Slant Range', fontweight='bold')
    axs[0, 0].set_xlabel('Slant Range [km]', fontweight='bold', fontsize=13)
    axs[0, 0].set_ylabel('SNR [dB]',         fontweight='bold', fontsize=13)
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].scatter(slant_range_km, rate_ideal, c='dodgerblue', s=10, alpha=0.8,  label='Ideal Channel')
    axs[0, 1].scatter(slant_range_km, rate_real,  c='red',        s=7,  alpha=0.35, label='Real Channel (LOS; ITU-618 + Shadowing)')
    axs[0, 1].set_title('Rate vs Real Slant Range')
    axs[0, 1].set_xlabel('Slant Range [km]')
    axs[0, 1].set_ylabel('Shannon Rate [Gbps]')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].hist(vals_db_ideal, bins=100, density=True, color='dodgerblue', alpha=0.5, label='Ideal Channel',                            edgecolor='black')
    axs[1, 0].hist(vals_db_real,  bins=100, density=True, color='red',        alpha=0.5, label='Real Channel (LOS; ITU-618 + Shadowing)', edgecolor='black')
    axs[1, 0].set_title('SNR Distribution')
    axs[1, 0].set_xlabel("SNR [dB]")
    axs[1, 0].set_ylabel("Probability Distribution")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].hist(rate_ideal, bins=100, density=True, color='dodgerblue', alpha=0.5, label='Ideal Channel',                            edgecolor='black')
    axs[1, 1].hist(rate_real,  bins=100, density=True, color='red',        alpha=0.5, label='Real Channel (LOS; ITU-618 + Shadowing)', edgecolor='black')
    axs[1, 1].set_title('Rate Distribution')
    axs[1, 1].set_xlabel('Shannon Rate [Gbps]')
    axs[1, 1].set_xlim(0, np.max(rate_ideal) * 1.1)
    axs[1, 1].set_ylabel('Probability Distribution')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'channel_sensitivity_real_users.png'
    plt.savefig(output_file, dpi=150)
    print(f"Test Complete. Plot saved to {output_file}")


if __name__ == "__main__":
    main()
