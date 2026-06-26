"""
Global FSO attenuation map generator.

Fetches GFS meteorological fields via Herbie, computes per-pixel atmospheric
attenuation (Mie + geometric scattering + scintillation) and renders:
  - Transparent PNG overlays for the Cesium 3-D globe.
  - A diagnostic matplotlib figure with lat/lon axes and colourbars.

All link parameters are read from ../config.yaml (cesium_map + online.atmosphere
+ offline.user_deployment sections).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))               # cesium/ → load_gfs_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # project root → channel_model

import warnings
import yaml
import numpy as np
import matplotlib.pyplot as plt

from load_gfs_data import GFSData
from channel_model.atmosphere import mie, geometric, scintillation

# ── Output directory for all map PNGs ────────────────────────────────────────
MAPS_DIR = os.path.join(os.path.dirname(__file__), 'maps')
os.makedirs(MAPS_DIR, exist_ok=True)

def _map_path(filename):
    """Return the full path for a map file inside maps/."""
    return os.path.join(MAPS_DIR, filename)

# ── Load configuration ────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_atm     = _cfg['online']['atmosphere']
_map_cfg = _cfg['cesium_map']
_deploy  = _cfg['offline']['user_deployment']

LAM_UM  = _atm['wavelength_um']               # wavelength [µm]
EL_DEG  = _map_cfg['elevation_deg']           # elevation angle [°]
HE_KM   = _map_cfg['ground_station_altitude_km']  # ground station altitude [km]
HA_KM   = _atm['troposphere_top_km']          # troposphere top [km]
Z_M     = float(_atm['turbulence_ceiling_m']) # scintillation ceiling [m]
C0      = _atm['ground_cn2_m_neg23']          # ground-level Cn² [m^{-2/3}]
H0_M    = _deploy['station_height_m']         # station height above ground [m]


# ── Borderless transparent PNG for Cesium overlays ───────────────────────────

def save_map(data, filename, cmap, vmin=None, vmax=None, dpi=300):
    """Render data as a borderless, transparent global PNG."""
    fig = plt.figure(figsize=(12, 6), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, extent=[-180, 180, -90, 90],
              origin="upper", vmin=vmin, vmax=vmax)
    fig.savefig(filename, transparent=True)
    plt.close(fig)
    print(f"Saved {filename}")


# ── Diagnostic matplotlib figure ─────────────────────────────────────────────

def save_attenuation_plot(A_total, A_geom, A_scin, A_mie_scalar, lat, lon, filename):
    """
    3-panel figure: total attenuation, geometric component, scintillation component.
    Mie value (scalar, identical everywhere) is shown in the figure title.
    """
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    panels = [
        (A_total, "Total attenuation (dB)",    "Reds",   0, 15),
        (A_geom,  "Geometric scattering (dB)", "Blues",  0, 10),
        (A_scin,  "Scintillation σ (dB)",      "Greens", 0,  3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, extent=extent, origin="upper",
                       aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")
    fig.suptitle(
        f"FSO attenuation — λ={LAM_UM} µm, el={EL_DEG}°, "
        f"hE={HE_KM} km | Mie={A_mie_scalar:.3f} dB (uniform)",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching NOAA GFS data...")
    gfs = GFSData(model="gfs", product="pgrb2.0p25", fxx=0)

    # —— Existing overlay maps ────────────────────────────────────────────────
    temp_c = gfs.field("TMP:2 m above ground") - 273.15
    save_map(temp_c, _map_path("temperature_map.png"), "turbo")

    ws_maps = {}
    for level in ["10 m", "50 m", "100 m"]:
        ws = gfs.wind_speed(level)
        ws_maps[level] = ws
        save_map(ws, _map_path(f"wind_speed_{level.replace(' ', '')}_map.png"),
                 "RdYlBu_r", vmin=0, vmax=30)

    ws_10m = ws_maps["10 m"]

    # —— Attenuation maps ─────────────────────────────────────────────────────
    print("Computing attenuation maps...")

    # GFS surface visibility [m] → km
    V_km = gfs.field("VIS:surface") / 1e3
    n_unreliable = int(np.sum(V_km > 24))
    if n_unreliable > 0:
        warnings.warn(
            f"{n_unreliable} pixels have V > 24 km: geometric attenuation values "
            "are unreliable above the GFS visibility cap (Kim & Kruse, 2001).",
            UserWarning, stacklevel=2,
        )
    save_map(V_km, _map_path("visibility_map.png"), "YlGn", vmin=0, vmax=24)

    # GFS planetary boundary layer height [m AGL] → km
    # Clear-sky pixels return 0 → fall back to full troposphere height
    hpbl_m = gfs.field("HPBL:surface")
    hpbl_km = hpbl_m / 1e3
    hA_eff = np.where(hpbl_km > 0, hpbl_km, HA_KM)
    hA_eff = np.clip(hA_eff, 0.1, HA_KM)
    save_map(hA_eff, _map_path("hpbl_map.png"), "Blues_r", vmin=0, vmax=HA_KM)

    # Mie: scalar (fixed wavelength + sea-level station + elevation)
    A_mie = mie.attenuation_dB(LAM_UM, HE_KM, EL_DEG)

    # Geometric: vectorised, φ derived per-pixel from V via Kim model (Kim & Kruse, 2001)
    A_geom = geometric.attenuation_dB(EL_DEG, LAM_UM, hA_eff, HE_KM, V_km=V_km)

    # Scintillation: array path triggered by keyword vg
    A_scin = scintillation.sigma_dB(LAM_UM, EL_DEG, H0_M, Z_M, C0=C0, vg=ws_10m)

    A_total = A_mie + A_geom + A_scin

    # Cesium overlay (borderless transparent PNG)
    save_map(A_total, _map_path("attenuation_map.png"), "Reds", vmin=0, vmax=15)

    # Diagnostic figure with axes and colourbars
    save_attenuation_plot(
        A_total, A_geom, A_scin, A_mie,
        gfs.lat, gfs.lon, _map_path("attenuation_plot.png"),
    )


if __name__ == "__main__":
    main()
