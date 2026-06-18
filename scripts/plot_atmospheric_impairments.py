"""
plot_atmospheric_impairments.py
================================
Three-panel atmospheric impairment figure (150–375 THz).

Reproduces the figure from the original P1622_plot.py:
  - Panel 1: Mie scattering attenuation A_S
  - Panel 2: Scintillation fade depth σ_dBN (1σ and 3σ)
  - Panel 3: Total attenuation A_S + σ_dBN

Run from the repo root:
    python scripts/plot_atmospheric_impairments.py
    python scripts/plot_atmospheric_impairments.py --show
    python scripts/plot_atmospheric_impairments.py --out path/to/out.png
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

from channel_model.atmosphere import mie, scintillation


def _load_default_atm():
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    atm_raw = cfg["online"]["atmosphere"]
    cloud   = atm_raw["cloud"]
    return SimpleNamespace(
        lam_um = atm_raw["wavelength_um"],
        hE_km  = cfg["offline"]["reference_station"]["altitude_km"],
        h0_m   = cfg["offline"]["user_deployment"]["station_height_m"],
        hA_km  = atm_raw["troposphere_top_km"],
        Z_m    = float(atm_raw["turbulence_ceiling_m"]),
        vrms   = atm_raw["rms_wind_speed_m_s"],
        C0     = atm_raw["ground_cn2_m_neg23"],
        LW     = cloud["liquid_water_content_g_m3"],
        N      = cloud["droplet_concentration_cm3"],
        phi    = cloud["kim_phi_coefficient"],
    )

_DEFAULT_ATM = _load_default_atm()


# ── Colour palette ────────────────────────────────────────────────────────────
C_MIE    = "#d62728"
C_SC1    = "#1f77b4"
C_SC3    = "#ff7f0e"
C_VLINE  = "#9467bd"
C_TOT    = "#2ca02c"

TARGET_FREQ_THz = 193.41
TARGET_LAM_um   = 3e2 / TARGET_FREQ_THz


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(colors="#333333")
    ax.yaxis.label.set_color("#333333")
    ax.xaxis.label.set_color("#333333")
    ax.title.set_color("#111111")
    ax.grid(True, color="#e0e0e0", lw=0.7)


def _add_wavelength_axis(ax: plt.Axes) -> plt.Axes:
    ax2 = ax.twiny()
    ax2.set_xlim(150, 375)
    wl_ticks = [0.8, 1.0, 1.3, 1.55, 2.0]
    ax2.set_xticks([3e2 / w for w in wl_ticks])
    ax2.set_xticklabels([f"{w} µm" for w in wl_ticks], fontsize=8, color="#555555")
    ax2.tick_params(colors="#555555")
    for sp in ax2.spines.values():
        sp.set_color("#cccccc")
    return ax2


def plot_atmospheric_impairments(
    atm=_DEFAULT_ATM,
    n_freq: int = 600,
    output_path=None,
    show: bool = False,
) -> Path:
    """
    Parameters
    ----------
    atm         : atmosphere config namespace (hE_km, h0_m, Z_m, vrms, C0).
    n_freq      : number of frequency points.
    output_path : output PNG path. Defaults to scripts/plots/atmospheric_impairments.png.
    show        : if True, open an interactive window after saving.

    Returns
    -------
    Path of the saved figure.
    """
    freq_THz = np.linspace(150, 375, n_freq)
    lam_arr  = 3e2 / freq_THz

    el_deg = 40.0
    hE_km  = atm.hE_km

    mie_dB      = np.array([mie.attenuation_dB(l, hE_km, el_deg) for l in lam_arr])
    scin_1s     = np.array([scintillation.sigma_dB(l, el_deg, atm.h0_m, atm.Z_m, atm.vrms, atm.C0)
                            for l in lam_arr])
    scin_3s     = 3.0 * scin_1s
    total_at_dB = mie_dB + scin_1s

    mie_target   = mie.attenuation_dB(TARGET_LAM_um, hE_km, el_deg)
    scin_target  = scintillation.sigma_dB(TARGET_LAM_um, el_deg, atm.h0_m, atm.Z_m, atm.vrms, atm.C0)
    total_target = mie_target + scin_target

    fig = plt.figure(figsize=(11, 12), facecolor="white")
    gs  = gridspec.GridSpec(3, 1, hspace=0.50, top=0.88, bottom=0.10, left=0.10, right=0.95)

    # Panel 1: Mie
    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1)
    ax1.plot(freq_THz, mie_dB, color=C_MIE, lw=2.0,
             label="Mie $A_S$ — P.1622-0 Annex 1, eq.(3)")
    ax1.axvline(TARGET_FREQ_THz, color=C_VLINE, lw=1.6, ls="--",
                label=f"{TARGET_FREQ_THz} THz  ({TARGET_LAM_um:.3f} µm)")
    ax1.axhline(mie_target, color=C_MIE, lw=0.8, ls=":", alpha=0.5)
    ax1.scatter([TARGET_FREQ_THz], [mie_target], color=C_VLINE, zorder=5, s=60)
    ax1.annotate(f"{mie_target:.3f} dB",
                 xy=(TARGET_FREQ_THz, mie_target),
                 xytext=(TARGET_FREQ_THz + 22, mie_target + 0.04),
                 color=C_VLINE, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_VLINE, lw=1.1))
    ax1.set_xlabel("Frequency (THz)", fontsize=10)
    ax1.set_ylabel("Mie Attenuation  $A_S$ (dB)", fontsize=10)
    ax1.set_title(f"Mie Scattering — P.1622-0 Annex 1, §3.1\n"
                  f"$h_E$ = {hE_km} km,  $\\theta_E$ = {el_deg}°",
                  fontsize=10, pad=6)
    ax1.set_xlim(150, 375)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9, framealpha=0.9, loc="upper left")
    _add_wavelength_axis(ax1)

    # Panel 2: Scintillation
    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2)
    ax2.plot(freq_THz, scin_1s, color=C_SC1, lw=2.0,
             label="$\\sigma_{dBN}$ (1$\\sigma$, 68.3% availability)")
    ax2.plot(freq_THz, scin_3s, color=C_SC3, lw=2.0, ls="--",
             label="$3\\sigma_{dBN}$ (99.7% availability)")
    ax2.axvline(TARGET_FREQ_THz, color=C_VLINE, lw=1.6, ls="--",
                label=f"{TARGET_FREQ_THz} THz  ({TARGET_LAM_um:.3f} µm)")
    ax2.axhline(scin_target,     color=C_SC1, lw=0.8, ls=":", alpha=0.5)
    ax2.axhline(3 * scin_target, color=C_SC3, lw=0.8, ls=":", alpha=0.5)
    ax2.scatter([TARGET_FREQ_THz, TARGET_FREQ_THz],
                [scin_target, 3 * scin_target],
                color=[C_SC1, C_SC3], zorder=5, s=60)
    ax2.annotate(f"{scin_target:.2f} dB  (1$\\sigma$)",
                 xy=(TARGET_FREQ_THz, scin_target),
                 xytext=(TARGET_FREQ_THz + 22, scin_target + 0.08),
                 color=C_SC1, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_SC1, lw=1.1))
    ax2.annotate(f"{3 * scin_target:.2f} dB  (3$\\sigma$)",
                 xy=(TARGET_FREQ_THz, 3 * scin_target),
                 xytext=(TARGET_FREQ_THz + 22, 3 * scin_target + 0.08),
                 color=C_SC3, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_SC3, lw=1.1))
    ax2.set_xlabel("Frequency (THz)", fontsize=10)
    ax2.set_ylabel("Scintillation Fade Depth (dB)", fontsize=10)
    ax2.set_title(f"Amplitude Scintillation — P.1622-0 Annex 1, eq.(4a)\n"
                  f"$\\theta_E$ = {el_deg}°,  $h_0$ = {atm.h0_m} m,  "
                  f"$v_{{rms}}$ = {atm.vrms} m/s,  $C_0$ = 1.7×10⁻¹⁴ m⁻²/³",
                  fontsize=10, pad=6)
    ax2.set_xlim(150, 375)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9, framealpha=0.9, loc="upper left")
    _add_wavelength_axis(ax2)

    # Panel 3: Total
    ax3 = fig.add_subplot(gs[2])
    _style_ax(ax3)
    ax3.plot(freq_THz, total_at_dB, color=C_TOT, lw=2.0,
             label="Total $A_{total}$ = $A_S$ + $\\sigma_{dBN}$ (1$\\sigma$)")
    ax3.axvline(TARGET_FREQ_THz, color=C_VLINE, lw=1.6, ls="--",
                label=f"{TARGET_FREQ_THz} THz  ({TARGET_LAM_um:.3f} µm)")
    ax3.axhline(total_target, color=C_TOT, lw=0.8, ls=":", alpha=0.5)
    ax3.scatter([TARGET_FREQ_THz], [total_target], color=C_TOT, zorder=5, s=60)
    ax3.annotate(f"{total_target:.4f} dB",
                 xy=(TARGET_FREQ_THz, total_target),
                 xytext=(TARGET_FREQ_THz + 22, total_target + 0.04),
                 color=C_TOT, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_TOT, lw=1.1))
    ax3.set_xlabel("Frequency (THz)", fontsize=10)
    ax3.set_ylabel("Total Attenuation  $A_{total}$ (dB)", fontsize=10)
    ax3.set_title(f"Total Atmospheric Attenuation — Mie + Scintillation (1σ)\n"
                  f"$h_E$ = {hE_km} km,  $\\theta_E$ = {el_deg}°,  $h_0$ = {atm.h0_m} m,  "
                  f"$v_{{rms}}$ = {atm.vrms} m/s,  $C_0$ = 1.7×10⁻¹⁴ m⁻²/³",
                  fontsize=10, pad=6)
    ax3.set_xlim(150, 375)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=9, framealpha=0.9, loc="upper left")
    _add_wavelength_axis(ax3)

    fig.suptitle(
        "ITU-R P.1622-0  |  FSO Earth–Space Link Impairments  |  150–375 THz",
        fontsize=13, color="#111111", fontweight="bold", y=0.97,
    )

    if output_path is None:
        out_dir = Path(__file__).parent / "plots"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / "atmospheric_impairments.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Plot saved → {output_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FSO atmospheric impairments.")
    parser.add_argument("--show", action="store_true", help="Open interactive window after saving.")
    parser.add_argument("--out",  type=str, default=None, help="Output PNG path.")
    args = parser.parse_args()
    plot_atmospheric_impairments(output_path=args.out, show=args.show)
