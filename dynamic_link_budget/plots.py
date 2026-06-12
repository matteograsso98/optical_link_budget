"""
plots.py
========
Tracé des 4 figures SNR/Rate vs Slant Range et leurs distributions.
Reproduit les figures de référence pour le cas FSO dynamique.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_snr_and_rate(
    slant_km_all,
    SNR_ideal_dB,
    SNR_real_dB,
    rate_ideal_gbps,
    rate_real_gbps,
    atm_model="mie_geom",
    save_path=None,
):
    """
    Trace les 4 figures :
      - SNR vs Slant Range
      - Rate vs Slant Range
      - Distribution SNR
      - Distribution Rate

    Paramètres
    ----------
    slant_km_all     : array (N,) — distances obliques [km]
    SNR_ideal_dB     : array (N,) — SNR canal idéal [dB]
    SNR_real_dB      : array (N,) — SNR canal réaliste [dB]
    rate_ideal_gbps  : array (N,) — débit Shannon idéal [Gbps]
    rate_real_gbps   : array (N,) — débit Shannon réaliste [Gbps]
    save_path        : str optionnel — chemin pour sauvegarder la figure
    """
    label_map = {
        "mie_geom":      "Mie + Geometric",
        "mie_scin":      "Mie + Scintillation",
        "mie_geom_scin": "Mie + Geometric + Scintillation",
    }
    atm_label = label_map.get(atm_model, atm_model)

    label_ideal = "Ideal Channel"
    label_real  = f"Real Channel ({atm_label})"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"FSO LEO Dynamic Link Budget\nReal Channel : {atm_label} (ITU-R P.1622)",
        fontsize=13, fontweight="bold"
    )
    c_ideal     = "steelblue"
    c_real      = "red"

    # ── Figure 1 : SNR vs Slant Range ─────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(slant_km_all, SNR_ideal_dB, s=4, alpha=0.5,
               color=c_ideal, label=label_ideal)
    ax.scatter(slant_km_all, SNR_real_dB,  s=4, alpha=0.3,
               color=c_real,  label=label_real)
    ax.set_xlabel("Real Slant Range [km]")
    ax.set_ylabel("SNR [dB]")
    ax.set_title("SNR vs Real Slant Range")
    #ax.set_xlim(300, 500)
    ax.legend(markerscale=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Figure 2 : Rate vs Slant Range ────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(slant_km_all, rate_ideal_gbps, s=4, alpha=0.5,
               color=c_ideal, label=label_ideal)
    ax.scatter(slant_km_all, rate_real_gbps,  s=4, alpha=0.3,
               color=c_real,  label=label_real)
    ax.set_xlabel("Real Slant Range [km]")
    ax.set_ylabel("Shannon Rate [Gbps]")
    ax.set_title("Rate vs Real Slant Range")
    #ax.set_xlim(300, 500)
    ax.legend(markerscale=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Figure 3 : Distribution SNR ───────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(SNR_ideal_dB, bins=60, alpha=0.6, density=True,
            color=c_ideal, label=label_ideal)
    ax.hist(SNR_real_dB,  bins=60, alpha=0.6, density=True,
            color=c_real,  label=label_real)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Probability Distribution")
    ax.set_title("SNR Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Figure 4 : Distribution Rate ──────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(rate_ideal_gbps, bins=60, alpha=0.6, density=True,
            color=c_ideal, label=label_ideal)
    ax.hist(rate_real_gbps,  bins=60, alpha=0.6, density=True,
            color=c_real,  label=label_real)
    ax.set_xlabel("Shannon Rate [Gbps]")
    ax.set_ylabel("Probability Distribution")
    ax.set_title("Rate Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f">>> Figures sauvegardées : {save_path}")

    plt.show()