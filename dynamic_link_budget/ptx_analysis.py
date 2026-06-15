"""
ptx_analysis.py
===============
Calcule et trace la puissance d'émission minimale P_tx_min nécessaire
pour obtenir SNR ≥ 0 dB en fonction de l'angle d'élévation.

Formule : SNR = P_tx · ||h||²  ≥ 1  →  P_tx_min = 1 / ||h||²

||h_ideal||²  = G_R · sin²(el) / (noise_W · FSPL)
||h_real||²   = ||h_ideal||²  / A_atm_lin   (pertes atm. incluses)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DEFAULT_ATM, DEFAULT_ORBIT, DEFAULT_TERMINAL
from optical_link_budget_paper.link.budget import receive_gain
from optical_link_budget_paper.link.geometry import slant_distance_km
from optical_link_budget_paper.atmosphere import mie, geometric, scintillation

atm = DEFAULT_ATM
orb = DEFAULT_ORBIT
trm = DEFAULT_TERMINAL

# ── Paramètres ────────────────────────────────────────────────────────────────
lam_m    = atm.lam_um * 1e-6
G_R      = receive_gain(trm.Dr_m, atm.lam_um)
noise_W  = 10 ** (trm.noise_dBm / 10) * 1e-3
hE_km    = atm.hE_km

elevations = np.linspace(10, 85, 500)

# ── Calcul de ||h||² et P_tx_min par élévation ────────────────────────────────
Ptx_ideal_W = np.zeros_like(elevations)
Ptx_real_W  = np.zeros_like(elevations)

for i, el in enumerate(elevations):
    d_m     = slant_distance_km(el, hE_km, orb.hS_km) * 1e3
    FSPL    = (4 * np.pi * d_m / lam_m) ** 2
    sin2_el = np.sin(np.radians(el)) ** 2

    h2_ideal = G_R * sin2_el / (noise_W * FSPL)
    Ptx_ideal_W[i] = 1.0 / h2_ideal

    A_mie  = mie.attenuation_dB(atm.lam_um, hE_km, el)
    A_geom = geometric.attenuation_dB(el, atm.LW, atm.N, atm.lam_um,
                                       atm.hA_km, hE_km, atm.phi)
    A_scin = scintillation.sigma_dB(atm.lam_um, el, atm.h0_m,
                                     atm.Z_m, atm.vrms, atm.C0)
    A_atm_lin = 10 ** ((A_mie + A_geom + A_scin) / 10)

    h2_real = h2_ideal / A_atm_lin
    Ptx_real_W[i] = 1.0 / h2_real

# Conversion en dBm
Ptx_ideal_dBm = 10 * np.log10(Ptx_ideal_W * 1e3)
Ptx_real_dBm  = 10 * np.log10(Ptx_real_W  * 1e3)

# ── Affichage des valeurs clés ─────────────────────────────────────────────────
for el_ref in [10, 20, 30, 45, 60, 75, 85]:
    idx = np.argmin(np.abs(elevations - el_ref))
    print(f"el={el_ref:2d}° | P_tx_min idéal = {Ptx_ideal_dBm[idx]:6.1f} dBm "
          f"({Ptx_ideal_W[idx]:.1f} W) | réel = {Ptx_real_dBm[idx]:6.1f} dBm "
          f"({Ptx_real_W[idx]:.1f} W)")

# ── Tracé ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("P_tx minimale pour SNR ≥ 0 dB\n"
             f"(hS={orb.hS_km} km, hE={hE_km} km, λ={atm.lam_um} µm, "
             f"Dr={trm.Dr_m*1e3:.0f} mm, bruit={trm.noise_dBm} dBm)",
             fontsize=11, fontweight="bold")

# ── Axe gauche : dBm ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(elevations, Ptx_ideal_dBm, color="steelblue", lw=2, label="Canal idéal (FSPL seul)")
ax.plot(elevations, Ptx_real_dBm,  color="red",       lw=2, label="Canal réel (Mie + Géom + Scin)")
ax.set_xlabel("Élévation [°]")
ax.set_ylabel("P_tx_min [dBm]")
ax.set_title("P_tx_min vs Élévation")
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # slant range décroît quand l'élévation monte

# ── Axe droit : Watts (log) ───────────────────────────────────────────────────
ax = axes[1]
ax.semilogy(elevations, Ptx_ideal_W, color="steelblue", lw=2, label="Canal idéal")
ax.semilogy(elevations, Ptx_real_W,  color="red",       lw=2, label="Canal réel")
ax.set_xlabel("Élévation [°]")
ax.set_ylabel("P_tx_min [W]")
ax.set_title("P_tx_min vs Élévation (Watts, échelle log)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.invert_xaxis()

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), "plots", "ptx_min_vs_elevation.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\n>>> Figure sauvegardée : {save_path}")
plt.show()
