"""
main.py — FSO link budget scenarios
=====================================
Runs the link budget for a range of elevation angles and prints
the results tables (equivalent to the original flat scripts).

Run from the repo root:
    python main.py
"""

import numpy as np

from optical_link_budget.config import DEFAULT_ATM, DEFAULT_ORBIT, DEFAULT_TERMINAL
from optical_link_budget.atmosphere import mie, scintillation, geometric
from optical_link_budget.link import geometry, budget

# ── Elevation sweep ───────────────────────────────────────────────────────────

ELEVATIONS_DEG = np.linspace(10, 90, 9)   # 10° → 90° in 9 steps

atm = DEFAULT_ATM
orb = DEFAULT_ORBIT
trm = DEFAULT_TERMINAL


# ── Derived terminal parameters ───────────────────────────────────────────────

G_T   = budget.transmit_gain(trm.Theta_T_rad)
G_R   = budget.receive_gain(trm.Dr_m, atm.lam_um)
L_T   = budget.pointing_loss_dB(G_T, trm.theta_T_rad)
L_R   = budget.pointing_loss_dB(G_R, trm.theta_R_rad)

print(f"G_T = {G_T:.2e}  ({10*np.log10(G_T):.1f} dBi)")
print(f"L_T = {L_T:.4f} dB")
print(f"G_R = {G_R:.2e}  ({10*np.log10(G_R):.1f} dBi)")
print(f"L_R = {L_R:.4f} dB")


# ── Per-elevation quantities ──────────────────────────────────────────────────

def compute_row(el: float) -> dict:
    """Compute all link budget quantities for a single elevation angle."""
    A_mie  = mie.attenuation_dB(atm.lam_um, atm.hE_km, el)
    A_scin = scintillation.sigma_dB(atm.lam_um, el, atm.h0_m, atm.Z_m, atm.vrms, atm.C0)
    A_geom = geometric.attenuation_dB(el, atm.LW, atm.N, atm.lam_um, atm.hA_km, atm.hE_km, atm.phi)

    d_GS  = geometry.slant_distance_km(el, atm.hE_km, orb.hS_km)
    FSPL  = geometry.fspl_dB(atm.lam_um, el, atm.hE_km, orb.hS_km)

    P_T_scin = budget.required_tx_power_dBm(
        trm.Pr_dBm,
        eta_T=trm.eta_T, eta_R=trm.eta_R,
        G_T=G_T, G_R=G_R,
        L_T_dB=L_T, L_R_dB=L_R,
        lam_um=atm.lam_um, el_deg=el,
        hE_km=atm.hE_km, hS_km=orb.hS_km,
        atm_model="mie_scin",
        h0_m=atm.h0_m, Z_m=atm.Z_m, vrms=atm.vrms,
    )
    P_T_geom = budget.required_tx_power_dBm(
        trm.Pr_dBm,
        eta_T=trm.eta_T, eta_R=trm.eta_R,
        G_T=G_T, G_R=G_R,
        L_T_dB=L_T, L_R_dB=L_R,
        lam_um=atm.lam_um, el_deg=el,
        hE_km=atm.hE_km, hS_km=orb.hS_km,
        atm_model="mie_geom",
        LW=atm.LW, N=atm.N, hA_km=atm.hA_km, phi=atm.phi,
    )

    return {
        "el":        el,
        "d_GS":      d_GS,
        "FSPL":      FSPL,
        "A_mie":     A_mie,
        "A_scin":    A_scin,
        "A_geom":    A_geom,
        "A_tot_scin": A_mie + A_scin,
        "A_tot_geom": A_mie + A_geom,
        "P_T_scin":  P_T_scin,
        "P_T_geom":  P_T_geom,
    }


rows = [compute_row(el) for el in ELEVATIONS_DEG]


# ── Table 1: Mie + Scintillation ─────────────────────────────────────────────

print("\n" + "=" * 75)
print("Table 1 — Mie + Scintillation (1σ)  |  ITU-R P.1622-0")
print("=" * 75)
print(f"{'Elev (°)':>9} | {'dGS (km)':>9} | {'FSPL (dB)':>10} | {'Mie (dB)':>9} | {'Scin (dB)':>10} | {'Total (dB)':>10}")
print("-" * 75)
for r in rows:
    print(f"{r['el']:>9.0f} | {r['d_GS']:>9.2f} | {r['FSPL']:>10.4f} | "
          f"{r['A_mie']:>9.4f} | {r['A_scin']:>10.4f} | {r['A_tot_scin']:>10.4f}")


# ── Table 2: Mie + Geometric scattering + P_T ────────────────────────────────

print("\n" + "=" * 87)
print("Table 2 — Mie + Geometric scattering  |  Liang et al.")
print("=" * 87)
print(f"{'Elev (°)':>9} | {'dGS (km)':>9} | {'FSPL (dB)':>10} | {'Mie (dB)':>9} | "
      f"{'Geom (dB)':>10} | {'Total (dB)':>10} | {'P_T (dBm)':>10}")
print("-" * 87)
for r in rows:
    print(f"{r['el']:>9.0f} | {r['d_GS']:>9.2f} | {r['FSPL']:>10.4f} | "
          f"{r['A_mie']:>9.4f} | {r['A_geom']:>10.4f} | {r['A_tot_geom']:>10.4f} | "
          f"{r['P_T_geom']:>10.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
from optical_link_budget.plots import plot_atmospheric_impairments
plot_atmospheric_impairments(atm=atm)
