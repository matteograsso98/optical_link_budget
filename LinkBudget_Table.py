import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from P1622 import mie_attenuation_dB, Cn2_profile ,scintillation_sigma_dB

def total_attenuation_dB(lam_um, el_deg, h0_m=5.5, Z_m=20000, vrms=21.0):
    return mie_attenuation_dB(lam_um, hE_km=1.0, el_deg=el_deg) + scintillation_sigma_dB(lam_um, el_deg, h0_m=h0_m, Z_m=Z_m, vrms=vrms)

# ── PARAMETERS ──────────────────────────────────────────────

hE  = 1.0   # km  (ground station altitude ASL)
el  = 40.0  # degrees elevation angle
lam_um  = 1.550   # µm
hE_km   = 1.0     # km
h0_m   = 5.5     # m
vrms    = 21.0    # m/s
Z_m    = 20000   # m
C0      = 1.7e-14 # m^{-2/3}

# ── COMPUTE ──────────────────────────────────────────────────────────────────
total_at_dB = total_attenuation_dB(lam_um, el, h0_m=5.5, Z_m=20000, vrms=21.0)

print(f"\nScintillation sigma_dBN  (eq.4a)   el=75°  vrms=21 m/s")
table2 = {193.5: 1.25, 282.0: 1.94, 352.9: 2.52, 563.9: 4.35}
for f, l in [(193.5, 1.550), (282.0, 1.064), (352.9, 0.850), (563.9, 0.532)]:
    v = scintillation_sigma_dB(l, 75.0)
    print(f"  {f:6.1f} THz ({l:.3f} µm): {v:.3f} dB  [Table 2: {table2[f]:.2f} dB]")

# ── TABLE 5 (Link Budget Analysis of FSO Satellite Network) ──────────────────────────────────

theta_E_deg = np.linspace(10, 90, 9)
total_attenuation_table = np.array([total_attenuation_dB(lam_um, el, h0_m=h0_m, Z_m=Z_m, vrms=vrms) for el in theta_E_deg])
mie_attenuation_table = np.array([mie_attenuation_dB(lam_um, hE_km, el) for el in theta_E_deg])
scin_1s_table = np.array([scintillation_sigma_dB(lam_um, el, h0_m=h0_m, Z_m=Z_m, vrms=vrms) for el in theta_E_deg])

print("\n| Elevation angle (°) | Total attenuation (dB)|  Mie attenuation (dB) |   Scintillation (dB)  |")
for el, att, mie, scin in zip(theta_E_deg, total_attenuation_table, mie_attenuation_table, scin_1s_table):
    print(f"|         {el:.0f}          |         {att:.4f}        |         {mie:.4f}        |         {scin:.4f}        |")
