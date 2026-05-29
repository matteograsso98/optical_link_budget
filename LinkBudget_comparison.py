import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from P1622 import mie_attenuation_dB, Cn2_profile, scintillation_sigma_dB

def total_attenuation_dB(lam_um, el_deg, h0_m=5.5, Z_m=20000, vrms=21.0):
    return mie_attenuation_dB(lam_um, hE_km=1.0, el_deg=el_deg) + scintillation_sigma_dB(lam_um, el_deg, h0_m=h0_m, Z_m=Z_m, vrms=vrms)

# ── FREQUENCY / WAVELENGTH GRID ──────────────────────────────────────────────
# Validity range of Annex 1: 150–375 THz  (≈ 0.8–2.0 µm)
freq_THz = np.linspace(150, 375, 600)
lam_um   = 3e2 / freq_THz       # c [µm·THz] = 3×10^8 m/s = 3×10^2 µm·THz

TARGET_FREQ = 193.41             # THz  — C-band, 1550 nm
TARGET_LAM  = 3e2 / TARGET_FREQ # µm

hE  = 1.0   # km  (ground station altitude ASL)
el  = 40.0  # degrees elevation angle

# ── COMPUTE ──────────────────────────────────────────────────────────────────
mie_dB  = np.array([mie_attenuation_dB(l, hE, el) for l in lam_um])
scin_1s = np.array([scintillation_sigma_dB(l, el)  for l in lam_um])
scin_3s = 3 * scin_1s
total_att_dB = total_attenuation_dB(lam_um, el, h0_m=5.5, Z_m=20000, vrms=21.0)

mie_at_target  = mie_attenuation_dB(TARGET_LAM, hE, el)
scin_at_target = scintillation_sigma_dB(TARGET_LAM, el)
#total_at_target = total_attenuation_dB(TARGET_LAM, el, h0_m=5.5, Z_m=20000, vrms=21.0)

print(f"\nScintillation sigma_dBN  (eq.4a)   el=75°  vrms=21 m/s")
table2 = {193.5: 1.25, 282.0: 1.94, 352.9: 2.52, 563.9: 4.35}
for f, l in [(193.5, 1.550), (282.0, 1.064), (352.9, 0.850), (563.9, 0.532)]:
    v = scintillation_sigma_dB(l, 75.0)
    print(f"  {f:6.1f} THz ({l:.3f} µm): {v:.3f} dB  [Table 2: {table2[f]:.2f} dB]")

print(f"\n@ {TARGET_FREQ} THz ({TARGET_LAM:.4f} µm), hE={hE} km, el={el}°")
print(f"  Mie A_S            = {mie_at_target:.4f} dB")
print(f"  Scin σ_dBN  (1σ)   = {scin_at_target:.4f} dB")
print(f"  Scin fade   (3σ)   = {3*scin_at_target:.4f} dB")
#print(f"  Total         = {total_at_target:.4f} dB")

# ── PARAMETERS DEFINITION ─────────────────────────────────────────────────────
lam_um  = 1.550   # µm
hE_km   = 1.0     # km
h0_m   = 5.5     # m
vrms    = 21.0    # m/s
Z_m    = 20000   # m
C0      = 1.7e-14 # m^{-2/3}

# ── TABLE 5 ──────────────────────────────────────────────────────────────────────

theta_E_deg = np.linspace(10, 90, 9)
total_attenuation_table5 = np.array([total_attenuation_dB(lam_um, el, h0_m=h0_m, Z_m=Z_m, vrms=vrms) for el in theta_E_deg])

print("\n| Elevation angle (°) | Total attenuation (dB)|")
for el, att in zip(theta_E_deg, total_attenuation_table5):
    print(f"|         {el:.0f}          |         {att:.4f}        |")
