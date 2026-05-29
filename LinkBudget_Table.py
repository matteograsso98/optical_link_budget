import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from P1622 import mie_attenuation_dB, Cn2_profile ,scintillation_sigma_dB, geometrical_scattering_dB, dGS_km, FSPL_dB, total_attenuation_MieGeom_dB, total_attenuation_MieScin_dB, trasmitted_power_dB

# ── PARAMETERS ──────────────────────────────────────────────


hE      = 1.0       # km  (ground station altitude ASL)
el      = 40.0      # degrees elevation angle
lam_um  = 1.550     # µm
hE_km   = 1.0      # km
hS_km   = 550.0     # km  (LEO satellite altitude)
hA_km   = 20.0      # km
h0_m    = 5.5       # m
vrms    = 21.0      # m/s
Z_m     = 20000     # m
C0      = 1.7e-14   # m^{-2/3}
N       = 0.5
LW      = 3.128*10**-4
phi     = 1.6
Dr      = 80e-3        # m  Receiver telescope diameter
etaT    = 0.8       # Transmitter optical efficiency
etaR    = 0.8       # Receiver optical efficiency
TehtaT_rad = 15E-6 # rad  Full transmitting divergence angle
thetaT_rad = 1E-6  # rad  Transmitter pointing error
thetaR_rad = 1E-6  # rad  Receiver pointing error

Gt = 16/(TehtaT_rad)**2  # Transmitter gain (eq.1)
Lt = 10*np.log10(np.exp(- Gt * (thetaT_rad**2))) #dB  Pointing loss eq.(1) →  negative
print(Gt)
print(Lt)

Gr = (np.pi * Dr / (lam_um*1e-6))**2  # Receiver gain
Lr = 10*np.log10(np.exp(- Gr * (thetaR_rad**2))) #dB  Pointing loss eq.(1) →  negative
print(Gr)
print(Lr)

Pr = -32.5 #dB


# ── TABLE 5 (Link Budget Analysis of FSO Satellite Network) ──────────────────────────────────

theta_E_deg = np.linspace(10, 90, 9)

total_attenuation_scin_table = np.array([total_attenuation_MieScin_dB(lam_um, e, h0_m, Z_m, vrms) for e in theta_E_deg])
total_attenuation_geom_table = np.array([total_attenuation_MieGeom_dB(lam_um, e, LW, N, hA_km, hE_km, phi) for e in theta_E_deg])

mie_attenuation_table   = np.array([mie_attenuation_dB(lam_um, hE_km, e) for e in theta_E_deg])
scin_1s_table           = np.array([scintillation_sigma_dB(lam_um, e, h0_m, Z_m, vrms) for e in theta_E_deg])
geom_scatt_table        = np.array([geometrical_scattering_dB(e, LW, N, lam_um, hA_km, hE_km, phi) for e in theta_E_deg])

dGS_table               = np.array([dGS_km(e, hE_km, hS_km) for e in theta_E_deg])
FSPL_table              = np.array([FSPL_dB(lam_um, hE_km, hS_km, e) for e in theta_E_deg])

Pt_table = np.array([trasmitted_power_dB(Pr, etaR, etaT, Gr, Gt, Lr, Lt, LW, N, hA_km, phi, lam_um, hE_km, hS_km, e) for e in theta_E_deg])



print("\n| Elevation (°) | dGS (km) |  FSPL (dB) | Mie (dB) |  Scin (dB) | Total (dB) |")
for e, att, mie, scin, dgs, fspl in zip(theta_E_deg, total_attenuation_scin_table,
                                         mie_attenuation_table, scin_1s_table,
                                         dGS_table, FSPL_table):
    print(f"| {e:>13.0f} | {dgs:>8.2f} | {fspl:>10.4f} | {mie:>8.4f} | {scin:>10.4f} | {att:>10.4f} |")

print("\n| Elevation (°) | dGS (km) |  FSPL (dB) | Mie (dB) |  Geom (dB) | Total (dB) | Pt (dB) |")
for e, att, mie, geom, dgs, fspl, pt in zip(theta_E_deg, total_attenuation_geom_table,
                                         mie_attenuation_table, geom_scatt_table,
                                         dGS_table, FSPL_table, Pt_table):
    print(f"| {e:>13.0f} | {dgs:>8.2f} | {fspl:>10.4f} | {mie:>8.4f} | {geom:>10.4f} | {att:>10.4f} | {pt:>7.4f} |")