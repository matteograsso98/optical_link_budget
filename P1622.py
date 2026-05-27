import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── P.1622-0 IMPLEMENTATIONS (Mie, Scintillation) ────────────────────────────────────────────────

def mie_attenuation_dB(lam_um, hE_km, el_deg):
    """
    ITU-R P.1622-0, Annex 1, §3.1 — Mie scattering attenuation.
    Valid: 150–375 THz, hE 0–5 km, el > 45° (accuracy ~0.1 dB).

    Parameters
    ----------
    lam_um : float   wavelength in micrometres
    hE_km  : float   ground station altitude above MSL in km
    el_deg : float   elevation angle in degrees

    Returns
    -------
    AS : float   total path Mie attenuation in dB (positive = loss)
    """
    # Step 1: wavelength-dependent empirical coefficients (eq. 1a–1d)
    a = -0.000545*lam_um**2 + 0.002*lam_um  - 0.0038
    b =  0.00628 *lam_um**2 - 0.0232*lam_um + 0.00439
    c = -0.028   *lam_um**2 + 0.101*lam_um  - 0.18
    d = -0.228   *lam_um**3 + 0.922*lam_um**2 - 1.26*lam_um + 0.719

    # Step 2: extinction ratio from hE to ∞ in Nepers (eq. 2)
    tau = a*hE_km**3 + b*hE_km**2 + c*hE_km + d

    # Step 3: path attenuation along slant (eq. 3)
    AS = 4.3429 * tau / np.sin(np.radians(el_deg))
    return AS


def Cn2_profile(h_m, C0=1.7e-14, vrms=21.0):
    """
    ITU-R P.1621, §5.1.1 — Hufnagel-Valley turbulence structure profile.

    Parameters
    ----------
    h_m  : array   height above ground in metres
    C0   : float   ground-level turbulence constant (m^{-2/3}), default 1.7e-14
    vrms : float   rms wind speed along vertical path (m/s)

    Returns
    -------
    Cn2 : array   turbulence structure parameter (m^{-2/3})
    """
    t1 = 8.148e-56 * vrms**2 * h_m**10 * np.exp(-h_m / 1000)
    t2 = 2.7e-16 * np.exp(-h_m / 1500)
    t3 = C0 * np.exp(-h_m / 100)
    return t1 + t2 + t3


def scintillation_sigma_dB(lam_um, el_deg, h0_m=5.5, Z_m=20000, vrms=21.0):
    """
    ITU-R P.1622-0, Annex 1, eq.(4a)+(4c) — amplitude scintillation.
    Returns 1-sigma log-irradiance fluctuation in dB (Earth-to-space direction).

    Parameters
    ----------
    lam_um : float   wavelength in micrometres
    el_deg : float   elevation angle in degrees
    h0_m   : float   earth station height above ground in metres (default 5.5 m)
    Z_m    : float   effective turbulence height in metres (default 20 000 m)
    vrms   : float   rms wind speed (m/s, default 21)

    Returns
    -------
    sigma_dBN : float   1-sigma scintillation fade depth in dB
    """
    lam_m = lam_um * 1e-6
    k = 2 * np.pi / lam_m

    # Numerical integration of Cn2(h) * h^(5/6) from h0 to Z  (eq. 4a)
    h_arr = np.linspace(h0_m, Z_m, 80000)
    integrand = Cn2_profile(h_arr, vrms=vrms) * h_arr**(5/6)
    integral = np.trapezoid(integrand, h_arr)

    # Variance of log-irradiance (Np^2)  —  eq. (4a)
    sigma2_lnN = 2.253 * k**(7/6) * (1 / np.sin(np.radians(el_deg)))**(11/6) * integral

    # Convert to dB^2  —  eq. (4c)
    sigma2_dBN = (10 / np.log(10))**2 * sigma2_lnN
    return np.sqrt(sigma2_dBN)


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

mie_at_target  = mie_attenuation_dB(TARGET_LAM, hE, el)
scin_at_target = scintillation_sigma_dB(TARGET_LAM, el)

# ── VALIDATION (console) ─────────────────────────────────────────────────────
print("=== VALIDATION vs P.1622-0 Table 2 ===")
print(f"\nMie  (Annex 1, eq.3)   hE={hE} km  el={el}°")
for f, l in [(193.5, 1.550), (282.0, 1.064), (352.9, 0.850), (563.9, 0.532)]:
    print(f"  {f:6.1f} THz ({l:.3f} µm): AS = {mie_attenuation_dB(l, hE, el):.4f} dB")

print(f"\nScintillation sigma_dBN  (eq.4a)   el=75°  vrms=21 m/s")
table2 = {193.5: 1.25, 282.0: 1.94, 352.9: 2.52, 563.9: 4.35}
for f, l in [(193.5, 1.550), (282.0, 1.064), (352.9, 0.850), (563.9, 0.532)]:
    v = scintillation_sigma_dB(l, 75.0)
    print(f"  {f:6.1f} THz ({l:.3f} µm): {v:.3f} dB  [Table 2: {table2[f]:.2f} dB]")

print(f"\n@ {TARGET_FREQ} THz ({TARGET_LAM:.4f} µm), hE={hE} km, el={el}°")
print(f"  Mie A_S            = {mie_at_target:.4f} dB")
print(f"  Scin σ_dBN  (1σ)   = {scin_at_target:.4f} dB")
print(f"  Scin fade   (3σ)   = {3*scin_at_target:.4f} dB")

# ── PLOT ─────────────────────────────────────────────────────────────────────
MIE_C  = '#d62728'   # red
SC1_C  = '#1f77b4'   # blue
SC3_C  = '#ff7f0e'   # orange
VL_C   = '#9467bd'   # purple — target frequency line

fig = plt.figure(figsize=(11, 8), facecolor='white')
gs  = gridspec.GridSpec(2, 1, hspace=0.50, top=0.88, bottom=0.10,
                        left=0.10, right=0.95)

def style_ax(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color('#cccccc')
    ax.tick_params(colors='#333333')
    ax.yaxis.label.set_color('#333333')
    ax.xaxis.label.set_color('#333333')
    ax.title.set_color('#111111')
    ax.grid(True, color='#e0e0e0', lw=0.7)

def add_wavelength_axis(ax):
    """Secondary top x-axis showing wavelength in µm."""
    ax2 = ax.twiny()
    ax2.set_xlim(150, 375)
    wl_ticks = [0.8, 1.0, 1.3, 1.55, 2.0]
    ax2.set_xticks([3e2/w for w in wl_ticks])
    ax2.set_xticklabels([f'{w} µm' for w in wl_ticks], fontsize=8, color='#555555')
    ax2.tick_params(colors='#555555')
    for sp in ax2.spines.values():
        sp.set_color('#cccccc')
    return ax2

# ── TOP: Mie ─────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1)

ax1.plot(freq_THz, mie_dB, color=MIE_C, lw=2.0,
         label='Mie $A_S$ — P.1622-0 Annex 1, eq.(3)')
ax1.axvline(TARGET_FREQ, color=VL_C, lw=1.6, ls='--',
            label=f'{TARGET_FREQ} THz  ({TARGET_LAM:.3f} µm)')
ax1.axhline(mie_at_target, color=MIE_C, lw=0.8, ls=':', alpha=0.5)
ax1.scatter([TARGET_FREQ], [mie_at_target], color=VL_C, zorder=5, s=60)
ax1.annotate(f'{mie_at_target:.3f} dB',
             xy=(TARGET_FREQ, mie_at_target),
             xytext=(TARGET_FREQ + 22, mie_at_target + 0.04),
             color=VL_C, fontsize=9,
             arrowprops=dict(arrowstyle='->', color=VL_C, lw=1.1))
ax1.set_xlabel('Frequency (THz)', fontsize=10)
ax1.set_ylabel('Mie Attenuation  $A_S$ (dB)', fontsize=10)
ax1.set_title(f'Mie Scattering — P.1622-0 Annex 1, §3.1\n'
              f'$h_E$ = {hE} km,  $\\theta_E$ = {el}°', fontsize=10, pad=6)
ax1.set_xlim(150, 375)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=9, framealpha=0.9, loc='upper left')
add_wavelength_axis(ax1)

# ── BOTTOM: Scintillation ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
style_ax(ax2)

ax2.plot(freq_THz, scin_1s, color=SC1_C, lw=2.0,
         label='$\\sigma_{dBN}$ (1$\\sigma$, 68.3% availability)')
ax2.plot(freq_THz, scin_3s, color=SC3_C, lw=2.0, ls='--',
         label='$3\\sigma_{dBN}$ (99.7% availability)')
ax2.axvline(TARGET_FREQ, color=VL_C, lw=1.6, ls='--',
            label=f'{TARGET_FREQ} THz  ({TARGET_LAM:.3f} µm)')
ax2.axhline(scin_at_target,   color=SC1_C, lw=0.8, ls=':', alpha=0.5)
ax2.axhline(3*scin_at_target, color=SC3_C, lw=0.8, ls=':', alpha=0.5)
ax2.scatter([TARGET_FREQ, TARGET_FREQ],
            [scin_at_target, 3*scin_at_target],
            color=[SC1_C, SC3_C], zorder=5, s=60)
ax2.annotate(f'{scin_at_target:.2f} dB  (1$\\sigma$)',
             xy=(TARGET_FREQ, scin_at_target),
             xytext=(TARGET_FREQ + 22, scin_at_target + 0.08),
             color=SC1_C, fontsize=9,
             arrowprops=dict(arrowstyle='->', color=SC1_C, lw=1.1))
ax2.annotate(f'{3*scin_at_target:.2f} dB  (3$\\sigma$)',
             xy=(TARGET_FREQ, 3*scin_at_target),
             xytext=(TARGET_FREQ + 22, 3*scin_at_target + 0.08),
             color=SC3_C, fontsize=9,
             arrowprops=dict(arrowstyle='->', color=SC3_C, lw=1.1))
ax2.set_xlabel('Frequency (THz)', fontsize=10)
ax2.set_ylabel('Scintillation Fade Depth (dB)', fontsize=10)
ax2.set_title(f'Amplitude Scintillation — P.1622-0 Annex 1, eq.(4a)\n'
              f'$\\theta_E$ = {el}°,  $h_0$ = 5.5 m,  '
              f'$v_{{rms}}$ = 21 m/s,  $C_0$ = 1.7×10⁻¹⁴ m⁻²/³', fontsize=10, pad=6)
ax2.set_xlim(150, 375)
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=9, framealpha=0.9, loc='upper left')
add_wavelength_axis(ax2)

fig.suptitle('ITU-R P.1622-0  |  FSO Earth–Space Link Impairments  |  150–375 THz',
             fontsize=13, color='#111111', fontweight='bold', y=0.97)

plt.savefig('/mnt/user-data/outputs/P1622_mie_scintillation.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nPlot saved.")
