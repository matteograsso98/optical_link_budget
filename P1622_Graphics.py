import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from P1622 import mie_attenuation_dB, Cn2_profile ,scintillation_sigma_dB

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
total_at_dB = mie_dB + scin_1s

mie_at_target  = mie_attenuation_dB(TARGET_LAM, hE, el)
scin_at_target = scintillation_sigma_dB(TARGET_LAM, el)
total_at_target = mie_at_target + scin_at_target

print(f"\nScintillation sigma_dBN  (eq.4a)   el=75°  vrms=21 m/s")
table2 = {193.5: 1.25, 282.0: 1.94, 352.9: 2.52, 563.9: 4.35}
for f, l in [(193.5, 1.550), (282.0, 1.064), (352.9, 0.850), (563.9, 0.532)]:
    v = scintillation_sigma_dB(l, 75.0)
    print(f"  {f:6.1f} THz ({l:.3f} µm): {v:.3f} dB  [Table 2: {table2[f]:.2f} dB]")

print(f"\n@ {TARGET_FREQ} THz ({TARGET_LAM:.4f} µm), hE={hE} km, el={el}°")
print(f"  Mie A_S            = {mie_at_target:.4f} dB")
print(f"  Scin σ_dBN  (1σ)   = {scin_at_target:.4f} dB")
print(f"  Scin fade   (3σ)   = {3*scin_at_target:.4f} dB")
print(f"  Total         = {total_at_target:.4f} dB")

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
print(f"  Total         = {total_at_target:.4f} dB")


# ── PLOT ─────────────────────────────────────────────────────────────────────
MIE_C  = '#d62728'   # red
SC1_C  = '#1f77b4'   # blue
SC3_C  = '#ff7f0e'   # orange
VL_C   = '#9467bd'   # purple — target frequency line
TOT_C  = '#2ca02c'   # green

fig = plt.figure(figsize=(11, 12), facecolor='white')
gs  = gridspec.GridSpec(3, 1, hspace=0.50, top=0.88, bottom=0.10,
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

# ── MID: Scintillation ────────────────────────────────────────────────────
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

# ── BOTTOM: Total attenuation ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3)

ax3.plot(freq_THz, total_at_dB, color=TOT_C, lw=2.0,
         label='Total $A_{total}$ — P.1622-0 Annex 1, eq.(4a)')
ax3.axvline(TARGET_FREQ, color=VL_C, lw=1.6, ls='--',
            label=f'{TARGET_FREQ} THz  ({TARGET_LAM:.3f} µm)')
ax3.axhline(total_at_target, color=TOT_C, lw=0.8, ls=':', alpha=0.5)
ax3.scatter([TARGET_FREQ], [total_at_target], color=[TOT_C], zorder=5, s=60)
ax3.annotate(f'{total_at_target:.4f} dB',
             xy=(TARGET_FREQ, total_at_target),
             xytext=(TARGET_FREQ + 22, total_at_target + 0.04),
             color=TOT_C, fontsize=9,
             arrowprops=dict(arrowstyle='->', color=TOT_C, lw=1.1))
ax3.set_xlabel('Frequency (THz)', fontsize=10)
ax3.set_ylabel('Total Attenuation  $A_{total}$ (dB)', fontsize=10)
ax3.set_title(f'Total Atmospheric Attenuation — P.1622-0 Annex 1, eq.(4a)\n'
              f'$h_E$ = {hE} km,  $\\theta_E$ = {el}°,  $h_0$ = 5.5 m,  '
              f'$v_{{rms}}$ = 21 m/s,  $C_0$ = 1.7×10⁻¹⁴ m⁻²/³', fontsize=10, pad=6)
ax3.set_xlim(150, 375)
ax3.set_ylim(bottom=0)
ax3.legend(fontsize=9, framealpha=0.9, loc='upper left')
add_wavelength_axis(ax3)


fig.suptitle('ITU-R P.1622-0  |  FSO Earth–Space Link Impairments  |  150–375 THz',
             fontsize=13, color='#111111', fontweight='bold', y=0.97)

plt.savefig('P1622_mie_scintillation.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nPlot saved.")