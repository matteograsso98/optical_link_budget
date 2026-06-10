"""Benchmark against Lyras et al. (2017) with the paper's exact assumptions.

Data set: ITU-R P.840-6 digital maps (ECMWF ERA-40 based), bilinearly
interpolated at the site coordinates (as prescribed by ITU-R P.1853-1 step
A1-A2) via the `itur` package with version pinned to 6 -- the same database
the paper uses for Figs. 4-6, 9 and 10.  (Paper Figs. 7-8 instead use
parameters fitted to the proprietary FERAS radiosounding data set and are
not reproducible without it.)

Reproduced configurations
-------------------------
Figs. 4-5 : Milan (45.47N, 9.19E, 300 m), elevation 40 deg,
            40 GHz and 1550 nm time-series snapshot.
Fig. 6    : ILWC CCDF, Tenerife and Milan, synthesizer vs ITU-R theoretical
            curve; paper axes (x: 0..0.8 step 0.2, y: 1e-2..1).
Fig. 9    : Athens, vertical link (90 deg), 30 / 90 / 120 GHz attenuation
            CCDF vs ITU-R P.840-6 (station altitude not stated in the
            paper; 0 km used -- the ITU reference curve is altitude-free).
Fig. 10   : single-link optical (1550 nm) CCDF for Tenerife / Hymettus /
            Milan with Table III geometry (ASTRA 19.2E elevations).
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itur.models import itu840

from cloud_attenuation import (CloudAttenuationSynthesizer,
                               ilwc_ccdf_analytic, itu_p840_attenuation,
                               rayleigh_kl)

C0 = 299792458.0
WL_OPT = 1550e-9
YEARS3 = int(3 * 365.25 * 86400 / 10)          # samples at Ts = 10 s

# ----- ITU-R P.840-6 log-normal parameters at the paper's sites ----------
itu840.change_version(6)


def p840_6_params(lat, lon):
    m, s, p = itu840.lognormal_approximation_coefficient(lat, lon)
    return float(m.value), float(s.value), float(p.value) / 100.0


# Table III / IV of the paper
SITES = {
    #  name        lat     lon     alt km  elev deg
    "Tenerife": (28.27, -17.89, 2.40, 37.46),
    "Hymettus": (37.96,  23.82, 1.00, 45.73),
    "Milan":    (45.47,   9.19, 0.30, 36.71),
}
ATHENS = (37.98, 23.73)

print("ITU-R P.840-6 map parameters (bilinear interpolation):")
PARAMS = {}
for name, (la, lo, alt, el) in SITES.items():
    PARAMS[name] = p840_6_params(la, lo)
    m, s, p = PARAMS[name]
    print(f"  {name:9s} m={m:+8.4f}  sigma={s:7.4f}  P_CLW={100*p:7.3f} %")
M_ATH, S_ATH, P_ATH = p840_6_params(*ATHENS)
print(f"  {'Athens':9s} m={M_ATH:+8.4f}  sigma={S_ATH:7.4f}  "
      f"P_CLW={100*P_ATH:7.3f} %")


def ccdf(samples, x):
    s = np.sort(samples)
    return 1.0 - np.searchsorted(s, x, side="right") / len(s)


# =================================================== Figs. 4-5  (snapshot)
print("\n=== Figs. 4-5: Milan, 40 deg, 40 GHz / 1550 nm (Ts = 1 s) ===")
t0 = time.time()
mM, sM, pM = PARAMS["Milan"]
syn = CloudAttenuationSynthesizer(mM, sM, pM, 40.0, 0.30,
                                  [C0 / 40e9, WL_OPT], ts=1.0, seed=11)
N_TOT, N_SNAP = 48 * 3600, 10_000
A_all, L_all = syn.synthesize(N_TOT, return_ilwc=True)
# pick a representative *intermittent* episode like the paper's Figs. 4-5:
# many on/off cloud events, peaks near the typical (p ~ 5-10 %) level
on = A_all[0] > 0
edges = np.flatnonzero(np.diff(on.astype(np.int8)) == 1)
n_ev = np.array([np.sum((edges >= s) & (edges < s + N_SNAP))
                 for s in range(0, N_TOT - N_SNAP, 1000)])
cand = np.flatnonzero(n_ev >= max(3, n_ev.max() - 1))
pk = np.array([A_all[0][c*1000:c*1000+N_SNAP].max() for c in cand])
s0 = int(cand[np.argmax(pk * (pk < np.quantile(A_all[0][on], 0.98)))]) * 1000
A_snap = A_all[:, s0:s0 + N_SNAP]
print(f"  elapsed {time.time()-t0:.0f} s, cloud fraction "
      f"{np.mean(A_all[0] > 0):.2f}, window @ t0 = {s0} s")
print(f"  40 GHz  max {A_snap[0].max():.3f} dB  (paper Fig. 4: ~0.45 dB)")
print(f"  1550 nm max {A_snap[1].max():.1f} dB  (paper Fig. 5: ~210 dB)")
np.savetxt("snapshot_timeseries.csv",
           np.column_stack([np.arange(N_SNAP), A_snap[0], A_snap[1],
                            L_all[s0:s0 + N_SNAP]]),
           delimiter=",", comments="",
           header="t_s,A_40GHz_dB,A_1550nm_dB,ILWC_kg_m2",
           fmt=["%d", "%.6f", "%.4f", "%.6f"])

t = np.arange(N_SNAP)
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
ax[0].plot(t, A_snap[0], lw=0.6, color="tab:blue")
ax[0].set_ylabel("Attenuation (dB)")
ax[0].set_title("Cloud attenuation, 40 GHz, 40\u00b0, Milan "
                "(cf. paper Fig. 4)")
ax[1].plot(t, A_snap[1], lw=0.6, color="tab:red")
ax[1].set_ylabel("Attenuation (dB)")
ax[1].set_xlabel("Time (s)")
ax[1].set_title("Cloud attenuation, 1550 nm, 40\u00b0, Milan "
                "(cf. paper Fig. 5)")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_4_5_snapshot.png", dpi=150)

# ============================== Fig. 10 (and ILWC for Fig. 6) long runs
print("\n=== Fig. 10: optical single links, 3-year runs (Ts = 10 s) ===")
A_opt, L_site = {}, {}
SEEDS = {"Tenerife": 101, "Hymettus": 202, "Milan": 303}
for name, (la, lo, alt, el) in SITES.items():
    t0 = time.time()
    m, s, p = PARAMS[name]
    sy = CloudAttenuationSynthesizer(m, s, p, el, alt, [WL_OPT],
                                     ts=10.0, seed=SEEDS[name])
    A, L0 = sy.synthesize(YEARS3, chunk=947_000, return_ilwc=True)
    A_opt[name], L_site[name] = A[0], L0
    print(f"  {name:9s} elapsed {time.time()-t0:5.0f} s,  "
          f"P(A>0) = {np.mean(A[0] > 0):.3f}")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
styles = {"Tenerife": ("b-", 1.8), "Hymettus": ("r:", 2.2),
          "Milan": ("k-.", 1.8)}
xo = np.linspace(0, 400, 500)
for name in SITES:
    st, lw = styles[name]
    lbl = {"Tenerife": "Tenerife, SP", "Hymettus": "Hymettus, GR",
           "Milan": "Milan, IT"}[name]
    ax.semilogy(xo, ccdf(A_opt[name], xo), st, lw=lw, label=lbl)
ax.set_xlabel("Attenuation (dB)")
ax.set_ylabel("Exceedance Probability")
ax.set_title("Cloud attenuation CCDF, three single optical links, 1550 nm "
             "(cf. paper Fig. 10)")
ax.set_xlim(0, 400)
ax.set_ylim(1e-2, 1)
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig("fig_10_optical_links.png", dpi=150)

# ================================================= Fig. 6 (ILWC CCDF)
print("\n=== Fig. 6: ILWC CCDF, Tenerife & Milan ===")
fig, ax = plt.subplots(figsize=(7.5, 5.5))
xl = np.linspace(1e-4, 0.8, 400)
for name, sim_style, th_style in [("Tenerife", "r:", "g-"),
                                  ("Milan", "k-.", "b--")]:
    m, s, p = PARAMS[name]
    ax.semilogy(xl, ccdf(L_site[name], xl), sim_style, lw=2.2,
                label=f"ILWC Space Time Synthesizer ({name})")
    ax.semilogy(xl, ilwc_ccdf_analytic(xl, m, s, p), th_style, lw=1.8,
                label=f"Theoretical/Database ({name})")
    p08_sim = float(np.mean(L_site[name] > 0.8))
    p08_th = float(ilwc_ccdf_analytic(0.8, m, s, p))
    print(f"  {name:9s} P(L>0.8): sim={p08_sim:.4f}  theory={p08_th:.4f}")
ax.set_xlabel("ILWC (mm)")
ax.set_ylabel("Exceedance Probability")
ax.set_title("ILWC CCDF (cf. paper Fig. 6)")
ax.set_xlim(0, 0.8)
ax.set_xticks(np.arange(0, 0.81, 0.2))
ax.set_ylim(1e-2, 1)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig_6_ilwc_ccdf.png", dpi=150)

# ============================ Fig. 9 (Athens, vertical, 30/90/120 GHz)
print("\n=== Fig. 9: Athens, vertical link, 30/90/120 GHz, 3 years ===")
t0 = time.time()
freqs = [30e9, 90e9, 120e9]
syA = CloudAttenuationSynthesizer(M_ATH, S_ATH, P_ATH, 90.0, 0.0,
                                  [C0 / f for f in freqs], ts=10.0, seed=99)
A9 = syA.synthesize(YEARS3, chunk=947_000)
print(f"  elapsed {time.time()-t0:.0f} s")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
colors = {30e9: ("b-", "bo"), 90e9: ("g:", "gx"), 120e9: ("k--", "r+")}
p_grid = np.logspace(-4, 0, 200)
for i, f in enumerate(freqs):
    ls, mk = colors[f]
    xg = np.linspace(0, max(A9[i].max(), 1e-6), 500)
    ax.semilogy(xg, ccdf(A9[i], xg), ls, lw=1.8,
                label=f"{f/1e9:.0f} GHz")
    a_itu = itu_p840_attenuation(p_grid, rayleigh_kl(f / 1e9),
                                 M_ATH, S_ATH, P_ATH, 90.0)
    ax.semilogy(a_itu, p_grid, mk, ms=5, mfc="none", markevery=7,
                label=f"{f/1e9:.0f} GHz (ITU-R P.840-6)")
    for p in (0.1, 0.01, 0.001):
        a_sim = np.quantile(A9[i], 1 - p)
        a_th = itu_p840_attenuation(p, rayleigh_kl(f / 1e9),
                                    M_ATH, S_ATH, P_ATH, 90.0)
        print(f"  {f/1e9:5.0f} GHz p={p:6.3f}  sim={a_sim:7.3f} dB  "
              f"ITU={a_th:7.3f} dB")
ax.set_xlabel("Attenuation dB")
ax.set_ylabel("Exceedance Probability")
ax.set_title("Cloud attenuation CCDF, vertical link, Athens "
             "(cf. paper Fig. 9)")
ax.set_xlim(0, 20)
ax.set_ylim(1e-4, 1)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig_9_athens_ccdf.png", dpi=150)

print("\nFigures: fig_4_5_snapshot.png, fig_6_ilwc_ccdf.png, "
      "fig_9_athens_ccdf.png, fig_10_optical_links.png, "
      "snapshot_timeseries.csv")
