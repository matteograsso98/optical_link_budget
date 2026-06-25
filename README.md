# Optical Link Budget (Static) 
Channel model for Free-Space Optical (FSO) links on LEO constellations.
Covers **ground-to-space** static link budgets; space-to-space links share the
same modules (no atmosphere sub-package needed — just `link/`).

Atmospheric models follow **ITU-R P.1621-2** and **ITU-R P.1622-0**.
Link budget formulas follow **Liang et al., arXiv:2204.13177v1**.

---

## Structure

```
optical_link_budget/
│
├── atmosphere/            # Pure ITU-R atmospheric physics — no link budget here
│   ├── mie.py             # Mie scattering attenuation         (P.1622-0 §3.1)
│   ├── scintillation.py   # Amplitude scintillation + Cn²      (P.1622-0 eq.4a, P.1621-2 §5.1.1)
│   ├── geometric.py       # Fog / dense-cloud extinction       (Liang eqs.8–10)
│   └── turbulence.py      # r0, θ0, τ0 turbulence parameters  (P.1621-2 §5)
│
├── link/                  # System-level: geometry + budget
│   ├── geometry.py        # Slant distance, FSPL               (Liang eqs.4–5)
│   └── budget.py          # Gains, pointing losses, P_T solver
│
│── tests/ --> here pytest 
├── config.py              # ← Change scenario parameters here
└── main.py                # Run scenarios, print tables
```

## Design principles

| Rule | Why |
|------|-----|
| `atmosphere/` knows nothing about terminals or orbits | Physics stays reusable for any link type |
| `link/` knows nothing about ITU-R internals | Geometry is independent of atmospheric model |
| All parameters in `config.py` | One place to edit a scenario |
| `main.py` only orchestrates | No physics buried in the runner |
| Every function has a docstring with its equation reference | Traceable back to the standard |

## Usage

```python
# Run the default scenario tables:
python -m optical_link_budget.main

# Use individual modules in a notebook:
from optical_link_budget.atmosphere import mie, scintillation
from optical_link_budget.link import geometry, budget
from optical_link_budget.config import DEFAULT_ATM, DEFAULT_ORBIT, DEFAULT_TERMINAL

A_mie = mie.attenuation_dB(1.55, hE_km=1.0, el_deg=40.0)
d_GS  = geometry.slant_distance_km(40.0, hE_km=1.0, hS_km=550.0)
```

## Parameters (defaults in `config.py`)

| Symbol | Value | Description |
|--------|-------|-------------|
| λ | 1.550 µm | Wavelength (C-band, 193.41 THz) |
| h_E | 1.0 km | Ground station altitude |
| h_S | 550.0 km | LEO satellite altitude |
| h_A | 20.0 km | Troposphere top (geometric scattering) |
| v_rms | 21.0 m/s | RMS wind speed (scintillation) |
| C0 | 1.7 × 10⁻¹⁴ m⁻²/³ | Ground-level Cn² — see note below † |
| LW | 3.128 × 10⁻⁴ g m⁻³ | Liquid water content |
| N | 0.5 cm⁻³ | Cloud droplet concentration |
| D_R | 80 mm | Receiver telescope diameter |
| η_T, η_R | 0.8 | Optical efficiencies |
| Θ_T | 15 µrad | Full beam divergence |
| θ_T, θ_R | 1 µrad | Pointing errors |
| P_r | −32.5 dBm | Target received power |

> **† C0 — fixed value rationale.**
> The ground-level refractive-index structure parameter C0 is currently fixed at
> **1.7 × 10⁻¹⁴ m⁻²/³** for all Cn² and scintillation attenuation computations
> (Hufnagel-Valley profile, `scintillation.py`).
> This is the default value specified in **ITU-R P.1621-2** and is used here
> because NOAA GFS does not export Cn² or the surface turbulence quantities
> (e.g. sensible heat flux, temperature structure parameter Ct²) needed to derive
> a spatially varying C0 from NWP data.
> Spatialising C0 from GFS surface fields remains future work — see the section
> below.

## What is NOT yet implemented

- **Spatially varying C0** — ground-level Cn² currently fixed at the ITU-R P.1621-2 default (1.7 × 10⁻¹⁴ m⁻²/³); derivation from GFS surface fields (temperature, pressure, wind, sensible heat flux) via Monin-Obukhov similarity theory is not yet implemented
- **Dynamic link budget** (time-varying elevation, orbital propagation)
- **Rayleigh scattering** (ITU-R P.1621-2 §3.1 — no analytical formula in the standard; see note in `P1621.py`)
- **Space-to-space link** (no atmosphere — straightforward to add using only `link/`)
- **Cloud availability** / outage statistics
- 
# Cloud Attenuation Time-Series Synthesizer (Ka-band → optical)

Python implementation of

> N. K. Lyras, C. I. Kourogiorgas, A. D. Panagopoulos, *"Cloud Attenuation
> Statistics Prediction From Ka-Band to Optical Frequencies: Integrated
> Liquid Water Content Field Synthesizer"*, IEEE Trans. Antennas Propag.,
> 65(1), pp. 319–328, Jan. 2017.

Synthesizes time series of slant-path cloud attenuation A(t, λ) in dB
following the paper's 11-step algorithm (p. 323): stochastic ILWC field
(n-D SDE, spatially + temporally correlated) → SMOC vertical LWC profiles →
cloud-type classification (Table I) → modified-gamma PSD (Table II) →
Mie extinction → slant-path integral.

## Files

| File | Purpose |
|---|---|
| `cloud_attenuation.py` | Library (synthesizer + analytic references) |
| `run_benchmark.py` | Reproduces paper Figs. 4–5, 6, 9, 10 |
| `test_blocks.py` | Unit checks: dielectric model vs ITU workbook, Mie vs Rayleigh |
| `fig_4_5_snapshot.png` | A(t) at 40 GHz & 1550 nm, Milan, 40° (paper Figs. 4–5) |
| `fig_6_ilwc_ccdf.png` | ILWC CCDF, Tenerife & Milan, paper axes (paper Fig. 6) |
| `fig_9_athens_ccdf.png` | 30/90/120 GHz vertical-link CCDF vs ITU-R P.840-6 (paper Fig. 9) |
| `fig_10_optical_links.png` | 1550 nm CCDF, three single links, Table III geometry (paper Fig. 10) |
| `snapshot_timeseries.csv` | 10 000 s example series (t, A_40GHz, A_1550nm, ILWC) |

## Quick use

```python
from itur.models import itu840
from cloud_attenuation import CloudAttenuationSynthesizer

itu840.change_version(6)                       # paper's database (ERA-40)
m, s, p = itu840.lognormal_approximation_coefficient(45.47, 9.19)  # Milan

syn = CloudAttenuationSynthesizer(
    m=float(m.value), sigma=float(s.value), p_clw=float(p.value)/100,
    elevation_deg=40.0, station_alt_km=0.30,
    wavelengths_m=[299792458/40e9, 1550e-9],   # 40 GHz, 1550 nm
    ts=1.0, seed=42)

A = syn.synthesize(3600)   # (n_wavelengths, n_samples) in dB, 1 sample/s
```

Long, memory-bounded runs: `syn.synthesize(N, chunk=500_000)` (filter and
cloud-base state carry across chunks).

## Inputs — exactly as in the paper

* **m, σ, P_CLW**: bilinear interpolation of the **ITU-R P.840-6 digital
  maps** (ECMWF ERA-40 statistics) at the site coordinates, as prescribed by
  ITU-R P.1853-1 steps A1–A2 — via the `itur` package pinned to version 6.
  Values used:

  | Site | lat / lon | m | σ | P_CLW |
  |---|---|---|---|---|
  | Milan, IT | 45.47 N, 9.19 E | −1.4090 | 0.6145 | 30.541 % |
  | Tenerife, SP | 28.27 N, 17.89 W | −4.0497 | 1.6805 | 22.010 % |
  | Hymettus, GR | 37.96 N, 23.82 E | −1.5573 | 0.6786 | 27.509 % |
  | Athens, GR | 37.98 N, 23.73 E | −1.5594 | 0.6804 | 27.825 % |

* **β₁ = 7.17×10⁻⁴ s⁻¹, β₂ = 2.01×10⁻⁵ s⁻¹, γ₁ = 0.349, γ₂ = 0.830** —
  ITU-R P.1853-1 §4.2 (as stated in the paper).
* Spatial correlation ρ(d) = 0.35 e^(−d/7.8) + 0.65 e^(−d/225.3)
  (eq. 7, MODIS), 1 km grid.
* Geometries: Figs. 4–5 → Milan, 40° elevation, 0.30 km altitude.
  Fig. 9 → Athens, vertical link. Fig. 10 → Table III (ASTRA 19.2°E):
  Tenerife 2.40 km / 37.46°, Hymettus 1.00 km / 45.73°, Milan 0.30 km / 36.71°.
* Liquid-water dielectric: Liebe double-Debye at 0 °C (P.840-6 basis);
  validated to 1–2 % against the ITU SG3 validation workbook
  (`CG-3M3J-13-ValEx-Rev8_3_0.xlsx`, sheet *P.840-9 A_Clouds* — now used as
  a unit test only, since those sheets are P.840-**9** test points, a
  different revision and different locations than the paper's inputs).
* Optical refractive index 1.318 − j 1.0×10⁻⁴ at 1550 nm (Hale & Querry);
  results insensitive to k there (Q_ext ≈ 2 regime).

## Documented assumptions where the paper is silent

* Cloud base h₀: drawn from the CloudSat GEV (eq. 17) **once per cloud event
  per grid column**, held for the event; resampled outside [0, 12] km.
* Cloud-type thresholds on dh at Table I midpoints:
  St ≤ 1.05 < Nb ≤ 1.95 < Cu ≤ 2.5 < Cb (km).
* Fig. 9 station altitude not stated in the paper → 0 km used (the ITU
  reference curve is altitude-independent).
* Paper Figs. 7–8 calibrate m, σ, P_CLW on the proprietary FERAS
  radiosounding data set; not reproducible without that data and therefore
  not included.

## Implementation notes (correctness + speed)

1. The n-D SDE (eqs. 1–6) is an Ornstein–Uhlenbeck process advanced with the
   exact one-pole recursion of ITU-R P.1853-1 (ρ = e^(−βTs)); spatial
   correlation enters through the Cholesky factor of C; the same noise
   vector drives fast and slow components.
2. (X¹, X²) initialized from their exact joint stationary distribution
   (Kronecker Cholesky), replacing the 500 000-sample ITU warm-up.
3. For fixed cloud type, the Mie integral collapses to one constant
   K(λ, type) [dB/km per g/m³] because g is linear in LWC (eq. 20):
   computed once; β_ext = K·w.
4. The slant-path integral has closed form per grid column (the SMOC
   profile is L × gamma PDF in h − h₀ → incomplete-gamma differences).
5. dh(L) and cloud type are precomputed lookup tables (c₁, c₂ depend only
   on L; cloud top solves gammapdf = 0.06, the SMOC w̃→0 remark).

Cost: ~10⁵ samples/s (20-column path); a 3-year run at Ts = 10 s ≈ 80–100 s.

## Validation summary (3-year runs, Ts = 10 s)

* **Fig. 6**: synthesizer vs ITU-R theoretical ILWC CCDF coincide;
  Milan P(L > 0.8 mm) = 0.0077 vs 0.0082 analytic (paper: ≈10⁻²);
  Tenerife reaches 10⁻² at ≈0.3 mm, as in the paper.
* **Fig. 9** (Athens, vertical): model on the ITU-R P.840-6 curve;
  e.g. 30 GHz, p = 1 %: 0.517 vs 0.552 dB. The systematic ≈5 % deficit is
  the SMOC profile truncation (w̃ ≤ 0.06·L upper cut), constant across
  frequency; Mie-vs-Rayleigh adds ≈1 % at 120 GHz.
* **Fig. 10**: P(A > 0): Milan 0.38, Hymettus 0.31, Tenerife 0.15 —
  reproducing the paper's ordering and the strong altitude suppression of
  Tenerife (2.4 km).
* **Figs. 4–5**: a representative intermittent 10⁴ s window shows event
  clusters peaking ≈0.2–0.7 dB at 40 GHz and ≈50–320 dB at 1550 nm
  (paper: ≈0.45 dB / ≈210 dB); the optical/RF scaling ratio matches the
  paper's (≈467). Individual sample paths are stochastic — only the
  statistics (event rates/durations from β₁, β₂, P_CLW; levels from m, σ)
  are reproducible, and they are.
* Mie K at 40 GHz = 1.289 vs Rayleigh 1.288 (dB/km)/(g/m³) — 0.1 %.

## Limitations

* First-order statistics benchmarked, as in the paper (Sec. IV).
* ρ(d) valid for separations < 250 km.
* Cloud liquid water only — no rain, gases, scintillation, aerosols
  (the paper's scope).
