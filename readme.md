# Optical_Link_Budget


Channel model for Free-Space Optical (FSO) links on LEO constellations.
Covers **ground-to-space** static link budgets; space-to-space links share the
same modules (no atmosphere sub-package needed — just `link/`).

Atmospheric models follow **ITU-R P.1621-2** and **ITU-R P.1622-0**.
Link budget formulas follow **Liang et al., arXiv:2204.13177v1**.

---

## Structure

```
fso_channel/
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
python -m fso_channel.main

# Use individual modules in a notebook:
from fso_channel.atmosphere import mie, scintillation
from fso_channel.link import geometry, budget
from fso_channel.config import DEFAULT_ATM, DEFAULT_ORBIT, DEFAULT_TERMINAL

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
| C0 | 1.7 × 10⁻¹⁴ m⁻²/³ | Ground-level Cn² |
| LW | 3.128 × 10⁻⁴ g m⁻³ | Liquid water content |
| N | 0.5 cm⁻³ | Cloud droplet concentration |
| D_R | 80 mm | Receiver telescope diameter |
| η_T, η_R | 0.8 | Optical efficiencies |
| Θ_T | 15 µrad | Full beam divergence |
| θ_T, θ_R | 1 µrad | Pointing errors |
| P_r | −32.5 dBm | Target received power |

## What is NOT yet implemented

- **Dynamic link budget** (time-varying elevation, orbital propagation)
- **Rayleigh scattering** (ITU-R P.1621-2 §3.1 — no analytical formula in the standard; see note in `P1621.py`)
- **Space-to-space link** (no atmosphere — straightforward to add using only `link/`)
- **Cloud availability** / outage statistics
