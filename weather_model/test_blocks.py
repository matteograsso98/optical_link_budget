"""Validation of the building blocks against the ITU-R SG3 validation
workbook (CG-3M3J-13-ValEx-Rev8_3_0.xlsx, sheet 'P.840-9 A_Clouds')."""
import numpy as np
from cloud_attenuation import (water_permittivity, rayleigh_kl,
                               mass_extinction_constant, water_refractive_index,
                               CLOUD_TYPES)

# (f GHz, eps', eps'', KL) from the workbook
SHEET = [
    (6.0,  62.833684655600976, 37.57102637584496,  0.0311277818545330),
    (15.0, 27.868407925943668, 36.33344190183775,  0.1901133490778464),
    (30.0, 12.75044038752651,  22.89669606542401,  0.7078539583865608),
    (45.0, 9.0530446523498,    16.167571903382598, 1.4430598865763187),
]

print("=== Double-Debye permittivity vs workbook (T = 0 C) ===")
for f, ep_ref, epp_ref, kl_ref in SHEET:
    ep, epp = water_permittivity(f)
    print(f" f={f:5.1f} GHz  eps'={ep:9.4f} (ref {ep_ref:9.4f}, "
          f"d={100*(ep/ep_ref-1):+.3f}%)   eps''={epp:9.4f} "
          f"(ref {epp_ref:9.4f}, d={100*(epp/epp_ref-1):+.3f}%)")

print("\n=== Rayleigh Kl (P.840-6 style) vs workbook KL (P.840-9) ===")
for f, _, _, kl_ref in SHEET:
    kl = rayleigh_kl(f)
    print(f" f={f:5.1f} GHz  Kl_Rayleigh={kl:8.5f}  KL_sheet={kl_ref:8.5f}  "
          f"ratio sheet/Rayleigh = {kl_ref/kl:.4f}")

print("\n=== Mie mass-extinction constants K [dB/km per g/m^3] ===")
for wl, label in [(299792458.0/40e9, "40 GHz"), (1550e-9, "1550 nm")]:
    m = water_refractive_index(wl)
    print(f" {label}:  m = {m.real:.4f} {m.imag:+.3e}j")
    for ct in CLOUD_TYPES:
        K = mass_extinction_constant(wl, ct)
        print(f"    {ct:13s} K = {K:10.4f}")

print("\n Rayleigh Kl @ 40 GHz =", f"{rayleigh_kl(40.0):.4f}",
      "(Mie values above should be close at RF)")
