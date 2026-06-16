"""
Amplitude scintillation — ITU-R P.1622-0, Annex 1, eq. (4a) and (4c)
======================================================================
Hufnagel-Valley turbulence profile from ITU-R P.1621-2, eq. (6).

Reference: ITU-R P.1622-0 (scintillation), ITU-R P.1621-2 §5.1.1 (Cn²).
"""

import numpy as np

from optical_link_budget_paper.atmosphere.turbulence import Cn2_profile


def sigma_dB(
    lam_um: float,
    el_deg: float,
    h0_m: float = 5.5,
    Z_m: float = 20_000.0,
    vrms: float = 21.0,
    C0: float = 1.7e-14,
    n_points: int = 80_000,
) -> float:
    """
    1-sigma log-irradiance scintillation depth in dB (Earth-to-space).

    ITU-R P.1622-0 eq. (4a): σ²_lnN = 2.253 · k^{7/6} · (1/sinθ)^{11/6} · ∫Cn²(h)·h^{5/6}dh
    ITU-R P.1622-0 eq. (4c): σ²_dBN = (10/ln10)² · σ²_lnN

    Parameters
    ----------
    lam_um   : Wavelength in micrometres.
    el_deg   : Elevation angle in degrees.
    h0_m     : Earth station height above ground in metres (default 5.5 m).
    Z_m      : Effective turbulence ceiling height in metres (default 20 000 m).
    vrms     : RMS wind speed in m/s (default 21.0).
    C0       : Ground-level Cn² in m^{-2/3} (default 1.7e-14).
    n_points : Number of integration points (default 80 000).

    Returns
    -------
    sigma_dB : 1-sigma scintillation fade depth in dB.
    """
    lam_m = lam_um * 1e-6
    k = 2.0 * np.pi / lam_m

    h_arr = np.linspace(h0_m, Z_m, n_points)
    integrand = Cn2_profile(h_arr, C0=C0, vrms=vrms) * h_arr ** (5.0 / 6.0)
    integral = np.trapz(integrand, h_arr)

    sigma2_lnN = (2.253 * k ** (7.0 / 6.0)
                  * (1.0 / np.sin(np.radians(el_deg))) ** (11.0 / 6.0)
                  * integral)
    sigma2_dBN = (10.0 / np.log(10.0)) ** 2 * sigma2_lnN
    return float(np.sqrt(sigma2_dBN))
