"""
Amplitude scintillation — ITU-R P.1622-0, Annex 1, eq. (4a) and (4c)
======================================================================
Hufnagel-Valley turbulence profile from ITU-R P.1621-2, eq. (6).

Reference: ITU-R P.1622-0 (scintillation), ITU-R P.1621-2 §5.1.1 (Cn²).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Turbulence profile (imported here to avoid circular deps)
# ---------------------------------------------------------------------------

def wind_rms(vg: float = 2.3) -> float:
    """
    RMS wind speed along the vertical path (Bufton simplified model).

    ITU-R P.1621-2 eq. (5):  v_rms = sqrt(v_g² + 30.69·v_g + 348.91)

    Parameters
    ----------
    vg : Ground wind speed in m/s.  Unknown → use 2.3 m/s → v_rms ≈ 21 m/s.

    Returns
    -------
    v_rms in m/s.
    """
    return float(np.sqrt(vg**2 + 30.69 * vg + 348.91))


def Cn2_profile(
    h_m: np.ndarray,
    C0: float = 1.7e-14,
    vg: float | None = None,
    vrms: float | None = None,
) -> np.ndarray:
    """
    Hufnagel-Valley Cn² profile.  ITU-R P.1621-2 eq. (6).

    Parameters
    ----------
    h_m  : Height above ground in metres.
    C0   : Ground-level Cn² in m^{-2/3}.
    vg   : Ground wind speed in m/s.  Used to derive vrms via wind_rms().
    vrms : RMS wind speed in m/s.  Takes precedence over vg when provided.
           If neither is given, defaults to vg = 2.3 m/s.
    """
    if vrms is None:
        vrms = wind_rms(2.3 if vg is None else vg)
    h = np.asarray(h_m, dtype=float)
    return (
        8.148e-56 * vrms*2 * h*10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + C0 * np.exp(-h / 100.0)
    )


def _scin_integral(
    h0_m: float,
    Z_m: float = 20_000.0,
    vrms: float = 21.0,
    C0: float = 1.7e-14,
    n_points: int = 80_000,
) -> float:
    """
    Elevation-independent part: ∫_{h0_m}^{Z_m} Cn²(h) · h^{5/6} dh.

    Separating this from the elevation factor allows the LUT builder to compute
    it once per user instead of once per (user, elevation) pair — ~160x speedup.
    """
    h_arr = np.linspace(h0_m, Z_m, n_points)
    integrand = Cn2_profile(h_arr, C0=C0, vrms=vrms) * h_arr ** (5.0 / 6.0)
    return float(np.trapezoid(integrand, h_arr))


def _sigma_from_integral(
    k: float,
    el_deg: float,
    integral,           # scalar or (U,) ndarray
):
    """
    Apply the elevation-dependent factor to a precomputed integral value.

    Parameters
    ----------
    k        : Wave number 2π/λ [rad/m].
    el_deg   : Elevation angle in degrees (scalar).
    integral : Result of _scin_integral — scalar or (U,) array.

    Returns
    -------
    sigma_dB values, same shape as integral.
    """
    sigma2_lnN = (2.253 * k ** (7.0 / 6.0)
                  * (1.0 / np.sin(np.radians(el_deg))) ** (11.0 / 6.0)
                  * integral)
    sigma2_dBN = (10.0 / np.log(10.0)) ** 2 * sigma2_lnN
    return np.sqrt(sigma2_dBN)


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
    integral = _scin_integral(h0_m, Z_m, vrms, C0, n_points)
    return float(_sigma_from_integral(k, el_deg, integral))
