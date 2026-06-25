"""
Geometric scattering attenuation (fog / dense clouds) — Liang et al., eqs. (8)–(10)
=====================================================================================
Beer-Lambert extinction through the troposphere.

Reference: Liang et al., arXiv:2204.13177v1, Section III-B-2, eqs. (8)–(10).
"""

import numpy as np


def visibility_km(LW: float, N: float) -> float:
    """
    Meteorological visibility V in km from cloud/fog microphysics.

    Liang et al. eq. (8):  V = 1.002 / (LW · N)^{0.6473}

    Parameters
    ----------
    LW : Liquid water content in g m^{-3} (must be > 0).
    N  : Cloud droplet number concentration in cm^{-3} (must be > 0).

    Returns
    -------
    V : Visibility in km.
    """
    if LW <= 0 or N <= 0:
        raise ValueError("LW and N must be strictly positive.")
    return 1.002 / (LW * N) ** 0.6473


def extinction_coefficient_km(lam_um: float, V_km: float, phi: float) -> float:
    """
    Geometric scattering extinction coefficient θ_A in km^{-1} (Kim model).

    Liang et al. eq. (9):  θ_A = (3.91 / V) · (λ_nm / 550)^{−φ}

    Parameters
    ----------
    lam_um : Wavelength in micrometres.
    V_km   : Visibility in km — scalar or numpy array.
    phi    : Particle-size coefficient (Kim model).

    Returns
    -------
    theta_A : Extinction coefficient in km^{-1}.
    """
    lam_nm = lam_um * 1e3
    return (3.91 / V_km) * (lam_nm / 550.0) ** (-phi)


def path_length_km(hA_km: float, hE_km: float, el_deg: float) -> float:
    """
    Slant path length through the troposphere in km.

    Liang et al. §III-B-2:  d_A = (h_A − h_E) / sin(θ_E)

    Parameters
    ----------
    hA_km  : Troposphere top height in km (typically 20 km).
    hE_km  : Ground station altitude in km.
    el_deg : Elevation angle in degrees.

    Returns
    -------
    d_A : Path length in km.
    """
    if hA_km <= hE_km:
        raise ValueError("hA_km must be greater than hE_km.")
    return (hA_km - hE_km) / np.sin(np.radians(el_deg))


def attenuation_dB(
    el_deg: float,
    lam_um: float,
    hA_km: float,
    hE_km: float,
    phi: float,
    *,
    LW: float | None = None,
    N: float | None = None,
    V_km=None,
):
    """
    Total geometric scattering attenuation in dB (positive = loss).

    Beer-Lambert law — Liang et al. eq. (10):  I_g = exp(−θ_A · d_A)
    Converted to dB:  A_g = −10 · log10(I_g) = 4.3429 · θ_A · d_A

    Visibility source — exactly one of the two must be provided:
      LW + N  : visibility derived from cloud microphysics via visibility_km().
                Scalar floats only; both must be > 0.
      V_km    : visibility supplied directly (e.g. from GFS surface field).
                Scalar or numpy array; pixels with V ≤ 0 are treated as
                clear sky (0 dB).

    Parameters
    ----------
    el_deg : Elevation angle in degrees.
    lam_um : Wavelength in micrometres.
    hA_km  : Troposphere top height in km.
    hE_km  : Ground station altitude in km.
    phi    : Kim model particle-size coefficient.
    LW     : Liquid water content in g m^{-3}  — keyword-only, ignored if V_km given.
    N      : Cloud droplet concentration in cm^{-3} — keyword-only, ignored if V_km given.
    V_km   : Meteorological visibility in km     — keyword-only, overrides LW/N.

    Returns
    -------
    A_g : Geometric scattering attenuation in dB.
          Scalar float when all inputs are scalar; numpy array when V_km is an array.
    """
    if V_km is not None:
        V_safe = np.where(np.asarray(V_km) > 0, V_km, np.inf)
    elif LW is not None and N is not None:
        V_safe = visibility_km(LW, N)
    else:
        raise ValueError(
            "Provide either V_km or both LW and N."
        )

    dA = path_length_km(hA_km, hE_km, el_deg)
    tA = extinction_coefficient_km(lam_um, V_safe, phi)
    Ig = np.exp(-tA * dA)
    result = -10.0 * np.log10(np.maximum(Ig, 1e-30))
    return float(result) if np.ndim(result) == 0 else result
