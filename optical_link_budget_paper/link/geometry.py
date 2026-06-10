"""
Link geometry — slant distance and free-space path loss
========================================================
Covers the purely geometric part of the link budget
(independent of atmospheric physics).

Reference: Liang et al., arXiv:2204.13177v1, eqs. (4)–(5).
"""

import numpy as np


RE_KM = 6371.0  # Mean Earth radius (km)


def slant_distance_km(
    el_deg: float,
    hE_km: float,
    hS_km: float,
    RE_km: float = RE_KM,
) -> float:
    """
    Slant distance from ground station to satellite.

    Liang et al. eq. (4):
        R  = R_E + h_E
        H  = h_S − h_E
        d_GS = R · (sqrt(((R+H)/R)² − cos²(θ_E)) − sin(θ_E))

    Parameters
    ----------
    el_deg : Elevation angle in degrees.
    hE_km  : Ground station altitude above MSL in km.
    hS_km  : Satellite altitude above MSL in km.
    RE_km  : Earth radius in km (default 6371 km).

    Returns
    -------
    d_GS : Slant distance in km.
    """
    R     = RE_km + hE_km
    H     = hS_km - hE_km
    theta = np.radians(el_deg)
    return float(R * (np.sqrt(((R + H) / R)**2 - np.cos(theta)**2) - np.sin(theta)))


def fspl_dB(
    lam_um: float,
    el_deg: float,
    hE_km: float,
    hS_km: float,
    RE_km: float = RE_KM,
) -> float:
    """
    Free-space path loss in dB (positive = loss).

    Liang et al. eq. (5):  FSPL = (4π · d_GS / λ)²

    Parameters
    ----------
    lam_um : Wavelength in micrometres.
    el_deg : Elevation angle in degrees.
    hE_km  : Ground station altitude in km.
    hS_km  : Satellite altitude in km.
    RE_km  : Earth radius in km.

    Returns
    -------
    FSPL in dB.
    """
    d_m  = slant_distance_km(el_deg, hE_km, hS_km, RE_km) * 1e3
    lam_m = lam_um * 1e-6
    fspl  = (4.0 * np.pi * d_m / lam_m) ** 2
    return float(10.0 * np.log10(fspl))
