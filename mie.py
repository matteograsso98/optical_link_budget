"""
Mie scattering attenuation — ITU-R P.1622-0, Annex 1, §3.1
============================================================
Valid: 150–375 THz (λ ≈ 0.8–2.0 µm), h_E ∈ [0, 5] km, elevation > 45°
Accuracy: ~0.1 dB

Reference: ITU-R P.1622-0, equations (1a–1d) and (3).
"""

import numpy as np


def extinction_ratio(lam_um: float, hE_km: float) -> float:
    """
    Integrated Mie extinction ratio τ from ground station to ∞ (Nepers).

    ITU-R P.1622-0 eq. (1a–1d) and (2).

    Parameters
    ----------
    lam_um : Wavelength in micrometres (valid: 0.8–2.0 µm).
    hE_km  : Ground station altitude above MSL in km (valid: 0–5 km).

    Returns
    -------
    tau : Extinction ratio in Nepers (clamped to 0 if negative, i.e. out of range).
    """
    a = -0.000545 * lam_um**2 + 0.002  * lam_um - 0.0038
    b =  0.00628  * lam_um**2 - 0.0232 * lam_um + 0.00439
    c = -0.028    * lam_um**2 + 0.101  * lam_um - 0.18
    d = -0.228    * lam_um**3 + 0.922  * lam_um**2 - 1.26 * lam_um + 0.719

    tau = a * hE_km**3 + b * hE_km**2 + c * hE_km + d
    return float(np.maximum(tau, 0.0))


def attenuation_dB(lam_um: float, hE_km: float, el_deg: float) -> float:
    """
    Total slant-path Mie attenuation in dB (positive = loss).

    ITU-R P.1622-0 eq. (3):  A_S = 4.3429 · τ / sin(θ_E)

    Parameters
    ----------
    lam_um : Wavelength in micrometres.
    hE_km  : Ground station altitude above MSL in km.
    el_deg : Elevation angle in degrees.

    Returns
    -------
    A_S : Mie attenuation in dB.
    """
    tau = extinction_ratio(lam_um, hE_km)
    return 4.3429 * tau / np.sin(np.radians(el_deg))
