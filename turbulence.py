"""
Atmospheric turbulence parameters — ITU-R P.1621-2, §5
=======================================================
Computes Fried coherence length r0, isoplanatic angle θ0,
and coherence time τ0 for an Earth-to-space FSO link.

Reference: ITU-R P.1621-2 (07/2015), eqs. (5)–(21).
"""

import numpy as np

Z_TURB_M = 20_000.0  # Effective turbulence ceiling (m) — P.1621-2 §5.1.1


# ---------------------------------------------------------------------------
# Wind model
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


def wind_profile(h_m: np.ndarray, vg: float = 2.3) -> np.ndarray:
    """
    Horizontal wind speed profile vs. altitude.

    ITU-R P.1621-2 eq. (19):  v(h) = v_g + 30 · exp(−((h − 9400)/4800)²)

    Parameters
    ----------
    h_m : Height above ground in metres.
    vg  : Ground wind speed in m/s.

    Returns
    -------
    v(h) in m/s.
    """
    h = np.asarray(h_m, dtype=float)
    return vg + 30.0 * np.exp(-((h - 9400.0) / 4800.0) ** 2)


# ---------------------------------------------------------------------------
# Layer height grid
# ---------------------------------------------------------------------------

def layer_heights_m(n: int = 139) -> np.ndarray:
    """
    Exponentially-spaced height grid from 1 m to Z_TURB_M.

    ITU-R P.1621-2 eq. (7):  h_i = exp((i−1) · ln(Z_TURB) / (N−1))

    Parameters
    ----------
    n : Number of layers (default 139 per ITU-R).

    Returns
    -------
    h_arr : Heights in metres, shape (n,).
    """
    i = np.arange(1, n + 1)
    factor = np.log(Z_TURB_M) / float(n - 1)
    return np.exp((i - 1) * factor)


# ---------------------------------------------------------------------------
# Turbulence profile (imported here to avoid circular deps)
# ---------------------------------------------------------------------------

def Cn2_profile(h_m: np.ndarray, C0: float = 1.7e-14, vg: float = 2.3) -> np.ndarray:
    """
    Hufnagel-Valley Cn² profile.  ITU-R P.1621-2 eq. (6).

    Parameters
    ----------
    h_m : Height above ground in metres.
    C0  : Ground-level Cn² in m^{-2/3}.
    vg  : Ground wind speed in m/s.
    """
    h    = np.asarray(h_m, dtype=float)
    vrms = wind_rms(vg)
    return (
        8.148e-56 * vrms**2 * h**10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + C0 * np.exp(-h / 100.0)
    )


# ---------------------------------------------------------------------------
# Coherence length r0
# ---------------------------------------------------------------------------

def coherence_length_r0(
    lam_um: float,
    el_deg: float,
    h0_m: float = 0.0,
    vg: float = 2.3,
    C0: float = 1.7e-14,
) -> float:
    """
    Fried coherence length r0 in metres.

    ITU-R P.1621-2 eqs. (9)–(13). Valid for h0 ∈ [0, 5] km, θ > 45°.

    Parameters
    ----------
    lam_um : Wavelength in micrometres.
    el_deg : Elevation angle in degrees.
    h0_m   : Earth station height above ground in metres.
    vg     : Ground wind speed in m/s.
    C0     : Ground-level Cn² in m^{-2/3}.

    Returns
    -------
    r0 in metres.
    """
    vrms  = wind_rms(vg)
    theta = np.radians(el_deg)

    C_wind   = (8.148e-17 * vrms**2) * (
        0.0026 * (1.0 - np.exp(0.001 * h0_m**1.055 - 5.0)) + 3.587369
    )
    C_height = -6.5594e-19 + 4.05e-13 * np.exp(-h0_m / 1500.0)
    C_turb   = -C0 * (1.383899e-85 - 100.0 * np.exp(-h0_m / 100.0))

    integral = C_wind + C_height + C_turb
    r0 = (1.1654e-8 * lam_um**1.2 * np.sin(theta)**0.6
          / max(integral, 1e-30)**0.6)
    return float(r0)


# ---------------------------------------------------------------------------
# Isoplanatic angle θ0
# ---------------------------------------------------------------------------

def isoplanatic_angle_theta0(
    lam_um: float,
    el_deg: float,
    h0_m: float = 0.0,
    vg: float = 2.3,
    C0: float = 1.7e-14,
) -> float:
    """
    Isoplanatic angle θ0 in radians.

    ITU-R P.1621-2 eqs. (15)–(18). Valid for h0 ∈ [0, 5] km, θ > 45°.

    Parameters
    ----------
    lam_um : Wavelength in micrometres.
    el_deg : Elevation angle in degrees.
    h0_m   : Earth station height above ground in metres.
    vg     : Ground wind speed in m/s.
    C0     : Ground-level Cn² in m^{-2/3}.

    Returns
    -------
    θ0 in radians.
    """
    vrms  = wind_rms(vg)
    theta = np.radians(el_deg)

    C_wind = (8.148e-10 * vrms**2) * (
        0.002 * (1.0 - np.exp(0.0018 * h0_m**1.014 - 9.0)) + 2.0043
    )
    C_height = (
        -7.0236e-23 * h0_m**4
        + 1.5015e-18 * h0_m**3
        - 8.9834e-15 * h0_m**2
        + 2.3855e-12 * h0_m
        + 9.6181e-8
    )
    C_turb = 3.3e5 * C0 * np.exp(-0.000222 * h0_m**1.45)

    integral = C_wind + C_height + C_turb
    theta0 = (3.663e-9 * lam_um**1.2 * np.sin(theta)**1.6
              / max(integral, 1e-30)**0.6)
    return float(theta0)


# ---------------------------------------------------------------------------
# Coherence time τ0
# ---------------------------------------------------------------------------

def coherence_time_tau0(
    lam_um: float,
    el_deg: float,
    h0_m: float = 0.0,
    vg: float = 2.8,
    C0: float = 1.7e-14,
) -> float:
    """
    Atmospheric coherence time τ0 in seconds.

    ITU-R P.1621-2 eqs. (19)–(21). Valid for θ > 45°.

    Parameters
    ----------
    lam_um : Wavelength in micrometres.
    el_deg : Elevation angle in degrees.
    h0_m   : Earth station height above ground in metres.
    vg     : Ground wind speed in m/s (ITU-R default 2.8 m/s for τ0).
    C0     : Ground-level Cn² in m^{-2/3}.

    Returns
    -------
    τ0 in seconds.
    """
    theta   = np.radians(el_deg)
    heights = layer_heights_m()
    heights = heights[heights >= h0_m]

    cn2 = Cn2_profile(heights, C0, vg)
    v   = wind_profile(heights, vg)
    dh  = np.diff(heights, prepend=h0_m)

    v53  = np.sum(cn2 * v**(5.0 / 3.0) * dh)   # eq. (20)
    tau0 = (2.729e-8 * lam_um**1.2 * np.sin(theta)**0.6
            / max(v53, 1e-30)**0.6)              # eq. (21)
    return float(tau0)
