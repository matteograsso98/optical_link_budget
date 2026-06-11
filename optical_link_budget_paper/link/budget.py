"""
Link budget — optical gains, pointing losses, transmit power
=============================================================
Computes terminal-side link budget terms and the required transmit
power given a target received power.

References
----------
- Liang et al., arXiv:2204.13177v1, eqs. (1)–(3).
- ITU-R P.1622-0 (atmospheric losses via fso_channel.atmosphere).
"""

import numpy as np

from optical_link_budget_paper.link.geometry import fspl_dB
from optical_link_budget_paper.atmosphere import mie, geometric, scintillation
# ---------------------------------------------------------------------------
# Terminal parameters
# ---------------------------------------------------------------------------

def transmit_gain(divergence_rad: float) -> float:
    """
    Transmitter optical gain G_T (linear, dimensionless).

    Liang et al. eq. (1):  G_T = 16 / Θ_T²

    Parameters
    ----------
    divergence_rad : Full transmit beam divergence angle in radians.

    Returns
    -------
    G_T (linear).
    """
    return 16.0 / divergence_rad**2


def receive_gain(Dr_m: float, lam_um: float) -> float:
    """
    Receiver optical gain G_R (linear, dimensionless).

    Liang et al. eq. (2):  G_R = (π · D_R / λ)²

    Parameters
    ----------
    Dr_m   : Receiver aperture diameter in metres.
    lam_um : Wavelength in micrometres.

    Returns
    -------
    G_R (linear).
    """
    lam_m = lam_um * 1e-6
    return (np.pi * Dr_m / lam_m) ** 2


def pointing_loss_dB(gain_linear: float, pointing_error_rad: float) -> float:
    """
    Pointing loss in dB (negative value, i.e. a loss).

    Liang et al. eq. (3):  L_p = exp(−G · θ_e²)
    In dB:  L_p_dB = 10 · log10(exp(−G · θ_e²))

    Parameters
    ----------
    gain_linear      : Optical gain (linear, G_T or G_R).
    pointing_error_rad : 1-axis pointing error in radians.

    Returns
    -------
    L_p in dB (≤ 0).
    """
    return float(10.0 * np.log10(np.exp(-gain_linear * pointing_error_rad**2)))


# ---------------------------------------------------------------------------
# Link budget solvers
# ---------------------------------------------------------------------------

def required_tx_power_dBm(
    Pr_dBm: float,
    *,
    # Terminal efficiencies
    eta_T: float,
    eta_R: float,
    # Gains (linear)
    G_T: float,
    G_R: float,
    # Pointing losses (dB, typically negative)
    L_T_dB: float,
    L_R_dB: float,
    # Geometry
    lam_um: float,
    el_deg: float,
    hE_km: float,
    hS_km: float,
    # Atmospheric model choice: "mie_geom" or "mie_scin"
    atm_model: str = "mie_geom",
    # Geometric scattering parameters (used when atm_model="mie_geom")
    LW: float = 3.128e-4,
    N: float = 0.5,
    hA_km: float = 20.0,
    phi: float = 1.6,
    # Scintillation parameters (used when atm_model="mie_scin")
    h0_m: float = 5.5,
    Z_m: float = 20_000.0,
    vrms: float = 21.0,
) -> float:
    """
    Required transmit power P_T in dBm to achieve a received power P_r.

    Link equation (all terms in dB):
        P_T = P_r
              − η_R_dB − η_T_dB
              − G_R_dB − G_T_dB
              − L_T_dB − L_R_dB
              + FSPL_dB
              + A_atm_dB

    Parameters
    ----------
    Pr_dBm    : Required received power in dBm.
    eta_T     : Transmitter optical efficiency (0 < η ≤ 1).
    eta_R     : Receiver optical efficiency (0 < η ≤ 1).
    G_T       : Transmitter gain (linear).
    G_R       : Receiver gain (linear).
    L_T_dB    : Transmitter pointing loss in dB (≤ 0).
    L_R_dB    : Receiver pointing loss in dB (≤ 0).
    lam_um    : Wavelength in µm.
    el_deg    : Elevation angle in degrees.
    hE_km     : Ground station altitude in km.
    hS_km     : Satellite altitude in km.
    atm_model : "mie_geom" (Mie + geometric scattering)
                or "mie_scin" (Mie + scintillation).
    LW, N, hA_km, phi : Geometric scattering parameters.
    h0_m, Z_m, vrms   : Scintillation parameters.

    Returns
    -------
    P_T in dBm.
    """
    A_mie = mie.attenuation_dB(lam_um, hE_km, el_deg)

    if atm_model == "mie_geom":
        A_atm = A_mie + geometric.attenuation_dB(el_deg, LW, N, lam_um, hA_km, hE_km, phi)
    elif atm_model == "mie_scin":
        A_atm = A_mie + scintillation.sigma_dB(lam_um, el_deg, h0_m, Z_m, vrms)
    else:
        raise ValueError(f"Unknown atm_model '{atm_model}'. Use 'mie_geom' or 'mie_scin'.")

    budget = (
        Pr_dBm
        - 10.0 * np.log10(eta_R)
        - 10.0 * np.log10(eta_T)
        - 10.0 * np.log10(G_R)
        - 10.0 * np.log10(G_T)
        - L_T_dB
        - L_R_dB
        + fspl_dB(lam_um, el_deg, hE_km, hS_km)
        + A_atm
    )
    return float(budget)
