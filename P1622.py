import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── P.1622-0 IMPLEMENTATIONS (Mie, Scintillation) ────────────────────────────────────────────────

def mie_attenuation_dB(lam_um, hE_km, el_deg):
    """
    ITU-R P.1622-0, Annex 1, §3.1 — Mie scattering attenuation.
    Valid: 150–375 THz, hE 0–5 km, el > 45° (accuracy ~0.1 dB).

    Parameters
    ----------
    lam_um : float   wavelength in micrometres
    hE_km  : float   ground station altitude above MSL in km
    el_deg : float   elevation angle in degrees

    Returns
    -------
    AS : float   total path Mie attenuation in dB (positive = loss)
    """
    # Step 1: wavelength-dependent empirical coefficients (eq. 1a–1d)
    a = -0.000545*lam_um**2 + 0.002*lam_um  - 0.0038
    b =  0.00628 *lam_um**2 - 0.0232*lam_um + 0.00439
    c = -0.028   *lam_um**2 + 0.101*lam_um  - 0.18
    d = -0.228   *lam_um**3 + 0.922*lam_um**2 - 1.26*lam_um + 0.719

    # Step 2: extinction ratio from hE to ∞ in Nepers (eq. 2)
    tau = a*hE_km**3 + b*hE_km**2 + c*hE_km + d

    # Step 3: path attenuation along slant (eq. 3)
    AS = 4.3429 * tau / np.sin(np.radians(el_deg))
    return AS


def Cn2_profile(h_m, C0=1.7e-14, vrms=21.0):
    """
    ITU-R P.1621, §5.1.1 — Hufnagel-Valley turbulence structure profile.

    Parameters
    ----------
    h_m  : array   height above ground in metres
    C0   : float   ground-level turbulence constant (m^{-2/3}), default 1.7e-14
    vrms : float   rms wind speed along vertical path (m/s)

    Returns
    -------
    Cn2 : array   turbulence structure parameter (m^{-2/3})
    """
    t1 = 8.148e-56 * vrms**2 * h_m**10 * np.exp(-h_m / 1000)
    t2 = 2.7e-16 * np.exp(-h_m / 1500)
    t3 = C0 * np.exp(-h_m / 100)
    return t1 + t2 + t3


def scintillation_sigma_dB(lam_um, el_deg, h0_m=5.5, Z_m=20000, vrms=21.0):
    """
    ITU-R P.1622-0, Annex 1, eq.(4a)+(4c) — amplitude scintillation.
    Returns 1-sigma log-irradiance fluctuation in dB (Earth-to-space direction).

    Parameters
    ----------
    lam_um : float   wavelength in micrometres
    el_deg : float   elevation angle in degrees
    h0_m   : float   earth station height above ground in metres (default 5.5 m)
    Z_m    : float   effective turbulence height in metres (default 20 000 m)
    vrms   : float   rms wind speed (m/s, default 21)

    Returns
    -------
    sigma_dBN : float   1-sigma scintillation fade depth in dB
    """
    lam_m = lam_um * 1e-6
    k = 2 * np.pi / lam_m

    # Numerical integration of Cn2(h) * h^(5/6) from h0 to Z  (eq. 4a)
    h_arr = np.linspace(h0_m, Z_m, 80000)
    integrand = Cn2_profile(h_arr, vrms=vrms) * h_arr**(5/6)
    integral = np.trapezoid(integrand, h_arr)

    # Variance of log-irradiance (Np^2)  —  eq. (4a)
    sigma2_lnN = 2.253 * k**(7/6) * (1 / np.sin(np.radians(el_deg)))**(11/6) * integral

    # Convert to dB^2  —  eq. (4c)
    sigma2_dBN = (10 / np.log(10))**2 * sigma2_lnN
    return np.sqrt(sigma2_dBN)

# ── "LINK BUDGET ANALYSIS FOR FSO" IMPLEMENTATIONS ────────────────────────────────────────────────


def geometrical_scattering_dB(
    el_deg,
    LW: float = 3.128*10**-4,
    N: float = 0.5 ,
    wavelength_nm: float = 1550,
    h_A: float = 20.0,
    h_E: float = 1.0,
    phi: float = 1.6,
) -> dict:
    """
    Compute atmospheric attenuation due to geometrical scattering (Ig).
    
    Parameters
    ----------
    LW          : Liquid water content (g/m^-3)
    N           : Cloud number concentration (cm^-3)
    wavelength_nm: Wavelength of the optical beam (nm)
    h_A         : Height of the troposphere layer (km)
    h_E         : Altitude of the ground station (km)
    phi         : Particle size coefficient (default 1.6)
    el_deg      : Elevation angle in degrees

    Returns
    -------
    dict with intermediate values and final Ig
    """

    # --- Input validation ---
    if LW <= 0 or N <= 0:
        raise ValueError("LW and N must be strictly positive.")
    if h_A <= h_E:
        raise ValueError("h_A (troposphere height) must be greater than h_E (station altitude).")

    # --- Step 1: Visibility V (km) — Eq. (8) ---
    V = 1.002 / ((LW * N) ** 0.6473)

    # --- Step 2: Attenuation coefficient theta_A — Eq. (9) ---
    theta_A = (3.91 / V) * (wavelength_nm / 550) ** (-phi)

    # --- Step 3: Path length through troposphere d_A (km) ---
    # Using elevation angle: d_A = (h_A - h_E) * csc(theta_E)
    d_A = (h_A - h_E) / np.sin(np.radians(el_deg))

    # --- Step 4: Geometrical scattering attenuation Ig — Eq. (10) ---
    # Beer-Lambert law: Ig = exp(-theta_A * d_A)
    return np.exp(-theta_A * d_A)
