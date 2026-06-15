"""
Scenario configuration — single source of truth
================================================
All physical and system parameters live here.
Change values in this file; everything else adapts automatically.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Channel / atmosphere
# ---------------------------------------------------------------------------

@dataclass
class AtmosphereConfig:
    """ITU-R P.1621-2 / P.1622-0 atmosphere parameters."""

    # Geometry
    hE_km: float = 1.0      # Ground station altitude above MSL (km)
    hA_km: float = 20.0     # Troposphere top height for geometric scattering (km)
    h0_m:  float = 5.5      # Earth station height above local ground (m) — scintillation
    Z_m:   float = 20_000.0 # Turbulence ceiling (m)

    # Wavelength
    lam_um: float = 1.550   # Operating wavelength (µm) — C-band, 193.41 THz

    # Turbulence / scintillation
    vrms: float = 21.0      # RMS wind speed (m/s) — corresponds to vg ≈ 2.3 m/s
    C0:   float = 1.7e-14   # Ground-level Cn² (m^{-2/3})

    # Geometric scattering (fog / clouds) — Kim model
    LW:  float = 3.128e-4   # Liquid water content (g m^{-3})
    N:   float = 0.5        # Cloud droplet concentration (cm^{-3})
    phi: float = 1.6        # Kim model particle-size coefficient

# ---------------------------------------------------------------------------
# Satellite / orbit
# ---------------------------------------------------------------------------

@dataclass
class OrbitConfig:
    """Satellite orbit parameters."""

    hS_km: float = 550.0    # Satellite altitude above MSL (km) — typical LEO


# ---------------------------------------------------------------------------
# Optical terminal
# ---------------------------------------------------------------------------

@dataclass
class TerminalConfig:
    """Transmit / receive terminal parameters."""

    # Receiver aperture
    Dr_m: float = 80e-3     # Receiver telescope diameter (m)

    # Optical efficiencies
    eta_T: float = 0.8      # Transmitter optical efficiency (0–1)
    eta_R: float = 0.8      # Receiver optical efficiency (0–1)

    # Transmitter beam
    Theta_T_rad: float = 15e-6  # Full transmit beam divergence angle (rad)

    # Pointing errors
    theta_T_rad: float = 1e-6   # Transmitter pointing error (rad)
    theta_R_rad: float = 1e-6   # Receiver pointing error (rad)

    # Required received power
    Pr_dBm: float = -32.5   # Target received power (dBm)

    # Convenience parameters for SNR calculations
    P_tx: float = 1000.0         # Transmit power (W) — convenience parameter for SNR calculations
    noise_dBm: float = -100.0  # Noise power (dBm) — convenience parameter for SNR calculations
    bandwidth_hz: float = 10e9 # bandwidth (Hz) — convenience parameter for SNR calculations

    # UPA geometry (for MIMO / beamforming)
    N_x: int   = 16
    N_y: int   = 16
    d_x: float = 0.5
    d_y: float = 0.5


# ---------------------------------------------------------------------------
# Convenience: pre-built default config
# ---------------------------------------------------------------------------

DEFAULT_ATM     = AtmosphereConfig()
DEFAULT_ORBIT   = OrbitConfig()
DEFAULT_TERMINAL = TerminalConfig()
