"""
DynamicLinkBudget — FSO LEO channel model
==========================================
Optical counterpart of channel.Channel (the RF model).

Structural differences from the RF model:
  - No 3GPP large-scale fading (not applicable to FSO)
  - No ITU-R P.618 (RF rain/gas); replaced by ITU-R P.1622 FSO atmosphere
  - 2-D (U × G) LUT: one row per user, vectorised over elevation angle
  - Shannon capacity via snr.py

Three atmospheric models (set via AtmosphereConfig.atm_model):
  "mie_geom"      — Mie (ITU-R P.1622) + geometric scattering (Kim/Liang)
  "mie_scin"      — Mie + amplitude scintillation (ITU-R P.1622)
  "mie_geom_scin" — Mie + geometric + scintillation (most complete)
"""

from __future__ import annotations

import os

import numpy as np

from optical_link_budget_paper.atmosphere import mie, geometric, scintillation
from optical_link_budget_paper.link import budget
from channel_model.lut_interp import lookup_lut_dB
from dynamic_link_budget.snr import compute_snr_dB, compute_shannon_rate
from config import AtmosphereConfig, OrbitConfig, TerminalConfig


class DynamicLinkBudget:
    """
    Dynamic FSO link budget for a LEO satellite pass.

    Parameters
    ----------
    atm : AtmosphereConfig
        Wavelength, turbulence profile, Kim/cloud model parameters, and
        the atmospheric model selector (atm_model).
    orb : OrbitConfig
        Satellite altitude.
    trm : TerminalConfig
        Aperture, optical efficiencies, transmit power, noise, bandwidth.

    Usage
    -----
    1. Instantiate with typed config objects.
    2. Call ``precompute_lut(user_hE_km, ...)`` once before the simulation
       loop to build the (U × G) atmospheric loss table.
    3. Call ``compute(ts_index, slant_range_m, az_deg, el_deg)`` at every
       timestep to get channel matrices, SNR, and Shannon rates.
    """

    def __init__(
        self,
        atm: AtmosphereConfig,
        orb: OrbitConfig,
        trm: TerminalConfig,
    ) -> None:
        self.atm = atm
        self.orb = orb
        self.trm = trm

        self.lam_m   = atm.lam_um * 1e-6
        self.G_R     = budget.receive_gain(trm.Dr_m, atm.lam_um)
        self.noise_W = 10 ** (trm.noise_dBm / 10) * 1e-3

        G_T_lin = budget.transmit_gain(trm.Theta_T_rad)
        L_T_dB  = budget.pointing_loss_dB(G_T_lin, trm.theta_T_rad)
        L_R_dB  = budget.pointing_loss_dB(self.G_R,  trm.theta_R_rad)

        self.G_T   = G_T_lin
        # Effective transmit power after terminal optical losses
        self.P_eff = (trm.P_tx
                      * trm.eta_T * 10 ** (L_T_dB / 10)
                      * trm.eta_R * 10 ** (L_R_dB / 10))

        # Atmospheric LUT — populated by precompute_lut()
        self._att_lut_dB       = None   # (U, G) float64 [dB]
        self._att_lut_elev_deg = None   # (G,)   float64 [deg]

    # ── Atmospheric LUT (ITU-R P.1622) ───────────────────────────────────────

    def precompute_lut(
        self,
        user_hE_km,
        user_h0_m=None,
        elevation_grid_deg=None,
        lut_cache_path=None,
    ) -> None:
        """
        Build the (U × G) atmospheric loss LUT for every user across an
        elevation grid. Mirrors channel.Channel.precompute_attenuation_lut().

        Parameters
        ----------
        user_hE_km        : (U,) ground station altitude above MSL [km].
        user_h0_m         : (U,) station height above local ground [m].
                            Required when atm_model includes scintillation.
        elevation_grid_deg: (G,) elevation grid [deg].
                            Default: 10° to 90° in 0.5° steps.
        lut_cache_path    : optional .npy path for disk caching.
                            Cache is invalidated when user_hE_km changes.
        """
        user_hE_km = np.asarray(user_hE_km, dtype=np.float64)
        U = user_hE_km.size

        if elevation_grid_deg is None:
            elevation_grid_deg = np.arange(10, 90.5, 0.5, dtype=np.float64)
        G = elevation_grid_deg.size

        # Disk cache — skip recomputation when inputs are unchanged
        if lut_cache_path is not None:
            meta_path = str(lut_cache_path) + ".elev.npy"
            hE_path   = str(lut_cache_path) + ".hE.npy"
            if (os.path.exists(lut_cache_path)
                    and os.path.exists(meta_path)
                    and os.path.exists(hE_path)
                    and np.allclose(np.load(hE_path), user_hE_km)):
                self._att_lut_dB       = np.load(lut_cache_path)
                self._att_lut_elev_deg = np.load(meta_path)
                return

        atm       = self.atm
        atm_model = atm.atm_model
        lam_um    = atm.lam_um

        # Pre-compute per-user scintillation path integrals.
        # The integral ∫Cn²(h)·h^{5/6}dh is independent of elevation, so we
        # evaluate it once per user and apply the sin^{-11/6}(el) factor over
        # the full elevation grid in the LUT loop below.
        scin_coeff = scin_dBfact = None
        scin_integrals = None

        if "scin" in atm_model:
            if user_h0_m is None:
                raise ValueError(
                    f"user_h0_m is required for atm_model='{atm_model}'."
                )
            user_h0_m = np.asarray(user_h0_m, dtype=np.float64)
            k = 2.0 * np.pi / self.lam_m
            scin_coeff  = 2.253 * k ** (7.0 / 6.0)
            scin_dBfact = (10.0 / np.log(10.0)) ** 2

            scin_integrals = np.empty(U, dtype=np.float64)
            for u in range(U):
                h = np.linspace(user_h0_m[u], atm.Z_m, 80_000)
                integrand = scintillation.Cn2_profile(
                    h, C0=atm.C0, vrms=atm.vrms
                ) * h ** (5.0 / 6.0)
                scin_integrals[u] = np.trapz(integrand, h)

        # Build LUT
        losses = np.zeros((U, G), dtype=np.float64)

        for i, el in enumerate(elevation_grid_deg):
            losses[:, i] = np.array(
                [mie.attenuation_dB(lam_um, hE, el) for hE in user_hE_km]
            )

            if "geom" in atm_model:
                losses[:, i] += np.array([
                    geometric.attenuation_dB(
                        el, atm.LW, atm.N, lam_um, atm.hA_km, hE, atm.phi
                    )
                    for hE in user_hE_km
                ])

            if "scin" in atm_model:
                sin_el     = np.sin(np.radians(el))
                sigma2_lnN = (scin_coeff
                              * (1.0 / sin_el) ** (11.0 / 6.0)
                              * scin_integrals)
                losses[:, i] += np.sqrt(scin_dBfact * sigma2_lnN)

        self._att_lut_dB       = losses
        self._att_lut_elev_deg = elevation_grid_deg

        if lut_cache_path is not None:
            np.save(lut_cache_path,                        losses)
            np.save(str(lut_cache_path) + ".elev.npy",    elevation_grid_deg)
            np.save(str(lut_cache_path) + ".hE.npy",      user_hE_km)

    # ── Channel computation ───────────────────────────────────────────────────

    def compute(
        self,
        ts_index: int,
        slant_range_m,
        az_deg,
        el_deg,
        user_indices=None,
    ) -> dict:
        """
        Compute FSO channel matrices, SNR, and Shannon capacity.

        Mirrors channel.Channel.compute() for API consistency.

        Parameters
        ----------
        ts_index      : timestep index (forwarded to the output dict).
        slant_range_m : (U,) slant range from ground station to satellite [m].
        az_deg        : (U,) azimuth angle [deg] — not used in FSO (no spatial
                        multiplexing), kept for API symmetry with Channel.
        el_deg        : (U,) elevation angle [deg].
        user_indices  : (U,) int row indices into the LUT.
                        Defaults to 0, 1, …, U-1.

        Returns
        -------
        dict with keys:
            ts_index   — echo of ts_index.
            H_ideal    — (U, 1) complex, free-space channel matrix.
            H_sys      — (U, 1) complex, channel with atmospheric losses applied.
            FSPL_dB    — (U,)  free-space path loss [dB].
            A_atm_dB   — (U,)  total atmospheric attenuation [dB].
            SNR_dB     — (U,)  ideal SNR [dB].
            SNR_real_dB— (U,)  realistic SNR with atmospheric losses [dB].
            rate_ideal — (U,)  Shannon capacity, ideal channel [bps].
            rate_real  — (U,)  Shannon capacity, realistic channel [bps].
            slant_m    — (U,)  slant range [m].
        """
        lam = self.lam_m
        el  = np.asarray(el_deg,        dtype=np.float64)
        d   = np.asarray(slant_range_m, dtype=np.float64)
        U   = el.size

        if user_indices is None:
            user_indices = np.arange(U, dtype=np.int64)

        # ── Ideal channel matrix ─────────────────────────────────────────────
        # H_ideal[u, 0] = ξ · G_T · exp(−j 2π d/λ) / d
        # Shape (U, 1): FSO is a scalar channel per user (no spatial multiplexing).
        xi = np.sqrt(self.G_R / self.noise_W) / (4.0 * np.pi / lam)

        slant_ok = np.isfinite(d) & (d > 0)
        safe_d   = np.where(slant_ok, d, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            phase_path_loss = np.exp(-1j * safe_d * 2.0 * np.pi / lam) / safe_d

        H_ideal = xi * phase_path_loss[:, None] * self.G_T   # (U, 1)

        # ── Atmospheric FSO losses (ITU-R P.1622) ───────────────────────────
        if self._att_lut_dB is not None:
            fso_loss_dB = lookup_lut_dB(
                self._att_lut_dB, self._att_lut_elev_deg, user_indices, el
            )
        else:
            # Fallback: on-the-fly computation using global config values
            atm = self.atm
            fso_loss_dB = np.array([
                mie.attenuation_dB(atm.lam_um, atm.hE_km, e)
                + geometric.attenuation_dB(
                    e, atm.LW, atm.N, atm.lam_um, atm.hA_km, atm.hE_km, atm.phi
                )
                for e in el
            ])

        # ── Realistic channel matrix ─────────────────────────────────────────
        loss_scaling = 1.0 / np.sqrt(10 ** (0.1 * fso_loss_dB))
        H_sys = H_ideal * loss_scaling[:, None]               # (U, 1)

        # Zero out non-visible links so downstream rate = 0
        bad = ~slant_ok
        if bad.any():
            H_ideal[bad, :] = 0.0
            H_sys[bad, :]   = 0.0

        # ── FSPL, SNR, Shannon capacity ──────────────────────────────────────
        with np.errstate(divide="ignore"):
            FSPL_dB = 20.0 * np.log10(np.maximum(4.0 * np.pi * d / lam, 1e-30))

        B = self.trm.bandwidth_hz

        return {
            "ts_index":     ts_index,
            "H_ideal":      H_ideal,
            "H_sys":        H_sys,
            "FSPL_dB":      FSPL_dB,
            "A_atm_dB":     fso_loss_dB,
            "SNR_dB":       compute_snr_dB(H_ideal, self.P_eff),
            "SNR_real_dB":  compute_snr_dB(H_sys,   self.P_eff),
            "rate_ideal":   compute_shannon_rate(H_ideal, B, self.P_eff),
            "rate_real":    compute_shannon_rate(H_sys,   B, self.P_eff),
            "slant_m":      d,
        }
