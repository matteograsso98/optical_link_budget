"""
OpticalChannel — dynamic FSO LEO channel model
==============================================
Free-space optical (FSO) channel for a LEO satellite pass.

Physical convention (optical IM/DD):
  We work with optical powers [W], not complex amplitudes. The channel is real
  and positive — there is no carrier phase and no array radiation pattern (a
  single narrow-beam aperture, so the channel is a 1×U vector, not an
  array-element matrix). The amplitude (field) gain is

      H = sqrt(G_T · G_R) · λ / (4π d)         (physical, noise NOT folded in)

  so |H|² = G_T·G_R·(λ/4πd)² is the link power gain, the received signal power
  is P_r = P_eff·|H|², and the SNR is formed explicitly against the receiver
  noise floor N0 in snr.py:  SNR = P_eff·|H|² / N0.

This is the optical counterpart of the RF ``RFChannel`` (rf_channel.py): same
precompute_lut → compute lifecycle, but with the radiation pattern and phase
term removed and ITU-R P.618 replaced by the optical ITU-R P.1622 atmosphere.

Three atmosphere models (atm.atm_model):
  "mie_geom"      — Mie (ITU-R P.1622) + geometric scattering (Kim/Liang)
  "mie_scin"      — Mie + amplitude scintillation (ITU-R P.1622)
  "mie_geom_scin" — Mie + geometric + scintillation (most complete)

Configuration parameters:
  Loaded directly from config.yaml via yaml.safe_load() in main_dynamic.py.
  The atm / orb / trm objects are SimpleNamespaces; no dependency on config.py.
"""

from __future__ import annotations

import os

import numpy as np

from channel_model.atmosphere import mie, geometric, scintillation
from channel_model.link import budget
from channel_model.base import BaseChannel
from channel_model.snr import compute_snr_dB, compute_shannon_rate


class OpticalChannel(BaseChannel):
    """
    Dynamic FSO link channel for a LEO satellite pass.

    Parameters
    ----------
    atm : SimpleNamespace  (fields: atm_model, lam_um, hA_km, Z_m, vrms, C0, LW, N, phi)
    orb : SimpleNamespace  (fields: hS_km)
    trm : SimpleNamespace  (fields: Dr_m, eta_R, theta_R_rad,
                                    noise_dBm_night, noise_dBm_day, noise_condition,
                                    P_tx, eta_T, Theta_T_rad, theta_T_rad, bandwidth_hz)

    All of these are built in main_dynamic.py from config.yaml.

    Usage
    -----
    1. Instantiate with the configuration namespaces.
    2. Call ``precompute_lut(user_hE_km, ...)`` once before the simulation
       loop to build the (U × G) atmospheric LUT.
    3. Call ``compute(ts_index, slant_range_m, az_deg, el_deg)`` each timestep
       to get the channel vectors, SNRs, and Shannon rates.
    """

    def __init__(self, atm, orb, trm) -> None:
        super().__init__()
        self.atm = atm
        self.orb = orb
        self.trm = trm

        self.lam_m   = atm.lam_um * 1e-6
        self.G_R     = budget.receive_gain(trm.Dr_m, atm.lam_um)

        # Receiver noise floor N0 [W] — background-limited, day/night dependent.
        # SNR is computed against this floor (not folded into the channel).
        self.noise_W_night = 10 ** (trm.noise_dBm_night / 10) * 1e-3
        self.noise_W_day   = 10 ** (trm.noise_dBm_day   / 10) * 1e-3
        self.noise_W = (self.noise_W_day if trm.noise_condition == "day"
                        else self.noise_W_night)

        G_T_lin = budget.transmit_gain(trm.Theta_T_rad)
        L_T_dB  = budget.pointing_loss_dB(G_T_lin, trm.theta_T_rad)
        L_R_dB  = budget.pointing_loss_dB(self.G_R,  trm.theta_R_rad)

        self.G_T   = G_T_lin
        # Effective transmit power after terminal optical losses (η_T · L_T · η_R · L_R)
        self.P_eff = (trm.P_tx
                      * trm.eta_T * 10 ** (L_T_dB / 10)
                      * trm.eta_R * 10 ** (L_R_dB / 10))

        # Physical channel amplitude scaling (field gain, noise NOT folded in).
        # H_ideal[u] = xi / d[u]  =>  |H_ideal[u]|² = G_T·G_R·(λ/4πd)²  (power gain).
        # SNR is then P_eff·|H|²/N0, computed explicitly in snr.py.
        self.xi = np.sqrt(self.G_T * self.G_R) * self.lam_m / (4.0 * np.pi)

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
        elevation grid. Optical counterpart of
        RFChannel.precompute_attenuation_lut().

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

        if "scin" in atm_model:
            if user_h0_m is None:
                raise ValueError(
                    f"user_h0_m is required for atm_model='{atm_model}'."
                )
            user_h0_m = np.asarray(user_h0_m, dtype=np.float64)

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
                losses[:, i] += np.array([
                    scintillation.sigma_dB(lam_um, el, user_h0_m[u], atm.Z_m, atm.vrms, atm.C0)
                    for u in range(U)
                ])

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
        Compute the FSO channel vectors, SNR, and Shannon capacity.

        Mirrors RFChannel.compute() for API consistency.

        Parameters
        ----------
        ts_index      : timestep index (forwarded to the output dict).
        slant_range_m : (U,) slant range from ground station to satellite [m].
        az_deg        : (U,) azimuth angle [deg] — not used in FSO (no spatial
                        multiplexing), kept for API symmetry with RFChannel.
        el_deg        : (U,) elevation angle [deg].
        user_indices  : (U,) int row indices into the LUT.
                        Defaults to 0, 1, …, U-1.

        Returns
        -------
        dict with keys:
            ts_index   — echo of ts_index.
            H_ideal    — (U, 1) float64, physical free-space amplitude gain (field gain).
            H_sys      — (U, 1) float64, amplitude gain incl. atmospheric losses.
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

        # ── Ideal FSO amplitude gain ─────────────────────────────────────────
        # In optical IM/DD the channel is real and positive (no carrier phase).
        # H_ideal[u] = xi / d[u],  with xi = sqrt(G_T·G_R)·λ/(4π)  (physical
        # field gain; noise is NOT folded in). The received signal power is
        # P_r = P_eff·|H|², and SNR = P_r / N0 is formed in snr.py.
        slant_ok = self._visibility_mask(d)
        safe_d   = np.where(slant_ok, d, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            H_ideal = (self.xi / safe_d)[:, None]            # (U, 1) real

        # ── FSO atmospheric losses (ITU-R P.1622) ────────────────────────────
        if self._att_lut_dB is None:
            raise RuntimeError(
                "Atmospheric LUT has not been precomputed. "
                "Call precompute_lut() before compute()."
            )
        fso_loss_dB = self._lookup_attenuation_dB(user_indices, el)

        # ── Realistic channel vector ─────────────────────────────────────────
        H_sys = H_ideal * self._amplitude_scaling(fso_loss_dB)[:, None]  # (U, 1)

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
            "SNR_dB":       compute_snr_dB(H_ideal, self.P_eff, self.noise_W),
            "SNR_real_dB":  compute_snr_dB(H_sys,   self.P_eff, self.noise_W),
            "rate_ideal":   compute_shannon_rate(H_ideal, B, self.P_eff, self.noise_W),
            "rate_real":    compute_shannon_rate(H_sys,   B, self.P_eff, self.noise_W),
            "slant_m":      d,
        }