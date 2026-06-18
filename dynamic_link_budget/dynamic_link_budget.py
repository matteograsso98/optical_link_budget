"""
DynamicLinkBudget — bilan de liaison FSO LEO dynamique  (V2)
=============================================================
Modèle de canal optique FSO pour un passage de satellite LEO.

Convention physique (optique IM/DD) :
  On travaille avec des puissances optiques [W], pas des amplitudes complexes.
  Le canal est réel et positif : gain d'amplitude h = sqrt(G_T · G_R) · λ/(4πd).
  SNR = P_eff · |h|² / N₀  =  P_r / N₀   (rapport de puissances optiques).

  xi = sqrt(G_T · G_R / N₀) · λ/(4π)  absorbé dès l'initialisation,
  de sorte que  H_ideal[u] = xi / d[u]  et  |H_ideal|² = G_T·G_R·(λ/4πd)²/N₀.

Trois modèles d'atmosphère (atm.atm_model) :
  "mie_geom"      — Mie (ITU-R P.1622) + diffusion géométrique (Kim/Liang)
  "mie_scin"      — Mie + scintillation d'amplitude (ITU-R P.1622)
  "mie_geom_scin" — Mie + géométrique + scintillation (modèle le plus complet)

Paramètres de configuration :
  Chargés directement depuis config.yaml via yaml.safe_load() dans main_dynamic.py.
  Les objets atm / orb / trm sont des SimpleNamespace ; aucune dépendance à config.py.
"""

from __future__ import annotations

import os

import numpy as np

from optical_link_budget_paper.atmosphere import mie, geometric, scintillation
from optical_link_budget_paper.link import budget
from channel_model.lut_interp import lookup_lut_dB
from dynamic_link_budget.snr import compute_snr_dB, compute_shannon_rate


class DynamicLinkBudget:
    """
    Bilan de liaison FSO dynamique pour un passage de satellite LEO.

    Paramètres
    ----------
    atm : SimpleNamespace  (champs : atm_model, lam_um, hA_km, Z_m, vrms, C0, LW, N, phi)
    orb : SimpleNamespace  (champs : hS_km)
    trm : SimpleNamespace  (champs : Dr_m, eta_R, theta_R_rad, noise_dBm,
                                     P_tx, eta_T, Theta_T_rad, theta_T_rad, bandwidth_hz)

    Tous ces objets sont créés dans main_dynamic.py à partir de config.yaml.

    Utilisation
    -----------
    1. Instancier avec les namespaces de configuration.
    2. Appeler ``precompute_lut(user_hE_km, ...)`` une fois avant la boucle
       de simulation pour construire la LUT atmosphérique (U × G).
    3. Appeler ``compute(ts_index, slant_range_m, el_deg)`` à chaque
       pas de temps pour obtenir les SNR et débits Shannon.
    """

    def __init__(self, atm, orb, trm) -> None:
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
        # Effective transmit power after terminal optical losses (η_T · L_T · η_R · L_R)
        self.P_eff = (trm.P_tx
                      * trm.eta_T * 10 ** (L_T_dB / 10)
                      * trm.eta_R * 10 ** (L_R_dB / 10))

        # Amplitude channel scaling constant (noise-normalised, G_T and G_R both included).
        # |H_ideal[u]|² = xi² / d[u]²  =>  SNR = P_eff · xi²/d² = P_eff·G_T·G_R·(λ/4πd)²/N₀
        self.xi = (np.sqrt(self.G_T * self.G_R / self.noise_W)
                   / (4.0 * np.pi / self.lam_m))

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
        el_deg,
        user_indices=None,
    ) -> dict:
        """
        Compute FSO channel matrices, SNR, and Shannon capacity.

        Parameters
        ----------
        ts_index      : timestep index (forwarded to the output dict).
        slant_range_m : (U,) slant range from ground station to satellite [m].
        el_deg        : (U,) elevation angle [deg].
        user_indices  : (U,) int row indices into the LUT.
                        Defaults to 0, 1, …, U-1.

        Returns
        -------
        dict with keys:
            ts_index   — echo of ts_index.
            H_ideal    — (U, 1) float64, gain d'amplitude espace libre [m⁻¹·W^{-1/2}].
            H_sys      — (U, 1) float64, gain d'amplitude avec pertes atmosphériques.
            FSPL_dB    — (U,)  perte d'espace libre [dB].
            A_atm_dB   — (U,)  atténuation atmosphérique totale [dB].
            SNR_dB     — (U,)  SNR idéal [dB].
            SNR_real_dB— (U,)  SNR réaliste avec pertes atmosphériques [dB].
            rate_ideal — (U,)  capacité de Shannon, canal idéal [bps].
            rate_real  — (U,)  capacité de Shannon, canal réaliste [bps].
            slant_m    — (U,)  distance oblique [m].
        """
        lam = self.lam_m
        el  = np.asarray(el_deg,        dtype=np.float64)
        d   = np.asarray(slant_range_m, dtype=np.float64)
        U   = el.size

        if user_indices is None:
            user_indices = np.arange(U, dtype=np.int64)

        # ── Gain d'amplitude FSO idéal ───────────────────────────────────────
        # En optique IM/DD le canal est réel et positif (pas de phase porteuse).
        # H_ideal[u] = xi / d[u],  avec xi = sqrt(G_T·G_R/N₀)·λ/(4π).
        # Ainsi  SNR_ideal = P_eff · H_ideal² = P_eff·G_T·G_R·(λ/4πd)²/N₀ = P_r/N₀.
        slant_ok = np.isfinite(d) & (d > 0)
        safe_d   = np.where(slant_ok, d, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            H_ideal = (self.xi / safe_d)[:, None]            # (U, 1) réel

        # ── Pertes atmosphériques FSO (ITU-R P.1622) ────────────────────────
        if self._att_lut_dB is None:
            raise RuntimeError(
                "La LUT atmosphérique n'a pas été précalculée. "
                "Appeler precompute_lut() avant compute()."
            )
        fso_loss_dB = lookup_lut_dB(
            self._att_lut_dB, self._att_lut_elev_deg, user_indices, el
        )

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
