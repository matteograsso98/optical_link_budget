"""
dynamic_link_budget.py
======================
Canal FSO dynamique — adapté de channel.py pour les liaisons optiques LEO.

Différences par rapport à channel.py (RF) :
  - Suppression du fading à grande échelle 3GPP (load_large_scale_model)
    → non pertinent pour FSO optique
  - Suppression ITU-R P.618 (RF)
    → remplacé par ITU-R P.1622 : Mie + Géométrique via LUT FSO
  - Steering vector UPA via antenna.py
  - SNR + Shannon via snr.py
"""

import numpy as np
from optical_link_budget_paper.atmosphere import mie, geometric
from optical_link_budget_paper.link import budget
from dynamic_link_budget.antenna import upa_steering_vector
from dynamic_link_budget.snr import compute_snr_dB, compute_shannon_rate


class DynamicLinkBudget:

    def __init__(self, link_config: dict) -> None:
        self.cfg = link_config

        # ── Paramètres optiques fixes ──────────────────────────────────
        self.lam_m   = link_config["lam_um"] * 1e-6
        self.G_R     = budget.receive_gain(
                           link_config["Dr_m"], link_config["lam_um"])
        noise_dBm    = link_config.get("noise_dBm", -100.0)
        self.noise_W = 10 ** (noise_dBm / 10) * 1e-3

        # ── LUT pertes atmosphériques FSO (ITU-R P.1622) ───────────────
        # Remplace itur P.618 de channel.py
        self._att_lut_dB       = None
        self._att_lut_elev_deg = None

    # ──────────────────────────────────────────────────────────────────
    # LUT ITU-R P.1622 — Mie + Géométrique
    # ──────────────────────────────────────────────────────────────────

    def precompute_lut(self, elevation_grid_deg=None,
                       lut_cache_path=None) -> None:
        """
        Précalcule les pertes atmosphériques FSO (ITU-R P.1622)
        sur une grille d'élévations.

        Remplace precompute_attenuation_lut() de channel.py :
          - channel.py  : itur P.618, dépendance géographique (U x G)
          - ici         : Mie + Géométrique P.1622, 1D (G,)
        """
        if elevation_grid_deg is None:
            elevation_grid_deg = np.arange(10, 90.5, 0.5, dtype=np.float64)

        if lut_cache_path is not None:
            import os
            meta_path = str(lut_cache_path) + ".elev.npy"
            if os.path.exists(lut_cache_path) and os.path.exists(meta_path):
                self._att_lut_dB       = np.load(lut_cache_path)
                self._att_lut_elev_deg = np.load(meta_path)
                print(">>> LUT FSO (P.1622) chargée depuis le cache")
                return

        cfg    = self.cfg
        losses = np.zeros(elevation_grid_deg.size, dtype=np.float64)
        for i, el in enumerate(elevation_grid_deg):
            losses[i] = (
                mie.attenuation_dB(cfg["lam_um"], cfg["hE_km"], el) +
                geometric.attenuation_dB(
                    el, cfg["LW"], cfg["N_droplets"],
                    cfg["lam_um"], cfg["hA_km"],
                    cfg["hE_km"], cfg["phi"])
            )

        self._att_lut_dB       = losses
        self._att_lut_elev_deg = elevation_grid_deg

        if lut_cache_path is not None:
            np.save(lut_cache_path, losses)
            np.save(str(lut_cache_path) + ".elev.npy", elevation_grid_deg)
        print(f">>> LUT FSO (P.1622) calculée : {losses.size} points")

    def _lookup_fso_losses_dB(self, elevation_deg):
        """
        Interpolation 1D dans le LUT.
        Identique à _lookup_attenuation_dB() de channel.py.
        """
        grid   = self._att_lut_elev_deg
        elev   = np.clip(np.asarray(elevation_deg, dtype=np.float64),
                         grid[0], grid[-1])
        idx_f  = (elev - grid[0]) / (grid[1] - grid[0])
        idx_lo = np.floor(idx_f).astype(np.int64)
        idx_hi = np.minimum(idx_lo + 1, grid.size - 1)
        frac   = idx_f - idx_lo
        return ((1.0 - frac) * self._att_lut_dB[idx_lo]
                + frac * self._att_lut_dB[idx_hi])

    # ──────────────────────────────────────────────────────────────────
    # compute() — canal FSO, sans fading 3GPP
    # ──────────────────────────────────────────────────────────────────

    def compute(self, ts_index, slant_range_m, az_deg, el_deg):
        """
        Calcule H_ideal, H_realistic, SNR et débit Shannon.

        Par rapport à channel.py :
          - Supprimé  : large_scale_loss_dB (3GPP TR 38.811)
          - Supprimé  : p618_total_loss_dB  (ITU-R P.618 RF)
          - Remplacé  : fso_loss_dB         (ITU-R P.1622 optique)

        Paramètres
        ----------
        ts_index      : int
        slant_range_m : array (U,) [m]
        az_deg        : array (U,) [deg]
        el_deg        : array (U,) [deg]

        Retourne
        --------
        dict : h_ideal, h_realistic, SNR_dB, SNR_real_dB,
               rate_ideal, rate_real, FSPL_dB, A_atm_dB, slant_m
        """
        lam = self.lam_m
        el  = np.asarray(el_deg,        dtype=np.float64)
        az  = np.asarray(az_deg,        dtype=np.float64)
        d   = np.asarray(slant_range_m, dtype=np.float64)

        # ══════════════════════════════════════════════════════════════
        # 1. H_IDEAL — identique à channel.py
        # ══════════════════════════════════════════════════════════════

        radiation_pattern = upa_steering_vector(
            az, el,
            self.cfg["N_x"], self.cfg["N_y"],
            self.cfg["d_x"], self.cfg["d_y"],
        )

        xi = np.sqrt(self.G_R / self.noise_W) / (4 * np.pi / lam)

        slant_ok = np.isfinite(d) & (d > 0)
        safe_d   = np.where(slant_ok, d, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            phase_path_loss = (
                np.exp(-1j * safe_d * 2*np.pi / lam) / safe_d
            )

        h_ideal = xi * radiation_pattern * phase_path_loss[:, None]

        # ══════════════════════════════════════════════════════════════
        # 2. PERTES ATMOSPHÉRIQUES FSO — ITU-R P.1622
        #    Remplace large_scale_loss + p618 de channel.py
        # ══════════════════════════════════════════════════════════════

        if self._att_lut_dB is not None:
            fso_loss_dB = self._lookup_fso_losses_dB(el)
        else:
            cfg = self.cfg
            fso_loss_dB = np.array([
                mie.attenuation_dB(cfg["lam_um"], cfg["hE_km"], e) +
                geometric.attenuation_dB(
                    e, cfg["LW"], cfg["N_droplets"],
                    cfg["lam_um"], cfg["hA_km"],
                    cfg["hE_km"], cfg["phi"])
                for e in el
            ])

        # ══════════════════════════════════════════════════════════════
        # 3. H_REALISTIC — identique à channel.py
        #    total_loss = fso_loss uniquement (plus de 3GPP)
        # ══════════════════════════════════════════════════════════════

        loss_scaling = 1.0 / np.sqrt(10 ** (0.1 * fso_loss_dB))
        h_realistic  = h_ideal * loss_scaling[:, None]

        bad = ~slant_ok
        if bad.any():
            h_ideal[bad, :]     = 0.0
            h_realistic[bad, :] = 0.0

        # ══════════════════════════════════════════════════════════════
        # 4. SNR ET SHANNON — via snr.py
        # ══════════════════════════════════════════════════════════════

        B = self.cfg.get("bandwidth_hz", 10e9)

        with np.errstate(divide="ignore"):
            FSPL_dB = 20 * np.log10(
                np.maximum(4*np.pi*d / lam, 1e-30)
            )

        return {
            "ts_index":    ts_index,
            "h_ideal":     h_ideal,
            "h_realistic": h_realistic,
            "FSPL_dB":     FSPL_dB,
            "A_atm_dB":    fso_loss_dB,
            "SNR_dB":      compute_snr_dB(h_ideal),
            "SNR_real_dB": compute_snr_dB(h_realistic),
            "rate_ideal":  compute_shannon_rate(h_ideal,     B),
            "rate_real":   compute_shannon_rate(h_realistic, B),
            "slant_m":     d,
        }