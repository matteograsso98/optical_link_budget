"""
dynamic_link_budget.py
======================
Canal FSO dynamique — structure calquée sur channel.py.

Différences par rapport à channel.py :
  - Pertes atmosphériques : Mie + Géométrique (LUT FSO) au lieu de ITU-R P.618
  - Steering vector UPA   : via antenna.py au lieu de l'objet tx
  - SNR + Shannon         : via snr.py (ajout)
  - Plots                 : via plots.py (ajout)
Le reste (H_ideal, H_realistic, large-scale fading 3GPP) est identique.
"""

import numpy as np

from optical_link_budget_paper.atmosphere import mie, geometric
from optical_link_budget_paper.link import budget
from channel_model.load_large_scale_model import load_large_scale_model
from dynamic_link_budget.antenna import upa_steering_vector
from dynamic_link_budget.snr import compute_snr_dB, compute_shannon_rate


class DynamicLinkBudget:

    def __init__(self, link_config: dict) -> None:
        self.cfg = link_config

        # ── Paramètres optiques fixes ──────────────────────────────────
        self.lam_m   = link_config["lam_um"] * 1e-6
        self.G_R     = budget.receive_gain(link_config["Dr_m"],
                                           link_config["lam_um"])
        noise_dBm    = link_config.get("noise_dBm", -100.0)
        self.noise_W = 10 ** (noise_dBm / 10) * 1e-3

        # ── Large-scale fading 3GPP TR 38.811 — identique à channel.py ─
        self.lsm = load_large_scale_model(
            link_config.get("channel_scenario", "urban")
        )

        # ── LUT pertes atmosphériques FSO — remplace itur ──────────────
        self._att_lut_dB       = None
        self._att_lut_elev_deg = None

    # ──────────────────────────────────────────────────────────────────
    # LUT FSO — remplace precompute_attenuation_lut() de channel.py
    # ──────────────────────────────────────────────────────────────────

    def precompute_lut(self, elevation_grid_deg=None, lut_cache_path=None):
        """Précalcule Mie + Géométrique sur grille d'élévations."""
        if elevation_grid_deg is None:
            elevation_grid_deg = np.arange(10, 90.5, 0.5, dtype=np.float64)

        if lut_cache_path is not None:
            import os
            meta_path = str(lut_cache_path) + ".elev.npy"
            if os.path.exists(lut_cache_path) and os.path.exists(meta_path):
                self._att_lut_dB       = np.load(lut_cache_path)
                self._att_lut_elev_deg = np.load(meta_path)
                print(">>> LUT FSO chargée depuis le cache")
                return

        cfg    = self.cfg
        losses = np.zeros(elevation_grid_deg.size, dtype=np.float64)
        for i, el in enumerate(elevation_grid_deg):
            losses[i] = (
                mie.attenuation_dB(cfg["lam_um"], cfg["hE_km"], el) +
                geometric.attenuation_dB(el, cfg["LW"], cfg["N_droplets"],
                    cfg["lam_um"], cfg["hA_km"], cfg["hE_km"], cfg["phi"])
            )

        self._att_lut_dB       = losses
        self._att_lut_elev_deg = elevation_grid_deg

        if lut_cache_path is not None:
            np.save(lut_cache_path, losses)
            np.save(str(lut_cache_path) + ".elev.npy", elevation_grid_deg)
        print(f">>> LUT FSO calculée : {losses.size} points")

    def _lookup_fso_losses_dB(self, elevation_deg):
        """Interpolation 1D dans le LUT — identique à channel.py."""
        grid   = self._att_lut_elev_deg
        elev   = np.clip(np.asarray(elevation_deg, dtype=np.float64),
                         grid[0], grid[-1])
        idx_f  = (elev - grid[0]) / (grid[1] - grid[0])
        idx_lo = np.floor(idx_f).astype(np.int64)
        idx_hi = np.minimum(idx_lo + 1, grid.size - 1)
        frac   = idx_f - idx_lo
        return (1.0 - frac)*self._att_lut_dB[idx_lo] + frac*self._att_lut_dB[idx_hi]

    # ──────────────────────────────────────────────────────────────────
    # compute() — identique à channel.py sauf pertes itur → FSO
    # ──────────────────────────────────────────────────────────────────

    def compute(self, ts_index, slant_range_m, az_deg, el_deg):
        """
        Calcule H_ideal, H_realistic, SNR et débit Shannon.

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
        U   = el.size

        # ══════════════════════════════════════════════════════════════
        # 1. H_IDEAL — identique à channel.py
        # ══════════════════════════════════════════════════════════════

        # Steering vector UPA via antenna.py
        # remplace tx.compute_tx_radiation_pattern(u, v)
        radiation_pattern = upa_steering_vector(
            az, el,
            self.cfg["N_x"], self.cfg["N_y"],
            self.cfg["d_x"], self.cfg["d_y"],
        )

        # ξ = sqrt(G_R / noise_W) / (4π/λ)
        xi = np.sqrt(self.G_R / self.noise_W) / (4 * np.pi / lam)

        # Garde slant invalide
        slant_ok = np.isfinite(d) & (d > 0)
        safe_d   = np.where(slant_ok, d, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            phase_path_loss = (
                np.exp(-1j * safe_d * 2*np.pi / lam) / safe_d
            )

        h_ideal = xi * radiation_pattern * phase_path_loss[:, None]

        # ══════════════════════════════════════════════════════════════
        # 2. LARGE-SCALE FADING — identique à channel.py
        # ══════════════════════════════════════════════════════════════

        elev_clipped = np.clip(el, 10, 90)
        row_index    = (np.round(elev_clipped, -1) / 10 - 1).astype(int)

        prop_model = self.cfg.get("propagation_model", "los")
        if prop_model == "los":
            los_flag = np.ones(U, dtype=bool)
        else:
            los_prob = self.lsm["large_scale_fading"]["los_prob"].values[row_index]
            los_flag = np.random.rand(U) * 100 <= los_prob

        sigma = np.where(
            los_flag,
            self.lsm["large_scale_fading"]["sigma_los"].values[row_index],
            self.lsm["large_scale_fading"]["sigma_nlos"].values[row_index],
        )
        large_scale_loss_dB = np.random.randn(U) * sigma

        if prop_model == "nlos":
            clutter = self.lsm["large_scale_fading"]["cl"].values[row_index]
            large_scale_loss_dB[~los_flag] += clutter[~los_flag]

        # ══════════════════════════════════════════════════════════════
        # 3. PERTES FSO — remplace ITU-R P.618 de channel.py
        # ══════════════════════════════════════════════════════════════

        if self._att_lut_dB is not None:
            fso_loss_dB = self._lookup_fso_losses_dB(el)
        else:
            cfg = self.cfg
            fso_loss_dB = np.array([
                mie.attenuation_dB(cfg["lam_um"], cfg["hE_km"], e) +
                geometric.attenuation_dB(e, cfg["LW"], cfg["N_droplets"],
                    cfg["lam_um"], cfg["hA_km"], cfg["hE_km"], cfg["phi"])
                for e in el
            ])

        # ══════════════════════════════════════════════════════════════
        # 4. H_REALISTIC — identique à channel.py
        # ══════════════════════════════════════════════════════════════

        total_loss_dB = large_scale_loss_dB + fso_loss_dB
        loss_scaling  = 1.0 / np.sqrt(10 ** (0.1 * total_loss_dB))
        h_realistic   = h_ideal * loss_scaling[:, None]

        bad = ~slant_ok
        if bad.any():
            h_ideal[bad, :]     = 0.0
            h_realistic[bad, :] = 0.0

        # ══════════════════════════════════════════════════════════════
        # 5. SNR ET SHANNON — via snr.py (ajout par rapport à channel.py)
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