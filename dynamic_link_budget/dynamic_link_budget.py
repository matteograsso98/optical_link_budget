"""
dynamic_link_budget.py
======================
Canal FSO dynamique — adapté de channel.py pour les liaisons optiques LEO.

Différences par rapport à channel.py (RF) :
  - Suppression du fading à grande échelle 3GPP (load_large_scale_model)
    → non pertinent pour FSO optique
  - Suppression ITU-R P.618 (RF)
    → remplacé par ITU-R P.1622 : Mie + Géométrique/Scintillation via LUT FSO
  - LUT 2D (U, G) : une ligne par utilisateur, vectorisé sur l'élévation
  - Steering vector UPA via antenna.py
  - SNR + Shannon via snr.py
"""

import numpy as np
from optical_link_budget_paper.atmosphere import mie, geometric, scintillation
from optical_link_budget_paper.link import budget
from dynamic_link_budget.antenna import upa_steering_vector
from dynamic_link_budget.snr import compute_snr_dB, compute_shannon_rate


class DynamicLinkBudget:

    def __init__(self, link_config: dict) -> None:
        self.cfg = link_config

        # atm_model : "mie_geom" | "mie_scin" | "mie_geom_scin"
        self.atm_model = link_config.get("atm_model", "mie_geom")

        # ── Paramètres optiques fixes ──────────────────────────────────
        self.lam_m   = link_config["lam_um"] * 1e-6
        self.G_R     = budget.receive_gain(
                           link_config["Dr_m"], link_config["lam_um"])
        noise_dBm    = link_config.get("noise_dBm", -100.0)
        self.noise_W = 10 ** (noise_dBm / 10) * 1e-3
        self.P_tx    = link_config.get("P_tx", 1.0)

        # ── LUT pertes atmosphériques FSO — (U, G) ─────────────────────
        self._att_lut_dB       = None   # (U, G) float64 [dB]
        self._att_lut_elev_deg = None   # (G,)   float64 [deg]
        self._scin_coeff       = None   # pré-facteur scintillation
        self._scin_dBfact      = None   # conversion variance → dB

    # ──────────────────────────────────────────────────────────────────
    # LUT ITU-R P.1622 — 2D (utilisateurs × élévation)
    # ──────────────────────────────────────────────────────────────────

    def precompute_lut(
        self,
        user_hE_km,
        user_h0_m=None,
        elevation_grid_deg=None,
        lut_cache_path=None,
    ) -> None:
        """
        Précalcule les pertes atmosphériques FSO pour chaque utilisateur
        sur une grille d'élévations.

        LUT résultante : (U, G) float64 [dB]
          U = nombre d'utilisateurs  (longueur de user_hE_km)
          G = points de la grille d'élévation

        Paramètres
        ----------
        user_hE_km : array (U,)
            Altitude au-dessus du MSL de chaque utilisateur [km].
            Utilisé par les modèles Mie et géométrique.
        user_h0_m : array (U,), optionnel
            Hauteur de la station au-dessus du sol local [m].
            Requis pour atm_model ∈ {"mie_scin", "mie_geom_scin"}.
        elevation_grid_deg : array, optionnel
            Grille d'élévation en degrés (défaut : 10 à 90 par 0.5°).
        lut_cache_path : str, optionnel
            Chemin .npy pour la mise en cache sur disque.

        Modèles atmosphériques (cfg["atm_model"]) :
          "mie_geom"      — Mie + diffusion géométrique (Kim, Liang et al.)
          "mie_scin"      — Mie + scintillation (ITU-R P.1622)
          "mie_geom_scin" — Mie + géométrique + scintillation
        """
        user_hE_km = np.asarray(user_hE_km, dtype=np.float64)
        U = user_hE_km.size

        if elevation_grid_deg is None:
            elevation_grid_deg = np.arange(10, 90.5, 0.5, dtype=np.float64)
        G = elevation_grid_deg.size

        # ── Cache disque ─────────────────────────────────────────────────────
        if lut_cache_path is not None:
            import os
            meta_path = str(lut_cache_path) + ".elev.npy"
            hE_path   = str(lut_cache_path) + ".hE.npy"
            if (os.path.exists(lut_cache_path)
                    and os.path.exists(meta_path)
                    and os.path.exists(hE_path)):
                cached_hE = np.load(hE_path)
                if np.allclose(cached_hE, user_hE_km):
                    self._att_lut_dB       = np.load(lut_cache_path)
                    self._att_lut_elev_deg = np.load(meta_path)
                    print(f">>> LUT FSO (P.1622) chargée depuis le cache : "
                          f"{self._att_lut_dB.shape}")
                    return
                print(">>> Cache LUT invalidé (user_hE_km a changé) — recalcul.")

        cfg       = self.cfg
        lam_um    = cfg["lam_um"]
        atm_model = self.atm_model
        LW        = cfg["LW"]
        N_drop    = cfg["N_droplets"]
        hA_km     = cfg["hA_km"]
        phi       = cfg["phi"]

        # ── Pré-calcul des intégrales de scintillation (une par utilisateur) ──
        # L'intégrale ∫_{h0}^{Z} Cn²(h)·h^(5/6) dh est indépendante de
        # l'élévation → on l'évalue une seule fois par utilisateur, puis on
        # applique le facteur 1/sin^(11/6)(el) vectorisé sur toute la grille.
        scin_integrals = None
        if atm_model in ("mie_scin", "mie_geom_scin"):
            if user_h0_m is None:
                raise ValueError(
                    "user_h0_m est requis pour atm_model "
                    f"'{atm_model}'."
                )
            user_h0_m = np.asarray(user_h0_m, dtype=np.float64)
            Z_m  = cfg.get("Z_m",  20_000.0)
            C0   = cfg.get("C0",   1.7e-14)
            vrms = cfg.get("vrms", 21.0)

            scin_integrals = np.empty(U, dtype=np.float64)
            for u in range(U):
                h = np.linspace(user_h0_m[u], Z_m, 80_000)
                integrand = scintillation.Cn2_profile(h, C0=C0, vrms=vrms) * h ** (5.0 / 6.0)
                scin_integrals[u] = np.trapz(integrand, h)

            k = 2.0 * np.pi / (lam_um * 1e-6)
            self._scin_coeff  = 2.253 * k ** (7.0 / 6.0)
            self._scin_dBfact = (10.0 / np.log(10.0)) ** 2

        # ── Construction de la LUT (U, G) ─────────────────────────────────────
        losses = np.zeros((U, G), dtype=np.float64)

        for i, el in enumerate(elevation_grid_deg):

            # Mie — hE_km varie par utilisateur
            losses[:, i] = np.array(
                [mie.attenuation_dB(lam_um, hE, el) for hE in user_hE_km]
            )

            if atm_model in ("mie_geom", "mie_geom_scin"):
                losses[:, i] += np.array(
                    [geometric.attenuation_dB(el, LW, N_drop, lam_um, hA_km, hE, phi)
                     for hE in user_hE_km]
                )

            if atm_model in ("mie_scin", "mie_geom_scin"):
                # sigma²_lnN = coeff · (1/sin el)^(11/6) · integral(u)
                sin_el     = np.sin(np.radians(el))
                sigma2_lnN = (self._scin_coeff
                              * (1.0 / sin_el) ** (11.0 / 6.0)
                              * scin_integrals)
                losses[:, i] += np.sqrt(self._scin_dBfact * sigma2_lnN)

        self._att_lut_dB       = losses
        self._att_lut_elev_deg = elevation_grid_deg

        if lut_cache_path is not None:
            np.save(lut_cache_path, losses)
            np.save(str(lut_cache_path) + ".elev.npy", elevation_grid_deg)
            np.save(str(lut_cache_path) + ".hE.npy",   user_hE_km)
        print(f">>> LUT FSO (P.1622 · {atm_model}) calculée : {losses.shape}")

    def _lookup_fso_losses_dB(self, user_indices, elevation_deg):
        """
        Interpolation 1-D par utilisateur dans la LUT (U, G).
        Identique à channel.py :: _lookup_attenuation_dB.
        """
        grid   = self._att_lut_elev_deg
        elev   = np.clip(np.asarray(elevation_deg, dtype=np.float64),
                         grid[0], grid[-1])
        idx_f  = (elev - grid[0]) / (grid[1] - grid[0])
        idx_lo = np.floor(idx_f).astype(np.int64)
        idx_hi = np.minimum(idx_lo + 1, grid.size - 1)
        frac   = idx_f - idx_lo
        ui     = np.asarray(user_indices, dtype=np.int64)
        return ((1.0 - frac) * self._att_lut_dB[ui, idx_lo]
                + frac        * self._att_lut_dB[ui, idx_hi])

    # ──────────────────────────────────────────────────────────────────
    # compute() — canal FSO, sans fading 3GPP
    # ──────────────────────────────────────────────────────────────────

    def compute(self, ts_index, slant_range_m, az_deg, el_deg,
                user_indices=None):
        """
        Calcule H_ideal, H_realistic, SNR et débit Shannon.

        Paramètres
        ----------
        ts_index      : int
        slant_range_m : array (U,) [m]
        az_deg        : array (U,) [deg]
        el_deg        : array (U,) [deg]
        user_indices  : array (U,) int, optionnel
            Indices des utilisateurs dans la LUT.
            Défaut : 0, 1, …, U-1 (tous les utilisateurs dans l'ordre).
            Utile quand on simule un sous-ensemble d'utilisateurs actifs.

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

        if user_indices is None:
            user_indices = np.arange(U, dtype=np.int64)

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
        #    LUT (U, G) : chaque utilisateur a son propre profil
        # ══════════════════════════════════════════════════════════════

        if self._att_lut_dB is not None:
            fso_loss_dB = self._lookup_fso_losses_dB(user_indices, el)
        else:
            # Fallback : calcul à la volée avec paramètres globaux
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
        # 3. H_REALISTIC
        # ══════════════════════════════════════════════════════════════

        loss_scaling = 1.0 / np.sqrt(10 ** (0.1 * fso_loss_dB))
        h_realistic  = h_ideal * loss_scaling[:, None]

        bad = ~slant_ok
        if bad.any():
            h_ideal[bad, :]     = 0.0
            h_realistic[bad, :] = 0.0

        # ══════════════════════════════════════════════════════════════
        # 4. SNR ET SHANNON
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
            "SNR_dB":      compute_snr_dB(h_ideal,     self.P_tx),
            "SNR_real_dB": compute_snr_dB(h_realistic, self.P_tx),
            "rate_ideal":  compute_shannon_rate(h_ideal,     B, self.P_tx),
            "rate_real":   compute_shannon_rate(h_realistic, B, self.P_tx),
            "slant_m":     d,
        }
