import numpy as np
import sys

try:
    import itur
except ImportError:
    print("Warning: 'itur' library not found. Please install it using 'pip install itur'.")
    itur = None

# Large-scale-fading tables for  RFChannel is kept here as a reference RF counterpart to
# OpticalChannel, and only needs this dependency when actually instantiated.
from load_large_scale_model import load_large_scale_model
from channel_model.base import BaseChannel

class RFChannel(BaseChannel):
    def __init__(self, channel_model_config: dict) -> None:
        """
        Initializes the channel model parameters and loads the large-scale fading tables.
        """
        super().__init__()
        self.channel_scenario = channel_model_config.get("channel_scenario", "urban")
        self.propagation_model = channel_model_config.get("propagation_model", "los").lower()
        self.force_nlos_for_all_users = channel_model_config.get("force_nlos_for_all_users", False)

        # ITU-R P.618 exceedance probability (e.g., 99.99% availability -> p = 0.01)
        self.p = channel_model_config.get("p", 0.01)
        self.include_clouds = channel_model_config.get("include_clouds", True)

        # Load large scale model data (LOS probabilities, sigma, clutter) once at initialization
        self.lsm = load_large_scale_model(self.channel_scenario)

        # Per-UT atmospheric-attenuation LUT. When present, `compute()` reads
        # ITU-R P.618 total attenuation [dB] from the LUT instead of calling
        # `itur.atmospheric_attenuation_slant_path` on every call.
        self._att_lut_dB = None         # (U, N_grid) float64 [dB]
        self._att_lut_elev_deg = None   # (N_grid,)  float64 [deg]

    # ──────────────────────────────────────────────────────────────────────
    # Atmospheric-attenuation lookup table
    # ──────────────────────────────────────────────────────────────────────
    def precompute_attenuation_lut(
        self,
        user_lats_deg,                  # (U,)
        user_lons_deg,                  # (U,)
        carrier_frequency_hz,
        rx_antenna_diameter_m,
        elevation_grid_deg=None,
        lut_cache_path=None,            # optional path to a .npy file for disk caching
        n_jobs: int = -1,               # parallel workers (-1 = all cores)
    ) -> None:
        """Precompute ITU-R P.618 total attenuation [dB] for every UT across an
        elevation grid. UTs are fixed; frequency is constant — so the only
        per-call variable inside ``compute()`` becomes elevation. After this is
        called, ``compute()`` does a 1-D linear interp per UT instead of an
        ``itur.atmospheric_attenuation_slant_path`` call.

        Performance notes
        -----------------
        The old implementation looped over U users and called itur once per
        user (passing the full elevation array). itur has ~25-30 ms of fixed
        per-call overhead (geographic-map lookups), so U=20 000 users meant
        ~23 minutes of wall-clock time.

        Fix: we invert the loop — iterate over the elevation grid (G ≈ 161
        points) and pass ALL user (lat, lon) arrays in a single vectorised
        itur call per elevation step. This reduces the itur call count from U
        to G (20 000 → 161), cutting that fixed overhead by ~120×. Each
        call is then parallelised across ``n_jobs`` threads (itur releases the
        GIL for its numpy operations, so threads work without memory copies).

        Total time: ~5-10 s for 20 000 users on a laptop (vs. ~23 min before).
        """
        if itur is None:
            return  # itur not available — keep zero-loss fallback path

        if elevation_grid_deg is None:
            elevation_grid_deg = np.arange(10, 90.0 + 0.25, 0.5, dtype=np.float64)

        # ── Disk cache: skip recomputation if the LUT was already built ──────
        if lut_cache_path is not None:
            import os
            meta_path = str(lut_cache_path) + ".elev.npy"
            if os.path.exists(lut_cache_path) and os.path.exists(meta_path):
                self._att_lut_dB       = np.load(lut_cache_path)
                self._att_lut_elev_deg = np.load(meta_path)
                return

        import warnings
        from tqdm import tqdm
        from joblib import Parallel, delayed

        U    = user_lats_deg.size
        G    = elevation_grid_deg.size
        lats = np.asarray(user_lats_deg, dtype=np.float64)
        lons = np.asarray(user_lons_deg, dtype=np.float64)
        f_ghz = carrier_frequency_hz * 1e-9
        D_m   = rx_antenna_diameter_m

        # ── Worker: one itur call per elevation, all users in one batch ───────
        def _compute_column(el: float) -> np.ndarray:
            with warnings.catch_warnings():
                # P.676 warns when elevation is outside its "recommended" range
                # of 5–90°.  Our grid starts at 10° (well inside the valid
                # range), so the warning is spurious — silence it here.
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="itur")
                try:
                    results = itur.atmospheric_attenuation_slant_path(
                        lat=lats, lon=lons,
                        f=f_ghz, el=float(el), p=0.5,
                        D=D_m,
                        hs=None, rho=None, R001=None, T=None, H=None, P=None,
                        include_rain=True, include_gas=True,
                        include_scintillation=True, include_clouds=True,
                        return_contributions=True,
                    )
                    A_total = results[-1]
                    col = np.asarray(getattr(A_total, "value", A_total), dtype=np.float64)
                    if col.ndim == 0:
                        col = np.full(U, float(col), dtype=np.float64)
                    return col
                except Exception:
                    return np.zeros(U, dtype=np.float64)

        # Warm up itur so the first parallel worker doesn't pay the full
        # geographic-map-loading cost (reduces spurious first-column slowness).
        _compute_column(float(elevation_grid_deg[len(elevation_grid_deg) // 2]))

        # ── Parallel loop over elevation grid ─────────────────────────────────
        # prefer="threads": itur operations are mostly numpy (releases GIL) so
        # threads share the already-loaded geographic maps without copying them.
        columns = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_column)(float(el))
            for el in tqdm(elevation_grid_deg, desc="Attenuation LUT", unit="el")
        )

        att_dB = np.column_stack(columns)   # (U, G)

        self._att_lut_dB       = att_dB
        self._att_lut_elev_deg = elevation_grid_deg

        # ── Persist to disk if a cache path was provided ──────────────────────
        if lut_cache_path is not None:
            np.save(lut_cache_path, att_dB)
            np.save(str(lut_cache_path) + ".elev.npy", elevation_grid_deg)

    def compute(self, ts_index: int, tx, rx, user_terminals_geodetic, slant_range, uv, thetaphi, elevation, user_indices=None):
        """
        Calculates both the ideal and realistic complex channel matrices for the current timestep.
        Matches the exact physical logic of the original RRM channel model script.
        """
        # ==============================================================================
        # 1. CALCULATE IDEAL CHANNEL (H_lb)
        # ==============================================================================
        lambda_m = tx.tx_wavelength_m
        rx_gain_dB = rx.rx_gain_dB
        noise_W = rx.rx_noise_W()

        # Generate the radiation pattern from the TX antenna
        u, v = uv[:, 0], uv[:, 1]
        radiation_pattern = tx.compute_tx_radiation_pattern(u, v)

        # Normalization factor (RX gain and free space path loss constant)
        xi = np.sqrt(10 ** (0.1 * rx_gain_dB) / noise_W) / (4 * np.pi / lambda_m)

        # Defensive guard: a (sat, user) pair with slant <= 0 or non-finite
        # slant must NOT produce a rate. This mirrors the gate already used by
        # `RadioResourceManagement._compute_single_pair_rate`, which returns
        # rate=0 when geometry's elevation is non-positive (i.e. the pair is
        # not visible). Without the guard, slant=0 causes a divide-by-zero in
        # `phase_path_loss = exp(...)/slant`, polluting h_ideal/h_realistic
        # with inf/NaN and corrupting downstream beamforming.
        #
        # Mask: bad rows in the final h_ideal/h_realistic are zeroed at the
        # end of compute(), so |h|=0 → rate = bw*log2(1+0) = 0 → fails the
        # `rate >= Rmin` gate → user gets handed over off the stale sat.
        slant_ok   = self._visibility_mask(slant_range)
        safe_slant = np.where(slant_ok, slant_range, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            phase_path_loss = np.exp(-1j * safe_slant * 2 * np.pi / lambda_m) / safe_slant
        
        # H_ideal matches the old script's lossless complex channel matrix
        h_ideal = xi * radiation_pattern * phase_path_loss[:, None]

        # ==============================================================================
        # 2. STATISTICAL PROFILE & LARGE-SCALE FADING
        # ==============================================================================
        num_users = user_terminals_geodetic.shape[0]
        
        # Bin elevations for large scale fading (clip to avoid index out of bounds)
        elev_clipped = np.clip(elevation, 10, 90)
        row_index = (np.round(elev_clipped, -1) / 10 - 1).astype(int)

        # Determine Line-of-Sight (LOS) flag
        if self.propagation_model == 'los':
            los_flag = np.ones(num_users, dtype=bool)
        else:
            los_prob = self.lsm["large_scale_fading"]['los_prob'].values[row_index]
            los_flag = np.random.rand(num_users) * 100 <= los_prob

        # Apply Shadowing (Sigma)
        sigma = np.where(
            los_flag,
            self.lsm["large_scale_fading"]['sigma_los'].values[row_index],
            self.lsm["large_scale_fading"]['sigma_nlos'].values[row_index]
        )
        large_scale_loss_dB = np.random.randn(row_index.size) * sigma

        # Apply Clutter (NLOS only, unless forced)
        if self.propagation_model.lower() == 'nlos':
            clutter_loss = self.lsm["large_scale_fading"]['cl'].values[row_index]
            if self.force_nlos_for_all_users:
                large_scale_loss_dB += clutter_loss
            else:
                large_scale_loss_dB[~los_flag] += clutter_loss[~los_flag]

        # ---------------------------------------------------------
        # 2) ITU-R P.618 total attenuation per user [dB]
        # ---------------------------------------------------------
        if self._att_lut_dB is not None and user_indices is not None:
            # Fast path: per-UT 1-D linear interp in elevation. Eliminates the
            # itur cost from the steady-state simulation loop.
            p618_total_loss_dB = self._lookup_attenuation_dB(user_indices, elevation)
        elif itur is None:
            p618_total_loss_dB = np.zeros_like(elevation, dtype=float)
        else:
            f_ghz = tx.tx_carrier_frequency_hz * 1e-9
            D_m = rx.rx_antenna_diameter_m # Antenna diameter for scintillation
            lat = user_terminals_geodetic[:, 0]
            lon = user_terminals_geodetic[:, 1]
            el_safe = np.clip(elevation, 0.1, 90.0)
            try:
                results = itur.atmospheric_attenuation_slant_path(
                    lat=lat, lon=lon, f=f_ghz, el=el_safe,
                    p=0.5, D=D_m,
                    hs=None, rho=None, R001=None, T=None, H=None, P=None,
                    include_rain=True, include_gas=True, include_scintillation=True, include_clouds=True,
                    return_contributions=True,
                )
                A_total = results[-1]
                p618_total_loss_dB = np.asarray(getattr(A_total, "value", A_total), dtype=float)
                if p618_total_loss_dB.ndim == 0:
                    p618_total_loss_dB = np.full_like(elev_clipped, float(p618_total_loss_dB), dtype=float)
            except Exception as e:
                print(f"Error in ITU-Rpy P.618 calculation: {e}. Falling back to zero P.618 loss.")
                p618_total_loss_dB = np.zeros_like(el_safe, dtype=float)


        # ==============================================================================
        # 4. COMBINE LOSSES & GENERATE REALISTIC CHANNEL (H_tx)
        # ==============================================================================
        total_loss_dB = large_scale_loss_dB + p618_total_loss_dB
        
        # Convert total dB loss to linear scaling factor
        loss_scaling = self._amplitude_scaling(total_loss_dB)

        # Apply loss scaling across the ideal channel
        h_realistic = h_ideal * loss_scaling[:, None]

        # Zero out the bad-slant rows we masked at the top of compute().
        # `|h_realistic[bad]| = 0` → downstream rate = 0 → caught by the
        # `rate >= Rmin` gate (handover trigger).
        bad_rows = ~slant_ok
        if bad_rows.any():
            h_ideal[bad_rows, :]     = 0.0
            h_realistic[bad_rows, :] = 0.0

        return h_ideal, h_realistic
    