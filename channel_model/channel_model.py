import numpy as np
import os
import sys

try:
    import itur
    from astropy import units as u
except ImportError:
    # Fallback if running isolated or itur is missing
    print("Warning: 'itur' library not found. Please install it using 'pip install itur'.")
    pass

# ==============================================================================
# SECTION 1: SHARED UTILITIES
# ==============================================================================

def get_frequency_band(carrier_hz):
    """Determines the frequency band ('S' or 'Ka') from the carrier frequency."""
    carrier_ghz = carrier_hz * 1e-9
    if 2 <= carrier_ghz <= 4:
        return 'S'
    elif 18 <= carrier_ghz <= 31:
        return 'Ka'
    else:
        # Default to Ka for frequencies outside the main S-band range
        return 'Ka' 

def euclidean3d_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Euclidean distance between all columns of A and B.
    Used by Handover module.
    """
    if A.shape[0] != 3 or B.shape[0] != 3:
        raise ValueError("Input matrices must have 3 rows for x, y, z coordinates.")
    
    A_reshaped = A[:, :, np.newaxis]
    B_reshaped = B[:, np.newaxis, :]
    
    return np.sqrt(np.sum((A_reshaped - B_reshaped)**2, axis=0))

# ==============================================================================
# SECTION 2: RRM CHANNEL MODEL
# ==============================================================================

def initialize_channel(config: dict, users: dict, rp: np.ndarray) -> dict:
    """
    Initializes the channel for the first time (t=0).
    Determines initial statistical profile (LOS/NLOS) and calculates initial matrices.
    """
    large_scale_df = config['scenario']['environment']['large_scale_fading']
    propag_model = config['scenario']['environment']['propagation'].get('propagation_model', 'los').lower()

    # --- 1. Calculate the ideal channel (H_LB) using the helper ---
    h_lb = _calculate_ideal_channel(config, users['slant_range'], rp)
    
    channel = {
        'H_LB': h_lb,
        'H_tx': h_lb.copy() # Default to ideal channel if no loss model
    }

    # --- 2. Determine and save the initial statistical profile ---
    if propag_model in ['los', 'nlos']:
        # This is the unique job of the initializer
        elev_clipped = np.clip(users['theta'], 10, 90)
        row_index = (np.round(elev_clipped, -1) / 10 - 1).astype(int)

        if propag_model == 'los':
            los_flag = np.ones_like(users['theta'], dtype=bool)
        else:
            los_prob = large_scale_df['los_prob'].values[row_index]
            los_flag = np.random.rand(users['theta'].size) * 100 <= los_prob

        # --- 3. Apply propagation losses using the helper ---
        h_tx = _apply_propagation_losses(config, h_lb, users, row_index, los_flag)
        
        # --- 4. Store all results in the channel dictionary ---
        channel['H_lb'] = h_lb
        channel['H_tx'] = h_tx
        channel['row_index'] = row_index
        channel['los_flag'] = los_flag

    return channel

def _calculate_ideal_channel(config: dict, slant_range: np.ndarray, radiation_pattern: np.ndarray) -> np.ndarray:
    """
    Calculates the ideal, lossless complex channel matrix based on geometry.
    """
    derived_params = config['derived']
    
    # Normalization factor including receiver gain and free space path loss constant
    xi = (np.sqrt(10**(0.1 * derived_params['rx_gain_dB']) / derived_params['noise_W']) /
          ((4 * np.pi / derived_params['lambda_m'])))

    # Phase shift and amplitude loss from path length
    phase_path_loss = (np.exp(-1j * slant_range * 2 * np.pi / derived_params['lambda_m']) /
                       slant_range)
    
    # H_ideal is the ideal channel, often used for location-based scheduling
    h_ideal = xi * radiation_pattern * phase_path_loss[:, None] # Broadcasting 

    return h_ideal

def _apply_propagation_losses(
    config: dict,
    ideal_channel: np.ndarray,
    users: dict,
    row_index: np.ndarray,
    los_flag: np.ndarray
) -> np.ndarray:
    """
    Applies losses to the ideal channel.

    To avoid double counting:
      - Large-scale fading here = shadowing + clutter (urban/blockage model).
      - Atmospheric/path losses here = ITU-R P.618 total attenuation (gas + rain + scint [+ clouds if enabled]).
      - We DO NOT add separate scintillation & P.676 gaseous loss outside P.618.

    Note: expect a users dict with 'lat' and 'lon' keys for P.618 computations.
    """
    propagation_cfg = config['scenario']['environment']['propagation']
    payload_cfg = config['scenario']['satellite_payload']
    large_scale_df = config['scenario']['environment']['large_scale_fading']
    environment_cfg = config['scenario']['environment']
    usr_terminal_cfg = config['scenario']['user_terminal_definitions']

    # ---------------------------------------------------------
    # 1) Large-scale fading (shadowing + clutter) [dB]
    # ---------------------------------------------------------
    sigma = np.where(
        los_flag,
        large_scale_df['sigma_los'].values[row_index],
        large_scale_df['sigma_nlos'].values[row_index]
    )

    large_scale_loss_dB = np.random.randn(row_index.size) * sigma

    # Clutter only on NLOS (unless "force_nlos_for_all_users")
    if propagation_cfg.get('propagation_model', 'los').lower() == 'nlos':
        clutter_loss = large_scale_df['cl'].values[row_index]
        if propagation_cfg.get('force_nlos_for_all_users', False):
            large_scale_loss_dB += clutter_loss
        else:
            large_scale_loss_dB[~los_flag] += clutter_loss[~los_flag]

    # ---------------------------------------------------------
    # 2) ITU-R P.618 total attenuation per user [dB]
    # ---------------------------------------------------------
    if itur is None:
        p618_total_loss_dB = np.zeros_like(el, dtype=float)
    else:
        f_ghz = payload_cfg['carrier_frequency_hz'] * 1e-9

        # P.618 p is "% of time exceeded". Example: 99.99% availability -> p = 0.01
        p_exceed_percent = float(config.get('scenario', {}).get('p_exceed_percent', 0.01))

        # Antenna diameter for scintillation
        D_m = float(config.get('scenario', {}).get('antenna_diameter_m', 0.6))

        # Receiver station height above mean sea level (km).
        # If per-user altitude exists, prefer that.
        if 'alt_km' in users:
            hs_km = np.asarray(users['alt_km'], dtype=float)
        else:
            hs_km = float(config.get('scenario', {}).get('ground_station_height_km', 0.0))

        # Per-user lat/lon (deg) for P.618
        #if 'lat' not in users or 'lon' not in users:
            #raise KeyError("users dict must contain 'lat' and 'lon' for P.618 computations.")
        lat = np.asarray(users["lat"], dtype=float)
        lon = np.asarray(users["lon"], dtype=float)
        el = np.asarray(users["theta"], dtype=float)

        # Ensure elevation is valid for the model; P.618 is typically used above a few degrees
        # (we keep user elevations, but avoid exactly 0 which would blow up path terms inside some models)
        el = np.clip(el, 0.1, 90.0)    

        include_clouds = bool(propagation_cfg.get('include_clouds', True))

        try:

            results = itur.atmospheric_attenuation_slant_path(
                lat=lat,
                lon=lon,
                f=f_ghz,
                el=el,
                p= 0.01, #environment_cfg['p'],     # % time exceeded
                D= usr_terminal_cfg['antenna_diameter_m'],      
                hs=None, rho=None, R001=None, T=None, H=None, P=None,  # let ITU-Rpy compute
                include_rain=True, include_gas=True, include_scintillation=True, include_clouds=True,
                return_contributions=True,
            )

            A_total = results[-1]  # Quantity or ndarray
            # Convert to plain float array
            p618_total_loss_dB = np.asarray(getattr(A_total, "value", A_total), dtype=float)

            # If scalar, broadcast
            if p618_total_loss_dB.ndim == 0:
                p618_total_loss_dB = np.full_like(el, float(p618_total_loss_dB), dtype=float)

        except Exception as e:
            print(f"Error in ITU-Rpy P.618 calculation: {e}. Falling back to zero P.618 loss.")
            p618_total_loss_dB = np.zeros_like(el, dtype=float)

    # ---------------------------------------------------------
    # 3) Combine losses (NO double counting of scint / gaseous)
    # ---------------------------------------------------------
    total_loss_dB = large_scale_loss_dB + p618_total_loss_dB

    loss_scaling = 1.0 / np.sqrt(10 ** (0.1 * total_loss_dB))
    h_tx = ideal_channel * loss_scaling[:, None]
    return h_tx

# ==============================================================================
# SECTION 3: HANDOVER CHANNEL MODEL
# (Refactored to be functional/stateless)
# ==============================================================================

def compute_handover_rate(config: dict, gs_ecef: np.ndarray, sat_ecef: np.ndarray, 
                          el: float, los_flag: bool, row_index: int) -> float:
    """
    Calculates rate using the channel matrix from channel_matrix.py without the beamforming information.
    Indeed, the beamforming matrix is not computed for every courple sat-user! The handover module needs
    to follow a different strategy. We compute SNR = |H|^2 * P_tx directly from H_tx and then 
    the achievable rate with the Shannon Theorem.
    
    Args:
        config (dict): The main simulation configuration dictionary (must contain 
                       'derived', 'scenario', etc., as expected by channel_helpers).
        gs_ecef (np.ndarray): 3x1 ECEF position of the ground station.
        sat_ecef (np.ndarray): 3x1 ECEF position of the satellite.
        el (float): Elevation angle in degrees.
        los_flag (bool): Persistent LOS status for this user (from t=0 or tracker).
        row_index (int): Persistent elevation bin index (from t=0 or tracker).

    Returns:
        float: The achievable data rate in bits per second (bps).
    """
    # 1. Geometry & Distance
    # Ensure inputs are shaped correctly for math
    gs_position = gs_ecef.reshape(-1, 1) if gs_ecef.ndim == 1 else gs_ecef
    sat_position = sat_ecef.reshape(-1, 1) if sat_ecef.ndim == 1 else sat_ecef
    
    # Calculate slant range (distance)
    dist_vec = gs_position - sat_position
    distance = np.linalg.norm(dist_vec)

    # 2. Call channel_helpers to get h_tx
    # We wrap scalar inputs into arrays because the helpers expect vectors
    sr_array = np.array([distance])
    rp_array = np.array([1.0]) # Assuming isotropic/normalized antenna gain for this step
    el_array = np.array([el])
    row_idx_array = np.array([row_index])
    los_flag_array = np.array([los_flag])

    # A) Ideal Channel (FSPL + Phase)
    h_lb = _calculate_ideal_channel(config, sr_array, rp_array)
    
    # B) Apply Propagation Losses (Shadowing, Atmos, Rain, Scintillation via P.618)
    h_tx = _apply_propagation_losses(config, h_lb, el_array, row_idx_array, los_flag_array)

    # 3. Compute Rate (SNR -> Capacity)
    # Extract h_tx value (complex channel gain)
    h_complex = h_tx[0]
    
    # |H|^2
    h_squared = np.abs(h_complex)**2
    
    # Retrieve Power and Noise parameters from config
    # (Adjust keys if your config structure differs slightly)
    P_tx = config['simulation_params']['P_K']      # Transmit Power [W]
    N_noise = config['derived']['noise_W']         # Noise Power [W]
    Bandwidth = config['simulation_params']['B']   # Bandwidth [Hz]

    # SNR = P_tx * |H|^2 
    snr_linear = P_tx * h_squared

    # Rate = B * log2(1 + SNR)
    rate = Bandwidth * np.log2(1 + snr_linear)

    return snr_linear, max(0.0, rate)


# ==============================================================================
# SECTION 4: FULL CHANNEL PIPELINE (geometry → steering → element → channel → rate)
# Mirrors test_channel_model.py lines 183–251.
# Call once per active satellite per timestep, for its connected users only.
# ==============================================================================

def compute_rates(
    config:         dict,
    r_sat_m:        np.ndarray,     # (3,)  satellite ECEF position [m]
    user_lats_deg:  np.ndarray,     # (K,)  user latitudes  [deg]
    user_lons_deg:  np.ndarray,     # (K,)  user longitudes [deg]
    sat_alt_m:      float,          # satellite altitude above Earth surface [m]
) -> tuple:
    """
    Full channel pipeline for K users connected to one satellite.

    Returns
    -------
    rates_gbps : (K,) float32   Shannon rate per user [Gbps]
    snr_db     : (K,) float32   Channel norm in dB  (20·log10(||H_tx||))
    """
    # Local imports to avoid circular dependencies
    import pandas as pd
    import pymap3d as pm
    from simulator.auxiliary.utils.latlon2satview import lat_lon_to_sat_view
    from simulator.auxiliary.utils.uv_to_thetaphi import uv_to_thetaphi
    from simulator.auxiliary.utils.steering_vector_uv import steering_vector_uv
    from simulator.auxiliary.utils.element_pattern import element_pattern

    # Normalise config once so downstream helpers can use simple key lookups.
    # (Only matters the first time this runs if the caller hasn't pre-processed.)
    lsf = config['scenario']['environment']['large_scale_fading']
    if isinstance(lsf, list):
        # Shallow-copy chain to avoid mutating the caller's dict
        config = {**config}
        config['scenario'] = {**config['scenario']}
        config['scenario']['environment'] = {**config['scenario']['environment']}
        config['scenario']['environment']['large_scale_fading'] = pd.DataFrame(lsf)
    # Flatten active user-terminal definition so usr_terminal_cfg['antenna_diameter_m'] works
    utd = config['scenario']['user_terminal_definitions']
    if 'antenna_diameter_m' not in utd:
        active_type = config['scenario'].get('active_user_terminal_type', 'vsat')
        terminals   = utd.get('user_terminals', [])
        active_ut   = next((t for t in terminals if t.get('type') == active_type), terminals[0] if terminals else {})
        config = {**config}
        config['scenario'] = {**config['scenario']}
        config['scenario']['user_terminal_definitions'] = {**utd, **active_ut}

    R_E = 6_378_137.0   # WGS-84 semi-major axis [m]

    # 1. Sub-satellite point (SSP)
    sat_lat, sat_lon, _ = pm.ecef.ecef2geodetic(
        r_sat_m[0], r_sat_m[1], r_sat_m[2], deg=True
    )

    # 2. u, v in sin-space  (mirrors test lines 184–193)
    u, v = lat_lon_to_sat_view(
        user_lats_deg, user_lons_deg,
        sat_lat, sat_lon,
        sat_alt_m,
        coordinate_type='u-v',
    )
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # 3. Off-nadir angle η  (test line 196)
    off_nadir = np.degrees(np.arcsin(np.sqrt(u**2 + v**2)))

    # 4. Satellite nadir half-angle (FOV) and elevation at user  (test lines 200–204)
    sat_fov_deg = np.degrees(np.arcsin(R_E / (R_E + sat_alt_m)))
    elevation   = np.degrees(
        np.arccos(np.clip(
            np.sin(np.radians(off_nadir)) / np.sin(np.radians(sat_fov_deg)),
            -1.0, 1.0
        ))
    )

    # 5. Earth central angle and slant range  (test lines 207–211)
    earth_angle = 90.0 - elevation - off_nadir
    slant_range = R_E * np.sin(np.radians(earth_angle)) / np.maximum(
        np.sin(np.radians(off_nadir)), 1e-12
    )   # [m]

    # 6. Antenna-frame angles θ, φ  (test line 224)
    theta, phi = uv_to_thetaphi(u, v)

    # 7. Steering vector  (test line 214)
    SV = steering_vector_uv(u, v, config)   # (K, n_elements)

    # 8. Element pattern  (test lines 228–238)
    tx_ant   = config['scenario']['satellite_payload']['tx_antenna']
    A_el_dB, _, _ = element_pattern(
        theta, phi,
        tx_ant.get('theta_3dB', 65.0),
        tx_ant.get('phi_3dB',   65.0),
        tx_ant.get('Am',        30.0),
        tx_ant.get('SLAv',      30.0),
        tx_ant['G_el_dB'],
    )
    A_el = 10 ** (0.05 * A_el_dB)          # linear amplitude
    RP   = SV * A_el[:, np.newaxis]         # (K, n_elements)

    # 9. Channel matrices  (test lines 241–245)
    users_dict = {
        'lat':         user_lats_deg,
        'lon':         user_lons_deg,
        'theta':       theta,       # used as elevation proxy inside initialize_channel
        'slant_range': slant_range,
    }
    ch        = initialize_channel(config, users_dict, RP)
    H_tx_norm = np.linalg.norm(ch['H_tx'], ord=2, axis=1)   # (K,)

    # 10. Shannon rate  (test line 251)
    B          = config['scenario']['satellite_payload']['bandwidth_mhz'] * 1e6  # [Hz]
    rates_gbps = (B * np.log2(1.0 + H_tx_norm) / 1e9).astype(np.float32)
    snr_db     = (20.0 * np.log10(np.maximum(H_tx_norm, 1e-12))).astype(np.float32)

    return rates_gbps, snr_db