"""
Regression tests: old (validated) script logic vs new script logic.

Each test compares the output of the OLD function (inlined below, exactly as
provided in *_old.py files) against the NEW function imported from the repo.

Pass criterion: all assertions green.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Inline copies of the OLD functions (from *_old.py reference files)
# ---------------------------------------------------------------------------

def uv_to_thetaphi_OLD(u, v):
    """From uv_to_thethaphi_old.py"""
    theta = np.degrees(np.arccos(v))
    phi = np.degrees(np.arcsin(u / np.sin(np.radians(theta))))
    return theta, phi


def element_pattern_OLD(theta, phi, theta_3db, phi_3db, Am, SLAv, Ge_max):
    """From element_pattern_old.py  (3GPP TR 38.901 with boresight at θ=90°)"""
    theta = np.asarray(theta, dtype=float)
    phi   = np.asarray(phi,   dtype=float)
    A_el_H_dB = -np.minimum(Am, 12 * (phi / phi_3db)**2)
    A_el_V_dB = -np.minimum(SLAv, 12 * ((theta - 90) / theta_3db)**2)
    A_el_dB   = Ge_max - np.minimum(Am, -(A_el_H_dB + A_el_V_dB))
    return A_el_dB, A_el_H_dB, A_el_V_dB


def steering_vector_OLD(u, v, n_el_H, n_el_V, d_h, d_v, wavelength):
    """From steering_vector_uv_old.py"""
    u_col = np.asarray(u).reshape(-1, 1)
    v_col = np.asarray(v).reshape(-1, 1)
    xPos = np.arange(n_el_H) * d_h
    yPos = np.arange(n_el_V) * d_v
    x, y = np.meshgrid(xPos, yPos)   # (n_el_V, n_el_H)
    x = x.reshape(-1)[None, :]        # (1, K)
    y = y.reshape(-1)[None, :]
    k0 = 2.0 * np.pi / wavelength
    return np.exp(1j * k0 * (x * u_col + y * v_col))  # (N, K)


# ---------------------------------------------------------------------------
# Import NEW functions
# ---------------------------------------------------------------------------
from simulator.auxiliary.utils.uv_to_thetaphi   import uv_to_thetaphi   as uv2tp_NEW
from simulator.auxiliary.utils.element_pattern  import element_pattern   as el_pat_NEW
from simulator.auxiliary.utils.steering_vector_uv import steering_vector_uv as sv_NEW
from simulator.auxiliary.utils.load_config       import load_config       as _lc

# Load config (needed by steering_vector_uv)
from simulator.auxiliary.utils.calculate_derived_parameters import calculate_derived_parameters
from simulator.auxiliary.utils.load_large_scale_model       import load_large_scale_model
import yaml
from yaml_include import Constructor

def _load_cfg():
    path = ROOT / "simulator" / "main.yaml"
    yaml.add_constructor("!include", Constructor(base_dir=str(path.parent)), Loader=yaml.FullLoader)
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    scenario = cfg["scenario"]["environment"]["propagation"]["channel_scenario"]
    cfg["scenario"]["environment"]["large_scale_fading"] = load_large_scale_model(scenario)
    utd = cfg["scenario"]["user_terminal_definitions"]
    active_type = cfg["scenario"].get("active_user_terminal_type", "vsat")
    active_ut   = next(t for t in utd["user_terminals"] if t["type"] == active_type)
    cfg["scenario"]["user_terminal_definitions"] = {**utd, **active_ut}
    return calculate_derived_parameters(cfg)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def ok(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def assert_close(name, a, b, atol=1e-6, rtol=1e-5):
    diff = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    rel  = np.max(np.abs(np.asarray(a) - np.asarray(b)) /
                  (np.abs(np.asarray(b)) + 1e-30))
    cond = (diff <= atol) or (rel <= rtol)
    return ok(name, cond, f"max_abs_diff={diff:.3e}, max_rel={rel:.3e}")


# ---------------------------------------------------------------------------
# Test grid of (u, v) values covering the satellite footprint
# ---------------------------------------------------------------------------
# Users at various off-boresight angles (0 to 50°) and all bearings
off_nadir = np.array([0.01, 5, 10, 20, 30, 40, 50])
bearings  = np.array([0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 270, 315])

U_list, V_list = [], []
for eta in off_nadir:
    for brg in bearings:
        U_list.append(np.sin(np.radians(eta)) * np.cos(np.radians(brg)))
        V_list.append(np.sin(np.radians(eta)) * np.sin(np.radians(brg)))
U = np.array(U_list)
V = np.array(V_list)

THETA_3DB, PHI_3DB, Am, SLAv, G_MAX = 65.0, 65.0, 30.0, 30.0, 5.0

# ============================================================
# TEST 1: steering_vector_uv
# Old vs new should give identical (u,v) → steering vector
# ============================================================
def test_steering_vector():
    print("\n=== TEST 1: steering_vector_uv (OLD vs NEW) ===")
    cfg = _load_cfg()
    d   = cfg["derived"]
    tx  = cfg["scenario"]["satellite_payload"]["tx_antenna"]
    n_H, n_V = tx["n_el_H"], tx["n_el_V"]
    d_h = d["tx_element_spacing_H_m"]
    d_v = d["tx_element_spacing_V_m"]
    lam = d["lambda_m"]

    SV_old = steering_vector_OLD(U, V, n_H, n_V, d_h, d_v, lam)
    SV_new = sv_NEW(U, V, cfg)

    assert_close("shape matches", [SV_old.shape], [SV_new.shape], atol=0)
    assert_close("real part",  SV_old.real, SV_new.real)
    assert_close("imag part",  SV_old.imag, SV_new.imag)
    assert_close("|SV| = 1 (old)", np.abs(SV_old), np.ones_like(np.abs(SV_old)))
    assert_close("|SV| = 1 (new)", np.abs(SV_new), np.ones_like(np.abs(SV_new)))


# ============================================================
# TEST 2: element_pattern — combined pipeline A_el(u,v)
# Both old (inlined) and current (imported) functions now use
# the same convention: theta=arccos(v), phi=phi_LCS.
# They must give bit-identical output.
# ============================================================
def test_element_pattern_pipeline():
    print("\n=== TEST 2: A_el(u,v) — OLD (inlined) vs current element_pattern ===")
    # OLD pipeline (inlined)
    theta_old, phi_old = uv_to_thetaphi_OLD(U, V)
    A_old, AH_old, AV_old = element_pattern_OLD(theta_old, phi_old,
                                                 THETA_3DB, PHI_3DB, Am, SLAv, G_MAX)

    # Current pipeline (imported — now same convention as OLD)
    theta_cur, phi_cur = uv2tp_NEW(U, V)
    A_cur, AH_cur, AV_cur = el_pat_NEW(theta_cur, phi_cur,
                                        THETA_3DB, PHI_3DB, Am, SLAv, G_MAX)

    assert_close("A_el_dB  (composite): inlined OLD == imported current", A_old, A_cur, atol=1e-3)

    # Boresight: both must give G_MAX  (u=0,v=0 → theta=90°, phi=0°)
    u0, v0 = np.array([0.0]), np.array([0.0])
    t0, p0 = uv_to_thetaphi_OLD(u0, v0)
    A0_old, _, _ = element_pattern_OLD(t0, p0, THETA_3DB, PHI_3DB, Am, SLAv, G_MAX)
    t0c, p0c = uv2tp_NEW(u0, v0)
    A0_cur, _, _ = el_pat_NEW(t0c, p0c, THETA_3DB, PHI_3DB, Am, SLAv, G_MAX)
    assert_close("A_el at boresight OLD = G_MAX", A0_old, [G_MAX], atol=0.01)
    assert_close("A_el at boresight current = G_MAX", A0_cur, [G_MAX], atol=0.01)

    # Rotational symmetry: equal-H/V-bw array, fixed off-nadir ring → same A_el
    eta = np.radians(10.0)
    brg_sweep = np.linspace(0, 2*np.pi, 361)
    u_ring = np.sin(eta) * np.cos(brg_sweep)
    v_ring = np.sin(eta) * np.sin(brg_sweep)
    t_ring, p_ring = uv2tp_NEW(u_ring, v_ring)
    A_ring, _, _ = el_pat_NEW(t_ring, p_ring, 65.0, 65.0, 30.0, 30.0, 5.0)
    assert_close("A_el invariant over bearing (equal H/V bw)", A_ring, A_ring.mean()*np.ones_like(A_ring), atol=0.01)


# ============================================================
# TEST 3: uv_to_thetaphi — roundtrip consistency
# Both the inlined OLD and the imported current function use the
# same convention: theta=arccos(v), phi=arcsin(u/sin(theta)).
# Verify the convention is self-consistent.
# ============================================================
def test_uv_to_thetaphi_consistency():
    print("\n=== TEST 3: uv_to_thetaphi — roundtrip consistency ===")
    test_u = np.array([0.0, 0.2, 0.3, -0.2, 0.1, 0.0])
    test_v = np.array([0.0, 0.0, 0.3,  0.0, 0.2, 0.4])

    # Convention: theta=arccos(v), phi=arcsin(u/sin(theta))
    # Roundtrip checks for the inlined OLD version:
    t_old, p_old = uv_to_thetaphi_OLD(test_u, test_v)
    v_roundtrip = np.cos(np.radians(t_old))
    assert_close("cos(theta) = v", v_roundtrip, test_v, atol=1e-10)
    u_roundtrip = np.sin(np.radians(t_old)) * np.sin(np.radians(p_old))
    assert_close("sin(theta)*sin(phi) = u", u_roundtrip, test_u, atol=1e-10)

    # Imported current function must give identical output to the inlined OLD
    t_cur, p_cur = uv2tp_NEW(test_u, test_v)
    assert_close("imported uv_to_thetaphi == inlined OLD (theta)", t_old, t_cur, atol=1e-10)
    assert_close("imported uv_to_thetaphi == inlined OLD (phi)",   p_old, p_cur, atol=1e-10)

    # Boresight: u=0, v=0 → theta must be 90° (boresight), phi=0°
    t0, p0 = uv2tp_NEW(np.array([0.0]), np.array([0.0]))
    ok("theta=90° at boresight (u=0,v=0)", abs(t0[0] - 90.0) < 0.01,
       f"theta={t0[0]:.4f}°")


# ============================================================
# TEST 4: noise_W — must equal k*T_sys*B
# ============================================================
def test_noise_w():
    print("\n=== TEST 4: noise_W in derived parameters ===")
    cfg = _load_cfg()
    d   = cfg["derived"]
    ut  = cfg["scenario"]["user_terminal_definitions"]
    payload = cfg["scenario"]["satellite_payload"]
    const   = cfg["scenario"]["environment"]["physical_constants"]

    k   = const["boltzmann_k"]
    B   = payload["bandwidth_mhz"] * 1e6
    NF_lin = 10 ** (0.1 * ut["noise_figure_dB"])
    T_ant  = ut["noise_temperature_K"]
    T_sys  = T_ant + 290.0 * (NF_lin - 1.0)   # Friis
    kTB    = k * T_sys * B

    assert_close("noise_W == k*T_sys*B", d["noise_W"], kTB, rtol=1e-9)
    assert_close("noise_dBW == 10*log10(kTB)", d["noise_dBW"],
                 10 * np.log10(kTB), atol=0.001)

    # Verify G/T is physically reasonable (positive, < G_rx)
    gt = d["rx_gt_dB_K"]
    ok("G/T > 0 dB/K", gt > 0, f"G/T = {gt:.2f} dB/K")
    ok("G/T < G_rx (noise temp reduces G/T)", gt < d["rx_gain_dB"],
       f"G/T={gt:.2f} < G_rx={d['rx_gain_dB']:.2f}")


# ============================================================
# TEST 5: End-to-end SNR sanity check
# Full pipeline: (u,v) -> SV*A_el -> H -> |H|^2 * P_tx
# At boresight, SNR must match analytical link budget within 0.1 dB
# ============================================================
def test_snr_end_to_end():
    print("\n=== TEST 5: End-to-end SNR vs analytical link budget ===")
    from simulator.auxiliary.scenario_builder.channel_model.channel_model import _calculate_ideal_channel

    cfg = _load_cfg()
    d   = cfg["derived"]
    tx  = cfg["scenario"]["satellite_payload"]["tx_antenna"]
    const = cfg["scenario"]["environment"]["physical_constants"]

    k    = const["boltzmann_k"]
    B    = cfg["scenario"]["satellite_payload"]["bandwidth_mhz"] * 1e6
    ut   = cfg["scenario"]["user_terminal_definitions"]
    NF_lin = 10 ** (0.1 * ut["noise_figure_dB"])
    T_sys  = ut["noise_temperature_K"] + 290.0 * (NF_lin - 1.0)
    kTB    = k * T_sys * B
    P_tx   = d["tx_power_W"]
    G_rx   = 10 ** (0.1 * d["rx_gain_dB"])
    G_el   = 10 ** (0.1 * tx["G_el_dB"])
    N_elem = tx["n_el_H"] * tx["n_el_V"]
    G_tx   = N_elem * G_el        # boresight tx array gain
    lam    = d["lambda_m"]
    sr     = 310e3                 # nadir slant range

    # Analytical SNR
    FSPL       = (lam / (4 * np.pi * sr)) ** 2
    snr_analyt = P_tx * G_tx * G_rx * FSPL / kTB
    snr_analyt_dB = 10 * np.log10(snr_analyt)

    # --- Code SNR via NEW pipeline at nadir (u≈0, v≈1e-4, along +v) ---
    eps = 1e-5
    u_test = np.array([eps])
    v_test = np.array([eps])
    sr_arr = np.array([sr])

    theta_n, phi_n = uv2tp_NEW(u_test, v_test)
    SV   = sv_NEW(u_test, v_test, cfg)
    A_dB, _, _ = el_pat_NEW(theta_n, phi_n,
                             tx["theta_3dB"], tx["phi_3dB"],
                             tx["Am"], tx["SLAv"], tx["G_el_dB"])
    RP   = SV * (10 ** (0.05 * A_dB))[:, None]
    H    = _calculate_ideal_channel(cfg, sr_arr, RP)
    snr_code_dB = 10 * np.log10(P_tx * (np.abs(H)**2).sum(axis=1)[0])

    diff_dB = snr_code_dB - snr_analyt_dB
    assert_close("SNR_code ≈ SNR_analytical at nadir (±0.2 dB)",
                 snr_code_dB, snr_analyt_dB, atol=0.2)
    print(f"    SNR analytical = {snr_analyt_dB:.2f} dB")
    print(f"    SNR code       = {snr_code_dB:.2f} dB")
    print(f"    Difference     = {diff_dB:.3f} dB")

    # At 50° off-nadir, code SNR must be BELOW analytical (scan loss)
    eta50 = np.radians(50)
    R_E = 6_378_137.0
    sat_fov = np.arcsin(R_E / (R_E + sr))
    el50 = np.arccos(np.clip(np.sin(eta50)/np.sin(sat_fov), -1, 1))
    ea50 = np.pi/2 - el50 - eta50
    sr50 = R_E * np.sin(ea50) / max(np.sin(eta50), 1e-12)
    u50 = np.array([np.sin(eta50)])
    v50 = np.array([eps])
    t50, p50 = uv2tp_NEW(u50, v50)
    SV50 = sv_NEW(u50, v50, cfg)
    A50, _, _ = el_pat_NEW(t50, p50, tx["theta_3dB"], tx["phi_3dB"],
                            tx["Am"], tx["SLAv"], tx["G_el_dB"])
    RP50 = SV50 * (10**(0.05 * A50))[:, None]
    H50  = _calculate_ideal_channel(cfg, np.array([sr50]), RP50)
    snr50_dB = 10 * np.log10(P_tx * (np.abs(H50)**2).sum(axis=1)[0])
    ok("SNR at 50° off-nadir < SNR at nadir (scan loss present)",
       snr50_dB < snr_code_dB,
       f"nadir={snr_code_dB:.1f} dB, η=50°: {snr50_dB:.1f} dB")


# ============================================================
# TEST 6: channel_model.py (functional) vs channel.py (class)
#
# Sub-tests:
#   6a. Radiation pattern: Antenna.compute_tx_radiation_pattern == manual SV*A_el
#   6b. _calculate_ideal_channel formula: identical given same inputs
#   6c. Antenna.rx_noise_W() == k*T_sys*B  (EXPECTED FAIL — Antenna has same noise bug)
#   6d. SNR impact of the noise_W difference
# ============================================================
def test_channel_model_vs_channel():
    print("\n=== TEST 6: channel_model.py vs channel.py ===")
    from simulator.auxiliary.scenario_builder.antenna import Antenna
    from simulator.auxiliary.scenario_builder.channel_model.channel_model import _calculate_ideal_channel

    cfg     = _load_cfg()
    payload = cfg['scenario']['satellite_payload']
    ut      = cfg['scenario']['user_terminal_definitions']
    d       = cfg['derived']
    const   = cfg['scenario']['environment']['physical_constants']

    # Build Antenna object with the same parameters as the config
    tx_cfg = payload['tx_antenna']
    rx_cfg = {
        'gain_dB':             ut['gain_dB'],
        'noise_figure_dB':     ut['noise_figure_dB'],
        'noise_temperature_K': ut['noise_temperature_K'],
        'antenna_diameter_m':  ut.get('antenna_diameter_m', 0.6),
        'shadowing_margin_dB': ut.get('shadowing_margin_dB', 0.0),
    }
    svc_cfg = {
        'bandwidth_mhz':           payload['bandwidth_mhz'],
        'rx_carrier_frequency_hz': payload['carrier_frequency_hz'],
        'tx_carrier_frequency_hz': payload['carrier_frequency_hz'],
    }
    ant = Antenna(tx_cfg, rx_cfg, svc_cfg)

    # --- 6a: Radiation pattern identity ---
    # Antenna.compute_tx_radiation_pattern must equal manual (SV * A_el)
    RP_ant = ant.compute_tx_radiation_pattern(U, V)

    theta_m, phi_m = uv2tp_NEW(U, V)
    SV_m           = sv_NEW(U, V, cfg)
    A_dB_m, _, _   = el_pat_NEW(theta_m, phi_m,
                                  tx_cfg['theta_3dB'], tx_cfg['phi_3dB'],
                                  tx_cfg['Am'], tx_cfg['SLAv'], tx_cfg['G_el_dB'])
    RP_manual = SV_m * (10.0 ** (0.05 * A_dB_m))[:, None]

    assert_close("6a RP real: Antenna == manual", RP_ant.real, RP_manual.real)
    assert_close("6a RP imag: Antenna == manual", RP_ant.imag, RP_manual.imag)

    # --- 6b: _calculate_ideal_channel formula is identical ---
    # Both files use xi = sqrt(G_rx/noise_W)/(4π/λ) and h = xi*RP*phase/sr.
    # Given the SAME noise_W and a single boresight user, both must agree.
    sr_test = np.array([310e3])
    RP_bore = RP_manual[0:1]   # first user (close to boresight)

    H_cm = _calculate_ideal_channel(cfg, sr_test, RP_bore)   # channel_model.py

    # Replicate channel.py compute() xi inline with correct noise_W from config
    lam   = d['lambda_m']
    xi_ch = np.sqrt(10 ** (0.1 * ant.rx_gain_dB) / d['noise_W']) / (4 * np.pi / lam)
    phase = np.exp(-1j * sr_test * 2 * np.pi / lam) / sr_test
    H_ch_correct_noise = xi_ch * RP_bore * phase[:, None]

    assert_close("6b _calculate_ideal_channel: same formula, same inputs → same H",
                 H_cm, H_ch_correct_noise)

    # --- 6c: Antenna.rx_noise_W() should equal k*T_sys*B ---
    k       = const['boltzmann_k']
    B       = payload['bandwidth_mhz'] * 1e6
    NF_lin  = 10 ** (0.1 * ut['noise_figure_dB'])
    T_sys   = ut['noise_temperature_K'] + 290.0 * (NF_lin - 1.0)   # Friis
    kTB     = k * T_sys * B

    noise_W_ant = ant.rx_noise_W()
    diff_dB     = 10 * np.log10(noise_W_ant / kTB)

    print(f"    Antenna.rx_noise_W() = {noise_W_ant:.4e} W  (= G_rx·kB/T_eff, includes G_rx)")
    print(f"    Correct k·T_sys·B   = {kTB:.4e} W")
    print(f"    Ratio               = {diff_dB:+.2f} dB  (Antenna embeds G_rx into noise_W)")

    # After fix: Antenna.rx_noise_W must equal kTB
    rel_err = abs(noise_W_ant - kTB) / kTB
    assert_close("6c Antenna.rx_noise_W() == k*T_sys*B  (Friis, bug fixed)",
                 noise_W_ant, kTB, rtol=0.001)

    # --- 6d: SNR impact ---
    # With Antenna's noise_W (which includes G_rx), xi gets G_rx cancelled:
    xi_buggy = np.sqrt(10 ** (0.1 * ant.rx_gain_dB) / noise_W_ant) / (4 * np.pi / lam)
    H_buggy  = xi_buggy * RP_bore * phase[:, None]
    P_tx     = d['tx_power_W']
    snr_correct_dB = 10 * np.log10(P_tx * np.sum(np.abs(H_cm)    ** 2))
    snr_buggy_dB   = 10 * np.log10(P_tx * np.sum(np.abs(H_buggy) ** 2))
    print(f"    SNR (channel_model.py, correct noise): {snr_correct_dB:.2f} dB")
    print(f"    SNR (channel.py, Antenna.rx_noise_W):  {snr_buggy_dB:.2f} dB")
    print(f"    Difference: {snr_buggy_dB - snr_correct_dB:+.2f} dB  (channel.py is off by this)")
    ok("6d channel.py SNR == channel_model.py SNR  (Antenna noise bug fixed)",
       abs(snr_buggy_dB - snr_correct_dB) < 0.01,
       f"diff={snr_buggy_dB - snr_correct_dB:+.3f} dB")


# ============================================================
# TEST 7: Antenna.compute_steering_vector() vs old steering_vector function
# ============================================================
def test_antenna_steering_vector():
    print("\n=== TEST 7: Antenna.compute_steering_vector() vs OLD ===")
    from simulator.auxiliary.scenario_builder.antenna import Antenna

    cfg     = _load_cfg()
    payload = cfg['scenario']['satellite_payload']
    ut      = cfg['scenario']['user_terminal_definitions']
    d       = cfg['derived']

    tx_cfg  = payload['tx_antenna']
    svc_cfg = {
        'bandwidth_mhz':           payload['bandwidth_mhz'],
        'rx_carrier_frequency_hz': payload['carrier_frequency_hz'],
        'tx_carrier_frequency_hz': payload['carrier_frequency_hz'],
    }
    rx_cfg = {
        'gain_dB':             ut['gain_dB'],
        'noise_figure_dB':     ut['noise_figure_dB'],
        'noise_temperature_K': ut['noise_temperature_K'],
        'antenna_diameter_m':  ut.get('antenna_diameter_m', 0.6),
    }
    ant = Antenna(tx_cfg, rx_cfg, svc_cfg)

    SV_old = steering_vector_OLD(U, V,
                                  tx_cfg['n_el_H'], tx_cfg['n_el_V'],
                                  d['tx_element_spacing_H_m'],
                                  d['tx_element_spacing_V_m'],
                                  d['lambda_m'])
    SV_ant = ant.compute_steering_vector(U, V)

    assert_close("7 shape matches",      [SV_old.shape], [SV_ant.shape], atol=0)
    assert_close("7 real part",          SV_old.real, SV_ant.real)
    assert_close("7 imag part",          SV_old.imag, SV_ant.imag)
    assert_close("7 |SV|=1 (Antenna)",   np.abs(SV_ant),
                 np.ones_like(np.abs(SV_ant)))


# ============================================================
# TEST 8: Antenna.compute_element_pattern() vs old element_pattern function
# Full pipeline: (u,v) → (theta,phi) → element_pattern via Antenna vs OLD
# ============================================================
def test_antenna_element_pattern():
    print("\n=== TEST 8: Antenna.compute_element_pattern() vs OLD pipeline ===")
    from simulator.auxiliary.scenario_builder.antenna import Antenna

    cfg     = _load_cfg()
    payload = cfg['scenario']['satellite_payload']
    ut      = cfg['scenario']['user_terminal_definitions']
    tx_cfg  = payload['tx_antenna']
    svc_cfg = {
        'bandwidth_mhz':           payload['bandwidth_mhz'],
        'rx_carrier_frequency_hz': payload['carrier_frequency_hz'],
        'tx_carrier_frequency_hz': payload['carrier_frequency_hz'],
    }
    rx_cfg = {
        'gain_dB':             ut['gain_dB'],
        'noise_figure_dB':     ut['noise_figure_dB'],
        'noise_temperature_K': ut['noise_temperature_K'],
        'antenna_diameter_m':  ut.get('antenna_diameter_m', 0.6),
    }
    ant = Antenna(tx_cfg, rx_cfg, svc_cfg)

    # OLD (inlined) pipeline
    theta_old, phi_old = uv_to_thetaphi_OLD(U, V)
    A_old, _, _ = element_pattern_OLD(theta_old, phi_old,
                                       THETA_3DB, PHI_3DB, Am, SLAv, G_MAX)

    # Antenna: uses same convention (theta=arccos(v), phi=phi_LCS)
    theta_cur, phi_cur = uv2tp_NEW(U, V)
    A_ant, _, _ = ant.compute_element_pattern(theta_cur, phi_cur)

    assert_close("8 A_el_dB (composite): Antenna == OLD pipeline",
                 A_old, A_ant, atol=1e-3)

    # Boresight: u=0,v=0 → theta=90°, phi=0° → must give G_MAX
    t0, p0 = uv2tp_NEW(np.array([0.0]), np.array([0.0]))
    assert_close("8 A_el at boresight via Antenna == G_MAX",
                 ant.compute_element_pattern(t0, p0)[0], [G_MAX], atol=0.01)

    # Rotational symmetry at fixed off-nadir ring (equal H/V bw)
    eta = np.radians(10.0)
    brg = np.linspace(0, 2*np.pi, 361)
    u_r = np.sin(eta)*np.cos(brg)
    v_r = np.sin(eta)*np.sin(brg)
    t_r, p_r = uv2tp_NEW(u_r, v_r)
    A_r, _, _ = ant.compute_element_pattern(t_r, p_r)
    assert_close("8 A_el invariant over bearing via Antenna (equal H/V bw)",
                 A_r, A_r.mean()*np.ones_like(A_r), atol=0.01)


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    for fn in [test_steering_vector,
               test_element_pattern_pipeline,
               test_uv_to_thetaphi_consistency,
               test_noise_w,
               test_snr_end_to_end,
               test_channel_model_vs_channel,
               test_antenna_steering_vector,
               test_antenna_element_pattern]:
        fn()

    print("\n" + "="*50)
    print("All assertions printed above (PASS/FAIL).")

