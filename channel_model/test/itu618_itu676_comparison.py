import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
#   1. ITU-R P.676-13 (Annex 2) - CORRECTED IMPLEMENTATION
# =====================================================================

def _phi(rp, rt, a, b, c, d):
    return np.power(rp, a) * np.power(rt, b) * np.exp(c * (1 - rp) + d * (1 - rt))

def _sq(n, rp, rt):
    if n == 1: return _phi(rp, rt, 0.0717, -1.8132, 0.0156, -1.6515)
    elif n == 2: return _phi(rp, rt, 0.5146, -4.6368, -0.1921, -5.7416)
    elif n == 3: return _phi(rp, rt, 0.3414, -6.5851, 0.2130, -8.5854)
    elif n == 4: return _phi(rp, rt, -0.0112, 0.0092, -0.1033, -0.0009)
    elif n == 5: return _phi(rp, rt, 0.2705, -2.7192, -0.3016, -4.1033)
    elif n == 6: return _phi(rp, rt, 0.2445, -5.9191, 0.0422, -8.0719)
    elif n == 7: return _phi(rp, rt, -0.1833, 6.5589, -0.2402, 6.1310)
    elif n == 8: return _phi(rp, rt, 1.8286, -1.9487, 0.4051, -2.8509) * 2.192
    elif n == 9: return _phi(rp, rt, 1.0045, 3.5610, 0.1588, 1.2834) * 12.59
    elif n == 10: return _phi(rp, rt, 0.9003, 4.1335, 0.0427, 1.6088) * 15.0
    elif n == 11: return _phi(rp, rt, 0.9886, 3.4176, 0.1827, 1.3429) * 14.28
    elif n == 12: return _phi(rp, rt, 1.4320, 0.6258, 0.3177, -0.5914) * 6.819
    elif n == 13: return _phi(rp, rt, 2.0717, -4.1404, 0.4910, -4.8718) * 1.908
    elif n == 14: return _phi(rp, rt, 3.2110, -14.940, 1.5830, -16.370) * -0.00306
    else: raise ValueError("N must be an integer between 1 and 14.")

def _gfun(f, fi):
    return 1 + ((f - fi) / (f + fi))**2

def _gamo(f, rp, rt):
    f = np.asarray(f)
    
    # Pre-calculate _sq values (scalars)
    sq_vals = {i: _sq(i, rp, rt) for i in range(1, 15)}
    
    def f1(f_sub): return 1e-3 * f_sub**2 * rp**2 * (7.2 * rt**2.8 / (f_sub**2 + 0.34 * rp**2 * rt**1.6) + 0.62 * sq_vals[3] / ((54 - f_sub)**(1.16 * sq_vals[1]) + 0.83 * sq_vals[2]))
    def f2(f_sub): return np.exp(np.log(sq_vals[8]) / 24 * (f_sub - 58) * (f_sub - 60) - np.log(sq_vals[9]) / 8 * (f_sub - 54) * (f_sub - 60) + np.log(sq_vals[10]) / 12 * (f_sub - 54) * (f_sub - 58))
    def f3(f_sub): return sq_vals[10] + (sq_vals[11] - sq_vals[10]) * (f_sub - 60) / 2
    def f4(f_sub): return np.exp(np.log(sq_vals[11]) / 8 * (f_sub - 64) * (f_sub - 66) - np.log(sq_vals[12]) / 4 * (f_sub - 62) * (f_sub - 66) + np.log(sq_vals[13]) / 8 * (f_sub - 62) * (f_sub - 64))
    def f5(f_sub): return 1e-3 * f_sub**2 * rp**2 * (3.02e-4 * rt**3.5 + 0.283 * rt**3.8 / ((f_sub - 118.75)**2 + 2.91 * rp**2 * rt**1.6) + 0.502 * sq_vals[6] * (1 - 0.0163 * sq_vals[7] * (f_sub - 66)) / ((f_sub - 66)**(1.4346 * sq_vals[4]) + 1.15 * sq_vals[5]))
    def f6(f_sub): return sq_vals[14] + 1e-3 * f_sub**2 * rp**2 * rt**3.5 * (3.02e-4 / (1 + 1.9e-5 * f_sub**1.5) + 0.283 * rt**0.3 / ((f_sub - 118.75)**2 + 2.91 * rp**2 * rt**1.6))

    condlist = [f <= 54, (f > 54) & (f <= 60), (f > 60) & (f <= 62), (f > 62) & (f <= 66), (f > 66) & (f <= 120), (f > 120) & (f <= 350)]
    funclist = [f1, f2, f3, f4, f5, f6]
    return np.piecewise(f, condlist, funclist)

def _gamw(f, rp, rt, rho):
    f = np.asarray(f)
    eta1 = 0.955 * rp * rt**0.68 + 0.006 * rho
    eta2 = 0.735 * rp * rt**0.5 + 0.0353 * rt**4 * rho
    term1 = 3.98 * eta1 * np.exp(2.23 * (1 - rt)) / ((f - 22.235)**2 + 9.42 * eta1**2) * _gfun(f, 22)
    term2 = 11.96 * eta1 * np.exp(0.7 * (1 - rt)) / ((f - 183.31)**2 + 11.14 * eta1**2)
    term3 = 0.081 * eta1 * np.exp(6.44 * (1 - rt)) / ((f - 321.226)**2 + 6.29 * eta1**2)
    term4 = 3.66 * eta1 * np.exp(1.60 * (1 - rt)) / ((f - 325.153)**2 + 9.22 * eta1**2)
    term5 = 25.37 * eta1 * np.exp(1.09 * (1 - rt)) / ((f - 380)**2)
    term6 = 17.40 * eta1 * np.exp(1.46 * (1 - rt)) / ((f - 448)**2)
    term7 = 844.6 * eta1 * np.exp(0.17 * (1 - rt)) / ((f - 557)**2) * _gfun(f, 557)
    term8 = 290.0 * eta1 * np.exp(0.41 * (1 - rt)) / ((f - 752)**2) * _gfun(f, 752)
    term9 = 83328 * eta2 * np.exp(0.99 * (1 - rt)) / ((f - 1780)**2) * _gfun(f, 1780)
    gam = 1e-4 * rho * rt**2.5 * f**2 * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)
    return gam

def itu676(pdry, T, rho, f):
    f = np.asarray(f)
    rp = (pdry + rho * T / 216.5) / 1013.0
    rt = 288.0 / T # CORRECTED to 288.0
    t1 = 4.64 / (1 + 0.066 * rp**-2.3) * np.exp(-((f - 59.7) / (2.87 + 12.4 * np.exp(-7.9 * rp)))**2)
    t2 = 0.14 * np.exp(2.12 * rp) / ((f - 118.75)**2 + 0.031 * np.exp(2.2 * rp))
    t3 = (0.0114 / (1 + 0.14 * rp**-2.6) * f * (-0.0247 + 0.0001 * f + 1.61e-6 * f**2) / (1 - 0.0169 * f + 4.1e-5 * f**2 + 3.2e-7 * f**3))
    h0 = 6.1 / (1 + 0.17 * rp**-1.1) * (1 + t1 + t2 + t3)
    mask = (f < 70) & (h0 > 10.7 * rp**0.3)
    h0 = np.where(mask, 10.7 * rp**0.3, h0)
    sigw = 1.013 / (1 + np.exp(-8.6 * (rp - 0.57)))
    hw = 1.66 * (1 + 1.39 * sigw / ((f - 22.235)**2 + 2.56 * sigw) + 3.37 * sigw / ((f - 183.31)**2 + 4.69 * sigw) + 1.58 * sigw / ((f - 325.1)**2 + 2.89 * sigw))
    gamma0 = _gamo(f, rp, rt)
    gammaw = _gamw(f, rp, rt, rho)
    return h0 * gamma0 + hw * gammaw

# =====================================================================
#   2. ITU-R P.618-13 - CORRECTED IMPLEMENTATION
# =====================================================================

def rain_k_alpha(f_ghz, el_deg, tau_deg):
    f = f_ghz
    theta = np.radians(el_deg)
    tau = np.radians(tau_deg)
    # ITU-R P.838-3 Coefficients (Simplified Table Lookup)
    aH = (-5.33980, -0.35351, -0.23789, -0.94158, -1.61838)
    bH = (-0.10008,  1.26970,  0.86036,  0.64552,  0.35437)
    cH = (1.13098,   0.45400,  0.15354,  0.16817,  0.13545)
    mH = -0.18961
    KH = 0.67849
    fH = -5.07504
    logf = np.log10(f)
    logkH = sum(a * np.exp(-((logf - b) ** 2) / c) for a, b, c in zip(aH, bH, cH))
    logkH += mH * logf + fH
    kH = 10 ** logkH
    
    aV = (-3.80595, -3.44965, -0.39902,  0.50167)
    bV = ( 0.56934, -0.22911,  0.73042,  1.07319)
    cV = ( 0.81061,  0.51059,  0.11899,  0.27195)
    mV = -0.16398
    KV =  0.63297
    fV = -5.00233
    logkV = sum(a * np.exp(-((logf - b) ** 2) / c) for a, b, c in zip(aV, bV, cV))
    logkV += mV * logf + fV
    kV = 10 ** logkV

    alphaH = 1.0 + (np.log10(KH) - np.log10(kH)) / (1 - mH)
    alphaV = 1.0 + (np.log10(KV) - np.log10(kV)) / (1 - mV)

    # Correct Geometry (P.838-3 Eq 3 & 4)
    k = (kH + kV + (kH - kV) * (np.cos(theta)**2) * np.cos(2 * tau)) / 2
    numerator_alpha = (kH * alphaH + kV * alphaV + (kH * alphaH - kV * alphaV) * (np.cos(theta)**2) * np.cos(2 * tau))
    alpha = numerator_alpha / (2 * k)
    return k, alpha

def rain_attenuation_p618(f_ghz, el_deg, hs, hr, R001, tau_deg=45):
    k, alpha = rain_k_alpha(f_ghz, el_deg, tau_deg)
    gamma_R = k * (R001 ** alpha)
    el_rad = np.radians(el_deg)
    
    if el_deg >= 5: Ls = (hr - hs) / np.sin(el_rad)
    else:
        Re = 8500
        Ls = (2 * (hr - hs)) / (np.sqrt(np.sin(el_rad)**2 + 2*(hr - hs)/Re) + np.sin(el_rad))
        
    Lg = Ls * np.cos(el_rad)
    r001 = 1 / (1 + 0.78 * np.sqrt(Lg * gamma_R / f_ghz) - 0.38 * (1 - np.exp(-2 * Lg)))
    
    # Correct Vertical Adjustment Factor
    term_sqrt = np.sqrt(Lg * gamma_R) / (f_ghz**2) 
    v001 = 1 / (1 + np.sqrt(np.sin(el_rad)) * (31 * (1 - np.exp(- (el_deg / (1 + 0)))) * term_sqrt - 0.45))
    
    Leff = Ls * r001 * v001
    return gamma_R * Leff

def rain_attenuation_exceeded(A001, el_deg, p):
    # P.618-13 Scaling
    if p >= 1: beta = 0
    else: beta = 0.855 + 0.546 * np.exp(-0.043 * el_deg)
    
    if p >= 1: Ap = A001 * (p / 0.01)**(-(0.12 + 0.4 * (np.log10(p / 0.01))**0.5))
    else:
        exponent = -(0.655 + 0.033 * np.log(p) - 0.045 * np.log(A001) - beta * (1 - p) * np.sin(np.radians(el_deg)))
        Ap = A001 * (p / 0.01)**exponent
    return Ap

def scintillation_p618(f_ghz, el_deg, p_percent):
    el = np.radians(el_deg)
    sigma_ref = 0.0036 + 0.0001 * f_ghz
    sigma = sigma_ref / (np.sin(el)**1.2)
    lp = np.log10(p_percent)
    ap = -0.061 * lp**3 + 0.072 * lp**2 - 1.71 * lp + 3.0
    return ap * sigma

def cloud_attenuation_p840(f_ghz, el_deg, L):
    # Simplified placeholder for visualization
    return L * (f_ghz/20.0)**2 / np.sin(np.radians(el_deg)) 

def itu618_total(f_ghz, el_deg, hs, hr, R001, p_percent, cloud_L, pdry, T, rho):
    # 1. Gaseous (P.676)
    A_zenith = itu676(pdry, T, rho, f_ghz)
    A_gas = A_zenith / np.sin(np.radians(el_deg))
    
    # 2. Rain (P.618)
    A001 = rain_attenuation_p618(f_ghz, el_deg, hs, hr, R001)
    A_rain = rain_attenuation_exceeded(A001, el_deg, p_percent)
    
    # 3. Cloud (Simplified)
    A_cloud = cloud_attenuation_p840(f_ghz, el_deg, cloud_L)
    
    # 4. Scintillation
    A_scint = scintillation_p618(f_ghz, el_deg, p_percent)
    
    # Total
    A_total = A_gas + np.sqrt((A_rain + A_cloud)**2 + A_scint**2)
    
    return A_gas, A_total

# =====================================================================
#   3. PLOTTING EXECUTION
# =====================================================================

# Parameters
f_range = np.linspace(10, 100, 100)
el_deg = 30
hs = 0.1 # Station height km
hr = 3.0 # Rain height km
pdry = 1013
T = 290
rho = 7.5 # g/m3

# Configuration 1: Heavy Rain (0.01%)
R_heavy = 50
p_heavy = 0.01
L_heavy = 0.5 

# Configuration 2: Light Rain (0.1%)
R_light = 10
p_light = 0.1
L_light = 0.1

gas_1, total_1, total_2 = [], [], []

for f in f_range:
    g, t = itu618_total(f, el_deg, hs, hr, R_heavy, p_heavy, L_heavy, pdry, T, rho)
    g2, t2 = itu618_total(f, el_deg, hs, hr, R_light, p_light, L_light, pdry, T, rho)
    gas_1.append(g)
    total_1.append(t)
    total_2.append(t2)

plt.figure(figsize=(14, 6))

# Plot 1: Attenuation Spectrum
plt.subplot(1, 2, 1)
plt.plot(f_range, gas_1, 'b--', label='ITU-676 (Gas Only)')
plt.plot(f_range, total_1, 'r-', label=f'ITU-618 Total ({p_heavy}%, R={R_heavy})')
plt.plot(f_range, total_2, 'g-', label=f'ITU-618 Total ({p_light}%, R={R_light})')
plt.fill_between(f_range, gas_1, total_1, color='red', alpha=0.1, label='Rain/Cloud Contribution')
plt.title(f'ITU-676 vs ITU-618 Attenuation (Elev={el_deg}°)')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Attenuation [dB]')
plt.grid(True, alpha=0.5)
plt.legend()

# Plot 2: Elevation Dependence (at 30 GHz)
f_fixed = 30
el_range = np.linspace(5, 90, 50)
gas_el, total_el_heavy = [], []

for el in el_range:
    g, t = itu618_total(f_fixed, el, hs, hr, R_heavy, p_heavy, L_heavy, pdry, T, rho)
    gas_el.append(g)
    total_el_heavy.append(t)

plt.subplot(1, 2, 2)
plt.plot(el_range, gas_el, 'b--', label='ITU-676 Baseline')
plt.plot(el_range, total_el_heavy, 'r-', label='ITU-618 Total (Heavy Rain)')
plt.fill_between(el_range, gas_el, total_el_heavy, color='orange', alpha=0.2, label='Weather Penalty')
plt.title(f'Elevation Dependence at {f_fixed} GHz')
plt.xlabel('Elevation Angle [deg]')
plt.ylabel('Attenuation [dB]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()