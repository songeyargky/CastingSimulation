# src/materials.py
# Thermophysical properties for low-chrome white cast iron.
#
# KEY FIX: get_solid_fraction now matches the two-stage Cp model exactly.
#
# PREVIOUS MODEL:
#   fs jumped from ~0.547 to 1.000 instantaneously at T = T_eutectic (1147C).
#   This was inconsistent with the Cp model which releases L_eutectic smoothly
#   over the band [T_eutectic - delta_T_eutectic, T_eutectic] = [1142, 1147C].
#
#   Consequence: a node at 1146.9C had fs=1.00 (fully solid) but cp=38,400
#   J/kg/K (eutectic latent heat being released). The two models disagreed
#   about the physical state, confusing the mushy-zone tracking used by the
#   defect calculations (Niyama, HCS).
#
# FIXED MODEL:
#   fs transitions linearly from f_E (=0.55) to 1.0 over the eutectic band:
#       fs(T) = f_E + (1 - f_E) * (T_eutectic - T) / delta_T_eutectic
#   for T in [T_eutectic - delta_T_eutectic, T_eutectic].
#
#   At T = T_eutectic (1147C):       fs = 0.55  (continuous with primary stage)
#   At T = T_eutectic - 5C (1142C):  fs = 1.00  (matches where cp drops back)
#   Below 1142C:                      fs = 1.00  (fully solid)
#
#   This adds 5 degrees of mushy-zone width that was previously invisible,
#   improving Niyama computation and HCS tracking for the eutectic transformation.

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_rho(T):
    if T >= config.T_liquidus:
        return config.rho_l
    elif T <= config.T_solidus:
        return config.rho_s
    else:
        f_l = (T - config.T_solidus) / (config.T_liquidus - config.T_solidus)
        return f_l * config.rho_l + (1 - f_l) * config.rho_s


def get_k(T):
    if T >= config.T_liquidus:
        return config.k_l
    elif T <= config.T_solidus:
        return config.k_s
    else:
        f_l = (T - config.T_solidus) / (config.T_liquidus - config.T_solidus)
        return f_l * config.k_l + (1 - f_l) * config.k_s


def get_cp(T):
    """
    Two-stage effective Cp:
      Liquid:          cp_l
      Primary mushy:   cp_s + L_primary / (T_liquidus - T_eutectic)
      Eutectic band:   cp_s + L_eutectic / delta_T_eutectic
      Solid:           cp_s
    """
    delta_T_eutectic = getattr(config, 'delta_T_eutectic', 5.0)
    T_eut = config.T_eutectic

    if T >= config.T_liquidus:
        return config.cp_l
    elif T <= config.T_solidus:
        return config.cp_s
    elif T > T_eut:
        delta_T_primary = config.T_liquidus - T_eut
        return config.cp_s + config.L_primary / delta_T_primary
    elif T >= T_eut - delta_T_eutectic:
        return config.cp_s + config.L_eutectic / delta_T_eutectic
    else:
        return config.cp_s


def get_solid_fraction(T):
    """
    Two-stage solid fraction, NOW CONSISTENT with the two-stage Cp model.

    Stage 1 (primary dendrites):  fs varies linearly from 0 to f_E
                                  over [T_eutectic, T_liquidus]
    Stage 2 (eutectic reaction):  fs varies linearly from f_E to 1.0
                                  over [T_eutectic - delta_T_eutectic, T_eutectic]
    Below eutectic band:          fs = 1.0  (fully solid)

    CHANGE from old model: the old code had fs jump from ~0.547 to 1.0
    instantaneously at T_eutectic. This version transitions smoothly over
    the same 5-degree band used by the Cp model for latent heat release.
    """
    delta_T_eutectic = getattr(config, 'delta_T_eutectic', 5.0)
    T_eut   = config.T_eutectic
    T_eut_end = T_eut - delta_T_eutectic   # temperature where eutectic is complete
    f_E     = getattr(config, 'f_E', 0.55)

    if T >= config.T_liquidus:
        return 0.0

    elif T <= T_eut_end:
        # Below the eutectic band: fully solid (covers T_solidus and below too)
        return 1.0

    elif T > T_eut:
        # Primary solidification: linear from 0 at T_liquidus to f_E at T_eutectic
        return f_E * (config.T_liquidus - T) / (config.T_liquidus - T_eut)

    else:
        # Eutectic band: linear from f_E at T_eutectic to 1.0 at T_eut_end
        # t=0 at T_eut, t=1 at T_eut_end (= T_eut - delta_T_eutectic)
        t = (T_eut - T) / delta_T_eutectic
        return f_E + (1.0 - f_E) * t


# Enthalpy functions (kept for reference, not used by temperature-based solver)
_CP_M   = 0.5 * (config.cp_s + config.cp_l)
_H_SOL  = config.cp_s * config.T_eutectic
_H_EUT  = _H_SOL + config.L_eutectic
_DHDT_M = _CP_M + config.L_primary / (config.T_liquidus - config.T_eutectic)
_H_LIQ  = _H_EUT + _DHDT_M * (config.T_liquidus - config.T_eutectic)


def H_from_T(T):
    if T <= config.T_eutectic:
        return config.cp_s * T
    elif T <= config.T_liquidus:
        return _H_EUT + _DHDT_M * (T - config.T_eutectic)
    else:
        return _H_LIQ + config.cp_l * (T - config.T_liquidus)


def T_from_H(H):
    if H <= _H_SOL:
        return H / config.cp_s
    elif H <= _H_EUT:
        return config.T_eutectic
    elif H <= _H_LIQ:
        return config.T_eutectic + (H - _H_EUT) / _DHDT_M
    else:
        return config.T_liquidus + (H - _H_LIQ) / config.cp_l


def get_solid_fraction_profile(T_arr):
    return np.array([get_solid_fraction(t) for t in T_arr])


if __name__ == "__main__":
    print("Material model consistency check\n")
    dT_eut = getattr(config, 'delta_T_eutectic', 5.0)
    T_eut  = config.T_eutectic
    print(f"Eutectic band: {T_eut - dT_eut:.1f} - {T_eut:.1f} C")
    print(f"{'T (C)':>8} | {'rho':>8} | {'k':>6} | {'cp':>8} | {'fs':>8} | {'phase'}")
    print("-" * 65)
    for T in [1260, 1250, 1200, 1148, 1147.1, 1147.0, 1146.0,
              1144.0, 1142.1, 1142.0, 1141.0, 1100, 900]:
        rho = get_rho(T); k = get_k(T); cp = get_cp(T); fs = get_solid_fraction(T)
        if T > config.T_liquidus:          phase = "liquid"
        elif T > T_eut:                    phase = "primary mushy"
        elif T >= T_eut - dT_eut:          phase = "EUTECTIC BAND"
        elif T > config.T_solidus:         phase = "solid (warm)"
        else:                              phase = "solid"
        print(f"{T:8.1f} | {rho:8.0f} | {k:6.1f} | {cp:8.0f} | {fs:8.4f} | {phase}")
