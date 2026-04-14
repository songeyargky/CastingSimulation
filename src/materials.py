# src/materials.py
# Temperature-dependent thermophysical properties for low chrome white cast iron

import sys
import os

# Add parent folder to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_rho(T):
    """
    Density as a function of temperature.
    T : temperature in °C
    Returns density in kg/m³
    """
    if T >= config.T_liquidus:
        # Fully liquid
        return config.rho_l
    elif T <= config.T_solidus:
        # Fully solid
        return config.rho_s
    else:
        # Mushy zone: linear interpolation between liquid and solid
        # Calculate liquid fraction
        f_l = (T - config.T_solidus) / (config.T_liquidus - config.T_solidus)
        rho = f_l * config.rho_l + (1 - f_l) * config.rho_s
        return rho

def get_k(T):
    """
    Thermal conductivity as a function of temperature.
    T : temperature in °C
    Returns k in W/m·K
    """
    if T >= config.T_liquidus:
        return config.k_l
    elif T <= config.T_solidus:
        return config.k_s
    else:
        # Mushy zone: linear interpolation
        f_l = (T - config.T_solidus) / (config.T_liquidus - config.T_solidus)
        k = f_l * config.k_l + (1 - f_l) * config.k_s
        return k

def get_cp(T):
    """
    Specific heat capacity (or effective Cp in mushy zone) as function of temperature.
    In the mushy zone, we add the latent heat effect using the apparent heat capacity method.
    T : temperature in °C
    Returns Cp in J/kg·K
    """
    if T >= config.T_liquidus:
        # Fully liquid
        return config.cp_l
    elif T <= config.T_solidus:
        # Fully solid
        return config.cp_s
    else:
        # Mushy zone: effective Cp = Cp_s + L / (T_liquidus - T_solidus)
        # This spreads the total latent heat evenly over the whole mushy range.
        # Note: For a two‑stage model (primary + eutectic), we would refine this later.
        # For now, a single average works for initial testing.
        L_total = config.L_total
        delta_T_mush = config.T_liquidus - config.T_solidus
        cp_eff = config.cp_s + L_total / delta_T_mush
        return cp_eff

# ---- Optional: advanced two‑stage Cp (for later, if you want to see the eutectic plateau) ----
def get_cp_two_stage(T):
    """
    Two‑stage effective Cp: primary dendrite stage + eutectic reaction.
    This gives a more realistic cooling curve with a distinct eutectic shelf.
    """
    if T >= config.T_liquidus:
        return config.cp_l
    elif T <= config.T_solidus:
        return config.cp_s
    elif T > config.T_eutectic:
        # Primary dendrite stage (above eutectic temperature)
        # Use only primary latent heat
        delta_T_primary = config.T_liquidus - config.T_eutectic
        cp_eff = config.cp_s + config.L_primary / delta_T_primary
        return cp_eff
    else:
        # Eutectic stage (at or below eutectic, down to solidus)
        # Use a very narrow temperature band to simulate the isothermal reaction
        delta_T_eutectic = 1.0  # 1°C band to avoid infinite Cp
        cp_eff = config.cp_s + config.L_eutectic / delta_T_eutectic
        return cp_eff

# ---- Quick test when you run this file directly ----
if __name__ == "__main__":
    print("=== Testing materials module ===")
    test_temps = [1250, 1175, 1147, 1100, 1000]  # °C: liquid, mushy, eutectic, solidus, solid
    for T in test_temps:
        rho = get_rho(T)
        k = get_k(T)
        cp = get_cp(T)
        print(f"T = {T}°C: ρ = {rho:.0f} kg/m³, k = {k:.1f} W/m·K, Cp = {cp:.0f} J/kg·K")
    
    print("\n--- Two‑stage Cp (eutectic peak) ---")
    for T in test_temps:
        cp2 = get_cp_two_stage(T)
        print(f"T = {T}°C: Cp_two_stage = {cp2:.0f} J/kg·K")