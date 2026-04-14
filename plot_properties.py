import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import src.materials as mat

temps = np.linspace(900, 1300, 200)
rho_vals = [mat.get_rho(T) for T in temps]
k_vals = [mat.get_k(T) for T in temps]
cp_vals = [mat.get_cp(T) for T in temps]
cp2_vals = [mat.get_cp_two_stage(T) for T in temps]

fig, axes = plt.subplots(3, 1, figsize=(8, 10))
axes[0].plot(temps, rho_vals)
axes[0].set_ylabel("Density (kg/m³)")
axes[0].axvline(1200, color='r', linestyle='--', label='Liquidus')
axes[0].axvline(1100, color='b', linestyle='--', label='Solidus')
axes[0].legend()

axes[1].plot(temps, k_vals)
axes[1].set_ylabel("Thermal conductivity (W/m·K)")

axes[2].plot(temps, cp_vals, label='Single‑stage effective Cp')
axes[2].plot(temps, cp2_vals, label='Two‑stage Cp (eutectic peak)', linestyle='--')
axes[2].set_ylabel("Specific heat (J/kg·K)")
axes[2].set_xlabel("Temperature (°C)")
axes[2].legend()
plt.tight_layout()
plt.savefig("figures/materials_properties.png", dpi=150)
plt.show()