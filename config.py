# config.py
# Low chrome white cast iron – permanent mold casting

# --- Thermal input parameters (ranges for experiments) ---
T_pour_range = [1300, 1400, 1450, 1500, 1550]   # °C
T_mold_range = [150, 200, 250, 300, 350]         # °C
diameters_mm = [60, 80, 100, 120]                # mm
h_range = [300, 600, 900, 1200]                  # W/m²K (interfacial heat transfer coefficient)

# --- Base case (used for validation and single runs) ---
base_case = {
    "T_pour": 1450,      # °C
    "T_mold": 200,       # °C
    "diameter_mm": 100,  # mm
    "h": 800,            # W/m²K
}

# --- Material properties (temperature-dependent, piecewise) ---
# Liquid zone (T > T_liquidus)
rho_l = 7000.0          # kg/m³
cp_l = 750.0            # J/kg·K
k_l = 35.0              # W/m·K

# Solid zone (T < T_solidus)
rho_s = 7300.0          # kg/m³
cp_s = 600.0            # J/kg·K
k_s = 25.0              # W/m·K

# Phase change temperatures
T_liquidus = 1200.0     # °C
T_solidus = 1100.0      # °C
T_eutectic = 1147.0     # °C

# Latent heat (total and split for two‑stage model)
L_total = 270000.0      # J/kg
L_primary = 0.3 * L_total   # ~81000 J/kg (austenite dendrites)
L_eutectic = 0.7 * L_total  # ~189000 J/kg (eutectic reaction)

# Solid fraction at eutectic start (end of primary stage)
f_E = 0.55              # dimensionless

# --- Simulation numerical parameters ---
N_nodes = 40            # number of radial nodes (after grid independence study)
safety_factor = 0.4     # Fourier number limit (Fo <= 0.4 for stability)
max_sim_time = 700.0    # seconds (upper limit, will stop earlier if fully solid)

# --- Air gap model (two‑stage h) ---
h_initial = 1200.0      # W/m²K (intimate contact, surface not fully solid)
h_gap = 300.0           # W/m²K (after surface solidifies)

# --- Radiation ---
emissivity = 0.85
sigma = 5.67e-8         # Stefan–Boltzmann constant (W/m²K⁴)

# --- Mechanical properties for Thermal Stress Index (TSI) ---
E_young = 200e9         # Pa (200 GPa)
alpha_cte = 12e-6       # 1/K (coefficient of thermal expansion)
coherency_fs = 0.85     # solid fraction above which stress can develop

# --- Chvorinov constant (for validation) ---
# t_s = C * (V/A)^2  with V/A = D/6,  C = 1.488 s/mm² (from literature, replace placeholder)
Chvorinov_C = 1.488     # s/mm²

def print_config():
    print("=== Low Chrome Mill Ball Simulation Configuration ===")
    print(f"Pouring temperature range: {T_pour_range} °C")
    print(f"Mold temperature range: {T_mold_range} °C")
    print(f"Ball diameters: {diameters_mm} mm")
    print(f"h range: {h_range} W/m²K")
    print(f"Base case: {base_case}")
    print(f"Liquidus: {T_liquidus}°C, Solidus: {T_solidus}°C, Eutectic: {T_eutectic}°C")
    print(f"Total latent heat: {L_total/1000:.1f} kJ/kg")
    print(f"Grid nodes: {N_nodes}, stability Fo limit: {safety_factor}")