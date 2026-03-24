# config.py
# Material properties for low chrome white cast iron (constant properties)

# Thermal properties
rho = 7800.0          # density (kg/m³)
cp = 600.0            # specific heat capacity (J/kg·K)
k = 35.0              # thermal conductivity (W/m·K)
L = 270000.0          # latent heat of fusion (J/kg)

# Phase change temperatures
T_liquidus = 1500.0   # °C
T_solidus = 1450.0    # °C

# Mechanical properties (for stress estimate, optional)
E = 200e9             # Young's modulus (Pa) – typical for cast iron
alpha_exp = 12e-6     # coefficient of thermal expansion (1/K)

# Simulation defaults (can be overridden later)
default_N = 40        # number of radial nodes
default_dt = 0.05     # time step (s)
default_total_time = 700.0  # maximum simulation time (s)

# Boundary condition defaults
default_h = 600.0     # heat transfer coefficient (W/m²·K) – middle of typical range
default_mold_temp = 250.0  # °C
default_pour_temp = 1500.0 # °C
default_ball_diameter = 0.100  # 100 mm in meters

def print_properties():
    """Display all material properties."""
    print("Material Properties (Low Chrome White Cast Iron):")
    print(f"  Density (rho)         = {rho} kg/m³")
    print(f"  Specific heat (cp)    = {cp} J/kg·K")
    print(f"  Thermal conductivity (k) = {k} W/m·K")
    print(f"  Latent heat (L)       = {L} J/kg")
    print(f"  Liquidus temp (Tl)    = {T_liquidus} °C")
    print(f"  Solidus temp (Ts)     = {T_solidus} °C")
    print(f"  Young's modulus (E)   = {E} Pa")
    print(f"  Thermal expansion (α) = {alpha_exp} 1/K")