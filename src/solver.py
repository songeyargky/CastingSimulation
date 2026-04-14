# src/solver.py
# Thermal solver using explicit finite difference method for spherical coordinates

import sys
import os
import numpy as np

# Add parent folder to path so we can import config and materials
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import src.materials as mat

def update_temperature(T, r, dr, dt, T_mold, h, debug=False):
    """
    Update temperature array for one time step using explicit FDM in spherical coordinates.
    
    Parameters:
    T : numpy array (current temperatures in °C, size N_nodes)
    r : numpy array (radial positions in meters, size N_nodes)
    dr : float (spatial step, meters)
    dt : float (time step, seconds)
    T_mold : float (mold temperature, °C)
    h : float (heat transfer coefficient, W/m²K)
    debug : bool (if True, print diagnostics)
    
    Returns:
    T_new : numpy array (updated temperatures)
    """
    N = len(T)
    T_new = T.copy()
    
    # Pre-compute material properties at each node (temperature-dependent)
    rho = np.array([mat.get_rho(T[i]) for i in range(N)])
    cp = np.array([mat.get_cp(T[i]) for i in range(N)])
    k = np.array([mat.get_k(T[i]) for i in range(N)])
    
    # Thermal diffusivity alpha = k / (rho * cp)
    alpha = k / (rho * cp)
    
    # ---- Center node (i = 0) - special spherical symmetry (3× rule) ----
    # Formula: T0_new = T0 + 6 * alpha * dt / dr^2 * (T1 - T0)
    # The factor 6 (not 2) comes from L'Hôpital's rule at r=0
    alpha0 = alpha[0]
    T_new[0] = T[0] + 6.0 * alpha0 * dt / (dr * dr) * (T[1] - T[0])
    
    # ---- Interior nodes (i = 1 to N-2) - spherical FDM ----
    for i in range(1, N-1):
        # Avoid division by zero at r=0 (i=0 already handled)
        ri = r[i]
        if ri <= 0:
            continue
        
        alpha_i = alpha[i]
        
        # Standard diffusion term: (T_{i+1} - 2T_i + T_{i-1})
        diffusion = T[i+1] - 2.0*T[i] + T[i-1]
        
        # Spherical correction term: (2/ri) * (T_{i+1} - T_{i-1}) / (2*dr) ? 
        # Actually the derived formula: 
        # T_new = T + alpha*dt/dr^2 * (T_{i+1} - 2T_i + T_{i-1}) 
        #         + alpha*dt/(ri*dr) * (T_{i+1} - T_{i-1})
        # Let's implement directly:
        correction = (alpha_i * dt / (ri * dr)) * (T[i+1] - T[i-1])
        
        T_new[i] = T[i] + alpha_i * dt / (dr*dr) * diffusion + correction
    
    # ---- Surface node (i = N-1) - convective + radiative boundary condition ----
    # Use half-control-volume energy balance
    i_surf = N-1
    rho_surf = rho[i_surf]
    cp_surf = cp[i_surf]
    k_surf = k[i_surf]
    
    # Convective heat flux
    q_conv = h * (T[i_surf] - T_mold)
    
    # Radiative heat flux (if included)
    # Convert temperatures to Kelvin for radiation
    T_surf_K = T[i_surf] + 273.15
    T_mold_K = T_mold + 273.15
    q_rad = config.emissivity * config.sigma * (T_surf_K**4 - T_mold_K**4)
    
    # Total heat flux leaving the surface
    q_total = q_conv + q_rad
    
    # Update surface node (half control volume thickness = dr/2)
    # Energy balance: rho*cp * (dr/2) * dT/dt = -q_total
    # So dT/dt = -2 * q_total / (rho*cp*dr)
    # Explicit update: T_new = T - 2 * dt * q_total / (rho*cp*dr)
    T_new[i_surf] = T[i_surf] - 2.0 * dt * q_total / (rho_surf * cp_surf * dr)
    
    # Optional: print debug info
    if debug:
        print(f"Center temp: {T_new[0]:.1f}°C, Surface temp: {T_new[i_surf]:.1f}°C")
        print(f"  q_conv = {q_conv:.0f} W/m², q_rad = {q_rad:.0f} W/m²")
    
    return T_new

# ---- Quick test when you run this file directly ----
if __name__ == "__main__":
    # Create a simple grid for testing (100 mm ball)
    from src.grid import create_grid
    
    diameter = 0.100  # 100 mm
    N = config.N_nodes
    r, dr, dt = create_grid(diameter * 1000, N)  # create_grid expects mm
    print(f"Test grid: dr = {dr*1000:.3f} mm, dt = {dt:.4f} s")
    
    # Initial temperature: uniform at pouring temperature
    T_initial = np.full(N, config.base_case["T_pour"])
    T_mold = config.base_case["T_mold"]
    h = config.base_case["h"]
    
    # Perform one time step
    T_new = update_temperature(T_initial, r, dr, dt, T_mold, h, debug=True)
    
    # Check if temperature changed (should decrease at surface, center unchanged initially)
    print(f"\nMax temperature change: {np.max(np.abs(T_new - T_initial)):.2f}°C")