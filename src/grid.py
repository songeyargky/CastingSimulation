# grid.py
# This module creates the radial grid and calculates the time step.

import numpy as np
import sys
import os

# Add the parent folder (project root) to Python's path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_grid(diameter_mm, N_nodes):
    """
    Creates a 1D radial grid for a sphere.
    
    Parameters:
    diameter_mm : float   - ball diameter in millimeters
    N_nodes : int         - number of nodes from center to surface
    
    Returns:
    r : numpy array       - radial positions (meters) from center (r=0) to surface (r=R)
    dr : float            - distance between two nodes (meters)
    dt : float            - stable time step (seconds)
    """
    # Convert diameter from mm to meters
    radius_m = (diameter_mm / 1000.0) / 2.0
    
    # Distance between nodes (spatial step)
    dr = radius_m / (N_nodes - 1)
    
    # Create array of radial positions: from 0 (center) to radius_m (surface)
    r = np.linspace(0, radius_m, N_nodes)
    
    # ---- Time step calculation (stability criterion) ----
    # We need thermal diffusivity of solid metal: alpha = k / (rho * Cp)
    # Use solid properties (the most restrictive for stability)
    alpha_solid = config.k_s / (config.rho_s * config.cp_s)
    
    # Maximum allowed time step based on Von Neumann stability (Fo <= 0.4)
    dt = config.safety_factor * (dr**2) / alpha_solid
    
    return r, dr, dt

# ---- Quick test when you run this file directly ----
if __name__ == "__main__":
    # Test with the base case ball diameter (100 mm) and N_nodes from config
    r, dr, dt = create_grid(config.base_case["diameter_mm"], config.N_nodes)
    
    print(f"Ball diameter: {config.base_case['diameter_mm']} mm")
    print(f"Number of nodes: {config.N_nodes}")
    print(f"Radius: {r[-1]*1000:.1f} mm")
    print(f"Spatial step dr: {dr*1000:.3f} mm")
    print(f"Time step dt: {dt:.4f} seconds")
    print("\nFirst 5 radial positions (meters):", r[:5])