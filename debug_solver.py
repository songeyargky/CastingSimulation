import numpy as np
from src.grid import create_grid
from src.solver import update_temperature
import config

# Setup
diameter_mm = 100
N = config.N_nodes
r, dr, dt = create_grid(diameter_mm, N)
T_mold = config.base_case["T_mold"]
h = config.base_case["h"]

print(f"dt = {dt:.6f} s")
print(f"dr = {dr*1000:.3f} mm")
print(f"N_nodes = {N}")

# Initial condition
T = np.full(N, config.base_case["T_pour"])
print(f"Initial T: min={T.min():.1f}, max={T.max():.1f}")

# Run a few time steps, printing every step
for step in range(10):
    T = update_temperature(T, r, dr, dt, T_mold, h, debug=True)
    print(f"Step {step+1}: Center={T[0]:.1f}, Surface={T[-1]:.1f}")
    if np.any(np.isnan(T)) or np.any(T < 0) or np.any(T > 3000):
        print("!!! Numerical blowup detected !!!")
        break