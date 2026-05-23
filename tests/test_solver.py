import numpy as np
import matplotlib.pyplot as plt
from src.grid import create_grid
from src.solver import update_temperature
import config

# Setup
diameter_mm = 100
N = config.N_nodes
r, dr, dt = create_grid(diameter_mm, N)
T_mold = config.base_case["T_mold"]
h = config.base_case["h"]

# Initial condition
T = np.full(N, config.base_case["T_pour"])

# Time loop (run until solidification or 700 seconds)
time = 0.0
max_time = 100.0  # seconds
history_center = []
history_surface = []
time_points = []

while time < max_time:
    T = update_temperature(T, r, dr, dt, T_mold, h)
    time += dt
    history_center.append(T[0])
    history_surface.append(T[-1])
    time_points.append(time)
    if time % 10 < dt:  # print every ~10 seconds
        print(f"t = {time:.1f}s: Center = {T[0]:.0f}°C, Surface = {T[-1]:.0f}°C")

# Plot
plt.figure(figsize=(10,5))
plt.plot(time_points, history_center, label='Center', linewidth=2)
plt.plot(time_points, history_surface, label='Surface', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Cooling curves for 100 mm ball (base case)')
plt.legend()
plt.grid(True)
plt.savefig('cooling_curve_100s.png', dpi=150)
plt.show()