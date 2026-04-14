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

# Time loop
time = 0.0
max_time = 10.0  # simulate 10 seconds
history_center = []
history_surface = []
time_points = []

while time < max_time:
    T = update_temperature(T, r, dr, dt, T_mold, h)
    time += dt
    history_center.append(T[0])
    history_surface.append(T[-1])
    time_points.append(time)

# Plot
plt.figure()
plt.plot(time_points, history_center, label='Center')
plt.plot(time_points, history_surface, label='Surface')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Cooling curves (first 10 seconds)')
plt.legend()
plt.grid(True)
plt.savefig('test_cooling_10s.png')
plt.show()