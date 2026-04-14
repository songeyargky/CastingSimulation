<<<<<<< HEAD
# Low Chrome Mill Ball Casting Simulation

1D finite difference simulation of solidification in spherical mill balls.

## Setup
1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies: `pip install numpy matplotlib pandas`
4. Run simulations from `src/` (to be implemented).

## Configuration
Edit `config.py` to change material properties and default parameters.

# Day 2 – Geometry Module & Grid Setup

## What we did

We built the **grid module** that divides the mill ball into 40 radial slices from the center to the surface.  
This module calculates:

- `r` – radial positions (meters) of each node
- `dr` – distance between two neighboring nodes (spatial step)
- `dt` – stable time step (seconds) based on the Von Neumann stability criterion

All numbers come from the central configuration file (`config.py`).

---

## Files created

- `src/grid.py` – contains the `create_grid()` function

---

## How to test the module

Activate your virtual environment and run:

```bash
source .venv/Scripts/activate
python -c "from src.grid import create_grid; r, dr, dt = create_grid(100, 40); print(f'dr = {dr*1000:.3f} mm, dt = {dt:.4f} s')"

