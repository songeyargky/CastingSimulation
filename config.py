# config.py  —  Low-Chrome Mill Ball Solidification Simulation
# ─────────────────────────────────────────────────────────────────────────────
#
# PARAMETER RANGES  (continuous — any value in range is valid)
# ─────────────────────────────────────────────────────────────────────────────
#
# T_pour: any value between 1250 and 1550 °C
#   1250 °C = T_liquidus (absolute minimum for filling — misrun risk critical)
#   1550 °C = practical upper limit for this alloy grade
#
# T_mold: any value between 25 and 450 °C
#   25 °C  = room temperature (cold mold, maximum undercooling)
#   450 °C = maximum practical preheat for permanent steel mold
#   Starting from 25°C rather than 100°C allows simulation of
#   cold-start foundry conditions (e.g. first cast of the shift,
#   winter workshop, outdoor/unheated foundry).
#
# T_ambient: ambient workshop temperature (°C)
#   Affects effective mold cooling: a cold ambient (5°C winter workshop)
#   increases the heat extraction from the mold back-face by convection
#   and radiation, effectively lowering T_mold over time.
#   The compensation factor adjusts h_initial to account for this:
#       h_eff = h_initial × (1 + ambient_compensation_factor)
#   where ambient_compensation_factor = h_conv_ambient × (T_mold - T_ambient)
#                                       / (h_initial × (T_liquidus - T_mold))
#
# EXPERIMENT LOOP SETTINGS
# ─────────────────────────
# The simulation loops over combinations of:
#   - T_pour:      sampled uniformly across [T_pour_min, T_pour_max]
#   - T_mold:      sampled uniformly across [T_mold_min, T_mold_max]
#   - h_initial:   sampled from h_range
#   - diameter_mm: from diameters_mm
#
# Set n_samples_T_pour and n_samples_T_mold to control grid density.
# Set use_random_sampling=True to use random sampling instead of grid.

import numpy as np

# ── Pouring temperature (°C) ──────────────────────────────────────────────
T_pour_min = 1250.0    # = T_liquidus (absolute floor)
T_pour_max = 1550.0
n_samples_T_pour = 25   # how many values to sample across the range

# ── Mold temperature (°C) — starts at room temperature ────────────────────
T_mold_min = 25.0      # room temperature (unpreheated mold)
T_mold_max = 450.0
n_samples_T_mold = 10

# ── Ambient (workshop) temperature (°C) ───────────────────────────────────
# Controls additional heat loss from mold back-face to environment.
# Typical values:
#   5  °C — cold winter workshop / outdoor foundry
#   15 °C — cool but controlled workshop
#   25 °C — standard room temperature (no compensation needed)
#   35 °C — warm/tropical foundry environment
T_ambient = 25.0

# How ambient temperature modifies effective heat extraction:
# When T_ambient < T_mold, the mold loses extra heat to the environment.
# This is modelled as an additive correction to h_initial.
#   h_ambient_conv = 15 W/m²K  (free convection from mold outer face)
#   q_extra = h_ambient_conv × (T_mold - T_ambient)
#   h_correction = q_extra / max(T_liquidus - T_mold, 1)
# The net effect: cold ambient effectively increases heat extraction
# and requires higher T_mold or T_pour to compensate.
h_ambient_conv = 15.0   # W/m²K — ambient convection from mold outer surface

# ── Ball diameters (mm) ───────────────────────────────────────────────────
diameters_mm = [60, 80, 100, 120]

# ── Heat transfer coefficients (W/m²K) ───────────────────────────────────
h_range = [300, 600, 900, 1200]

# ── Sampling mode ─────────────────────────────────────────────────────────
# False = evenly-spaced grid across T_pour_min/max and T_mold_min/max
# True  = random sampling (use experiment_random_seed for reproducibility)
use_random_sampling   = False
experiment_random_seed = 42

# ── Base case (single reference simulation) ───────────────────────────────
base_case = {
    "T_pour"      : 1500.0,
    "T_mold"      : 25.0,
    "diameter_mm" : 80,
    "h"           : 600,
}

# ── Material properties — low-chrome white cast iron ─────────────────────
rho_l = 7000.0
cp_l  = 750.0
k_l   = 35.0
rho_s = 7300.0
cp_s  = 600.0
k_s   = 12.0

# Phase change temperatures (°C)
T_liquidus = 1250.0
T_solidus  = 1100.0
T_eutectic = 1147.0

# Latent heat (two-stage)
L_total    = 270000.0
L_primary  = 81000.0
L_eutectic = 189000.0
f_E        = 0.55
delta_T_eutectic = 5.0

# ── Numerical parameters ──────────────────────────────────────────────────
N_nodes       = 40
safety_factor = 0.02
max_sim_time  = 700.0

# ── Air gap model ─────────────────────────────────────────────────────────
h_initial = 1200.0
h_gap     = 300.0

# Two-zone contact resistance
R_contact_metal = 0.001   # m²K/W — metal contact (fs < fs_gap_onset)
R_contact_gap   = 0.020   # m²K/W — air gap (fs >= fs_gap_onset)
fs_gap_onset    = 0.70
q_total_cap     = 150_000  # W/m² — flux cap (instability guard)

# SCI thresholds for permanent mold (K/s)
SCI_cr_low  = 20.0
SCI_cr_high = 50.0

# ── Radiation ─────────────────────────────────────────────────────────────
emissivity = 0.85
sigma      = 5.67e-8

# ── Mechanical properties ─────────────────────────────────────────────────
E_young      = 200e9
alpha_cte    = 12e-6
coherency_fs = 0.85

# ── Chvorinov's Rule ──────────────────────────────────────────────────────
# t_solidif = Chvorinov_C × (V/A)²
# V/A for a sphere of diameter D: (D/6) mm
# Units: t in seconds when V/A in mm
Chvorinov_C     = 1.488
Chvorinov_tol   = 0.15    # flag deviation > 15%

# ── Experiment output ─────────────────────────────────────────────────────
results_csv_path = "simulation_results.csv"


# ── Helper: generate parameter grid or random samples ─────────────────────
def get_experiment_parameters():
    """
    Return list of (T_pour, T_mold, h, diameter_mm) tuples for the
    experiment loop, using either evenly-spaced grid or random sampling.

    T_pour and T_mold are CONTINUOUS — any value in their range is valid.
    """
    rng = np.random.default_rng(experiment_random_seed)

    if use_random_sampling:
        T_pours = rng.uniform(T_pour_min, T_pour_max, n_samples_T_pour)
        T_molds = rng.uniform(T_mold_min, T_mold_max, n_samples_T_mold)
    else:
        T_pours = np.linspace(T_pour_min, T_pour_max, n_samples_T_pour)
        T_molds = np.linspace(T_mold_min, T_mold_max, n_samples_T_mold)

    params = []
    for tp in T_pours:
        for tm in T_molds:
            for h in h_range:
                for d in diameters_mm:
                    if tp > tm + 50:   # sanity: pour must be meaningfully above mold
                        params.append((float(tp), float(tm), int(h), int(d)))
    return params


def ambient_h_correction(T_mold_val):
    """
    Compute the additive correction to h_initial due to ambient heat loss.

    When T_ambient < T_mold, the mold outer face loses heat to the
    environment, effectively increasing the net heat extraction rate
    from the casting.

    Returns delta_h (W/m²K) to ADD to h_initial.
    """
    if T_ambient >= T_mold_val:
        return 0.0   # warm ambient: no extra extraction
    q_extra    = h_ambient_conv * (T_mold_val - T_ambient)
    delta_T_cc = max(T_liquidus - T_mold_val, 1.0)
    return q_extra / delta_T_cc


def print_config():
    print("=== Simulation Configuration ===")
    print(f"T_pour range   : {T_pour_min}–{T_pour_max} °C  ({n_samples_T_pour} samples)")
    print(f"T_mold range   : {T_mold_min}–{T_mold_max} °C  ({n_samples_T_mold} samples)")
    print(f"T_ambient      : {T_ambient} °C  (ambient compensation active)")
    print(f"Diameters      : {diameters_mm} mm")
    print(f"h_range        : {h_range} W/m²K")
    print(f"Random sampling: {use_random_sampling}")
    print(f"T_liquidus     : {T_liquidus} °C   T_solidus : {T_solidus} °C")
    print(f"Chvorinov C    : {Chvorinov_C}   tolerance: {Chvorinov_tol*100:.0f}%")