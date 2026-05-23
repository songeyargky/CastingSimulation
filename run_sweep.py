# run_parameter_sweep.py – Full parameter sweep (T_pour, T_mold, diameter only)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
import src.materials as mat
from src.grid import create_grid
from src.solver import update_temperature
from src.defects import (
    compute_misrun_risk,
    compute_cold_shut_risk,
    compute_surface_crack_index,
    compute_warpage_index,
)

# ============================================================================
# Simulation settings
# ============================================================================
MAX_SIM_TIME     = config.max_sim_time
SAVE_INTERVAL    = 1.0          # seconds between snapshots
EQUIL_TOL        = 5.0          # stop when all nodes within this of T_mold
SAVE_COOLING_PLOTS = False      # set True to save individual cooling curves (many files)
CHVORINOV_C      = config.Chvorinov_C
CHVORINOV_TOL    = config.Chvorinov_tol

# Output directory
os.makedirs("sweep_results", exist_ok=True)
if SAVE_COOLING_PLOTS:
    os.makedirs("cooling_curves", exist_ok=True)

# ============================================================================
# Single simulation runner
# ============================================================================
def run_single_case(T_pour, T_mold, diameter_mm):
    """
    Run one simulation and return:
        solidification_time (s)
        defect risks (MRI, CSRI, SCI, WI_risk)
        cooling rate profile summary (mean, max)
        thermal gradient profile summary (mean, max)
        chvorinov validation
    """
    N = config.N_nodes
    r, dr, dt = create_grid(diameter_mm, N)

    T = np.ones(N) * T_pour
    surface_fs = 0.0
    time_val = 0.0
    next_save = SAVE_INTERVAL

    time_points = []
    centre_temps = []
    surface_temps = []
    T_history = []
    fs_history = []

    solidification_time = None

    while time_val < MAX_SIM_TIME:
        T, surface_fs = update_temperature(
            T, r, dr, dt, T_mold,
            h_initial=config.h_initial,
            h_gap=config.h_gap,
            time=time_val
        )
        time_val += dt

        if time_val >= next_save:
            time_points.append(time_val)
            centre_temps.append(T[0])
            surface_temps.append(T[-1])
            T_history.append(T.copy())
            fs_history.append(mat.get_solid_fraction_profile(T))
            next_save += SAVE_INTERVAL

            # Solidification time: centre first reaches T_solidus
            if solidification_time is None and T[0] <= config.T_solidus:
                solidification_time = time_val

            # Early stop if equilibrated
            if np.all(T - T_mold < EQUIL_TOL):
                break

        if np.any(np.isnan(T)) or np.any(T < 0) or np.any(T > 3500):
            print(f"Instability at t={time_val:.2f}s – skipping case")
            return None

    if solidification_time is None:
        solidification_time = time_val   # not fully solidified within window

    # Compute defect risks
    MRI, superheat, t_surf_liq, _ = compute_misrun_risk(
        T_history, fs_history, time_points, T_pour, config.h_initial
    )
    CSRI, CSR, T_surf_early, _ = compute_cold_shut_risk(
        T_history, time_points, T_pour, T_mold
    )
    SCI, t_surf_sol, max_cool_rate, _ = compute_surface_crack_index(
        T_history, fs_history, time_points
    )
    WI, t_full_solid, dT_post_solid, _ = compute_warpage_index(
        T_history, fs_history, time_points, r
    )
    WI_risk = float(np.clip(WI / 0.010, 0.0, 1.0))

    # Compute cooling rate and thermal gradient profiles (over solidification phase)
    # Find snapshot when centre first becomes fully solid
    i_sol_end = len(T_history) - 1
    for i, fs_arr in enumerate(fs_history):
        if fs_arr[0] >= 0.99:
            i_sol_end = i
            break

    if i_sol_end < 2:
        # Not enough data
        mean_cool = max_cool = mean_grad = max_grad = np.nan
    else:
        dt_save = SAVE_INTERVAL
        N_nodes = len(T_history[0])
        R_profile = np.zeros(N_nodes)
        n_used = 0
        for i in range(1, min(i_sol_end, len(T_history)-2)):
            R_step = np.abs(np.array(T_history[i+1]) - np.array(T_history[i-1])) / (2.0 * dt_save)
            R_profile += R_step
            n_used += 1
        if n_used > 0:
            R_profile /= n_used
        mean_cool = float(np.mean(R_profile))
        max_cool  = float(np.max(R_profile))

        G_profile = np.zeros(N_nodes)
        for i in range(i_sol_end + 1):
            G_step = np.abs(np.gradient(T_history[i], dr))
            G_profile = np.maximum(G_profile, G_step)
        mean_grad = float(np.mean(G_profile))
        max_grad  = float(np.max(G_profile))

    # Chvorinov validation
    t_chvorinov = CHVORINOV_C * (diameter_mm / 6.0) ** 2
    deviation = (solidification_time - t_chvorinov) / t_chvorinov * 100.0
    chvorinov_ok = abs(deviation) <= (CHVORINOV_TOL * 100.0)

    # Optional: save cooling curve plot
    if SAVE_COOLING_PLOTS and len(time_points) > 1:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(time_points, centre_temps, label='Centre', lw=2)
        ax.plot(time_points, surface_temps, label='Surface', lw=2)
        ax.axhline(config.T_liquidus, color='steelblue', ls=':', label=f'T_liq={config.T_liquidus}C')
        ax.axhline(config.T_solidus, color='steelblue', ls='--', label=f'T_sol={config.T_solidus}C')
        ax.axhline(T_mold, color='grey', ls='-.', label=f'T_mold={T_mold}C')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Cooling curve: D={diameter_mm}mm, T_pour={T_pour:.0f}C, T_mold={T_mold:.0f}C')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)
        fname = f"cooling_curves/sweep_D{diameter_mm}_Tp{int(T_pour)}_Tm{int(T_mold)}.png"
        plt.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close()

    return {
        "solidification_time_s": solidification_time,
        "MRI": MRI,
        "CSRI": CSRI,
        "SCI": SCI,
        "WI_risk": WI_risk,
        "mean_cooling_rate_Ks": mean_cool,
        "max_cooling_rate_Ks": max_cool,
        "mean_thermal_gradient_Km": mean_grad,
        "max_thermal_gradient_Km": max_grad,
        "chvorinov_time_s": t_chvorinov,
        "deviation_percent": deviation,
        "chvorinov_within_15pct": chvorinov_ok,
    }


# ============================================================================
# Main sweep
# ============================================================================
def main():
    # Get all parameter combinations from config
    params = config.get_experiment_parameters()   # returns list of (T_pour, T_mold, h, diameter_mm)
    # But we ignore h (use fixed h_initial, h_gap from config)
    # Extract only T_pour, T_mold, diameter_mm
    combinations = []
    for tp, tm, h_val, diam in params:
        # Only use if h_val matches config.h_initial? Actually config.h_initial is fixed.
        # We'll ignore the h from the tuple and just use tp, tm, diam.
        combinations.append((tp, tm, diam))
    # Remove duplicates (if any)
    combinations = list(set(combinations))
    print(f"Total unique (T_pour, T_mold, diameter) combinations: {len(combinations)}")

    results = []
    for tp, tm, diam in tqdm(combinations, desc="Sweep progress"):
        res = run_single_case(tp, tm, diam)
        if res is None:
            continue
        results.append({
            "T_pour": tp,
            "T_mold": tm,
            "diameter_mm": diam,
            **res
        })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = "sweep_results/parameter_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSweep completed. Results saved to {csv_path}")

    # Print summary
    print("\n=== Chvorinov validation summary ===")
    passed = df['chvorinov_within_15pct'].sum()
    total = len(df)
    print(f"  {passed}/{total} cases within ±15% of Chvorinov's rule.")

    print("\n  Deviation by diameter:")
    for diam in config.diameters_mm:
        sub = df[df['diameter_mm'] == diam]
        if not sub.empty:
            mean_dev = sub['deviation_percent'].mean()
            print(f"    {diam} mm: mean deviation = {mean_dev:.1f}%")

    # Optional: plot solidification time vs diameter for a reference case
    # Use the first T_pour, T_mold from the list as reference
    if not df.empty:
        ref_tp = config.T_pour_min
        ref_tm = config.T_mold_min
        ref_df = df[(df['T_pour'] == ref_tp) & (df['T_mold'] == ref_tm)]
        if not ref_df.empty:
            plt.figure()
            plt.plot(ref_df['diameter_mm'], ref_df['solidification_time_s'], 'o-', label='Simulated')
            diam_vals = np.array(sorted(config.diameters_mm))
            t_chv = CHVORINOV_C * (diam_vals / 6.0) ** 2
            plt.plot(diam_vals, t_chv, 's-', label='Chvorinov')
            plt.xlabel('Diameter (mm)')
            plt.ylabel('Solidification time (s)')
            plt.title(f'Comparison with Chvorinov rule (T_pour={ref_tp}C, T_mold={ref_tm}C)')
            plt.legend()
            plt.grid(True)
            plt.savefig("sweep_results/chvorinov_comparison.png", dpi=150)
            plt.close()

    print("\nSweep complete.")


if __name__ == "__main__":
    main()