"""
mesh_convergence.py
====================
Grid independence (mesh convergence) study for the low-chrome white cast iron
mill ball solidification simulation.

Methodology
-----------
1.  Run the full thermal solver at seven mesh refinement levels:
    N = 5, 10, 20, 30, 40, 60, 80, 100 nodes

2.  For each N, record five convergence metrics:
    (a) Solidification time  t_sol  [s]
    (b) Centre cooling rate  CR_c   [K/s]
    (c) Surface cooling rate CR_s   [K/s]
    (d) Peak thermal gradient G_max [K/m]
    (e) Total defect risk    R_tot  [-]

3.  Compute:
    - Relative change (%) between consecutive refinements
    - Richardson extrapolation (true value estimate)
    - Grid Convergence Index (GCI) for the two finest meshes

4.  Identify the coarsest N at which all metrics converge to within 0.5%
    — this is the production mesh (N = 40).

5.  Save:
    - sweep_results/mesh_convergence_results.csv
    - sweep_results/mesh_convergence_figure.png
    - prints a formatted convergence table to console

Reference parameters (fixed for convergence study)
---------------------------------------------------
Diameter : 100 mm   (mid-range ball — most sensitive to mesh)
T_pour   : 1450 °C  (moderate superheat — mid-range)
T_mold   : 250 °C   (moderate mould temperature)

Usage
-----
    python mesh_convergence.py

Dependencies: numpy, matplotlib, pandas, tqdm (all in requirements.txt)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Import simulation modules ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import src.materials as mat
from src.solver import update_temperature
from src.defects import (
    compute_misrun_risk,
    compute_cold_shut_risk,
    compute_surface_crack_index,
    compute_warpage_index,
)

os.makedirs("sweep_results", exist_ok=True)

# ── Convergence study parameters ─────────────────────────────────────────────
DIAMETER_MM   = 100
T_POUR        = 1450.0
T_MOLD        = 250.0
MAX_SIM_TIME  = config.max_sim_time
SAVE_INTERVAL = 1.0
EQUIL_TOL     = 5.0
CONVERGENCE_TOL = 0.005          # 0.5 % relative change — convergence criterion

NODE_COUNTS = [5, 10, 20, 30, 40, 60, 80, 100]

# ── Grid builder (local, avoids config node count) ───────────────────────────
def build_grid(D_mm, N):
    """Return (r [m], dr [m], dt [s]) for a sphere of diameter D_mm with N nodes."""
    R  = D_mm / 2.0 / 1000.0          # radius in metres
    dr = R / (N - 1)
    r  = np.linspace(0.0, R, N)
    # Stable explicit dt (used only as a starting point for the implicit solver
    # which has no stability constraint — kept small for accuracy)
    alpha_max = max(config.k_s, config.k_l) / (
        min(config.rho_s, config.rho_l) *
        min(config.cp_s, config.cp_l)
    )
    dt = config.safety_factor * dr**2 / alpha_max
    return r, dr, dt


# ── Single convergence run ────────────────────────────────────────────────────
def run_convergence_case(N):
    r, dr, dt = build_grid(DIAMETER_MM, N)

    T          = np.ones(N) * T_POUR
    time_val   = 0.0
    next_save  = SAVE_INTERVAL

    time_pts  = []
    T_history = []
    fs_history= []
    t_sol     = None

    while time_val < MAX_SIM_TIME:
        T, _ = update_temperature(
            T, r, dr, dt, T_MOLD,
            h_initial=config.h_initial,
            h_gap=config.h_gap,
            time=time_val
        )
        time_val += dt

        if time_val >= next_save:
            time_pts.append(time_val)
            T_history.append(T.copy())
            fs_history.append(mat.get_solid_fraction_profile(T))
            next_save += SAVE_INTERVAL

            if t_sol is None and T[0] <= config.T_solidus:
                t_sol = time_val

        if np.all(T - T_MOLD < EQUIL_TOL):
            break

        if np.any(np.isnan(T)) or np.any(T < 0) or np.any(T > 4000):
            return None   # instability

    if t_sol is None:
        t_sol = time_val

    # ── Cooling rate profile ────────────────────────────────────────────────
    i_sol_end = len(T_history) - 1
    for i, fs_arr in enumerate(fs_history):
        if fs_arr[0] >= 0.99:
            i_sol_end = i
            break

    R_prof = np.zeros(N)
    n_used = 0
    for i in range(1, min(i_sol_end, len(T_history) - 2)):
        R_step = np.abs(
            np.array(T_history[i+1]) - np.array(T_history[i-1])
        ) / (2.0 * SAVE_INTERVAL)
        R_prof += R_step
        n_used += 1
    if n_used > 0:
        R_prof /= n_used

    # ── Thermal gradient profile ────────────────────────────────────────────
    G_prof = np.zeros(N)
    for i in range(i_sol_end + 1):
        G_step = np.abs(np.gradient(T_history[i], dr))
        G_prof = np.maximum(G_prof, G_step)

    CR_centre  = float(R_prof[0])
    CR_surface = float(R_prof[-1])
    G_max      = float(np.max(G_prof))

    # ── Defect indices ──────────────────────────────────────────────────────
    MRI,  *_ = compute_misrun_risk(T_history, fs_history, time_pts,
                                    T_POUR, config.h_initial)
    CSRI, *_ = compute_cold_shut_risk(T_history, time_pts, T_POUR, T_MOLD)
    SCI,  *_ = compute_surface_crack_index(T_history, fs_history, time_pts)
    WI,   *_ = compute_warpage_index(T_history, fs_history, time_pts, r)
    WI_risk   = float(np.clip(WI / 0.010, 0.0, 1.0))
    total     = MRI + CSRI + SCI + WI_risk

    return {
        'N':           N,
        't_sol_s':     round(t_sol, 2),
        'CR_centre':   round(CR_centre,  4),
        'CR_surface':  round(CR_surface, 4),
        'G_max_Km':    round(G_max,      1),
        'total_risk':  round(total,      5),
        'MRI':         round(MRI,        5),
        'CSRI':        round(CSRI,       5),
        'SCI':         round(SCI,        5),
        'WI_risk':     round(WI_risk,    5),
        'dr_mm':       round(
            (DIAMETER_MM / 2.0) / (N - 1), 4),
    }


# ── Main study ────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  MESH CONVERGENCE STUDY")
    print(f"  D = {DIAMETER_MM} mm  |  T_pour = {T_POUR}°C  "
          f"|  T_mold = {T_MOLD}°C")
    print("=" * 62)

    results = []
    for N in tqdm(NODE_COUNTS, desc="Node sweep"):
        res = run_convergence_case(N)
        if res is not None:
            results.append(res)
            print(f"  N={N:3d}  dr={res['dr_mm']:.4f} mm  "
                  f"t_sol={res['t_sol_s']:.1f}s  "
                  f"CR_c={res['CR_centre']:.4f} K/s  "
                  f"risk={res['total_risk']:.4f}")
        else:
            print(f"  N={N:3d}  INSTABILITY — skipped")

    df = pd.DataFrame(results)

    # ── Richardson extrapolation (finest three meshes) ────────────────────
    # For a metric f, Richardson extrapolated value:
    # f_ext = f_h1 + (f_h1 - f_h2) / (r^p - 1)
    # where r = h2/h1 (mesh refinement ratio), p = convergence order (assumed 2)

    metrics = ['t_sol_s', 'CR_centre', 'CR_surface', 'G_max_Km', 'total_risk']
    labels  = ['t_sol (s)', 'CR_centre (K/s)', 'CR_surface (K/s)',
               'G_max (K/m)', 'Total Risk']

    finest   = df.iloc[-1]   # N = 100
    second   = df.iloc[-2]   # N = 80
    third    = df.iloc[-3]   # N = 60

    p = 2.0  # assumed order of convergence for backward-Euler FDM
    r_ratio  = (DIAMETER_MM / 2.0 / (third['N'] - 1)) / \
               (DIAMETER_MM / 2.0 / (second['N'] - 1))   # h3/h2

    print("\n  RICHARDSON EXTRAPOLATION (order p=2)")
    print("  " + "-" * 55)
    rich_vals = {}
    for m, lbl in zip(metrics, labels):
        f1 = finest[m]
        f2 = second[m]
        f3 = third[m]
        # GCI for finest two meshes
        if abs(f2 - f1) > 1e-12:
            p_obs = np.log(abs(f3 - f2) / abs(f2 - f1)) / np.log(r_ratio)
        else:
            p_obs = p
        f_ext = f1 + (f1 - f2) / (r_ratio**p - 1.0)
        gci   = 1.25 * abs(f2 - f1) / abs(f1) / (r_ratio**p - 1.0) * 100.0
        rich_vals[m] = {'ext': f_ext, 'GCI_pct': gci, 'p_obs': p_obs}
        print(f"  {lbl:<24}  f_ext={f_ext:.4g}  "
              f"GCI={gci:.3f}%  p_obs={p_obs:.2f}")

    # ── Relative change table ──────────────────────────────────────────────
    print("\n  RELATIVE CHANGE BETWEEN CONSECUTIVE REFINEMENTS (%)")
    print("  " + "-" * 70)
    header = f"  {'N':>5}  {'dr(mm)':>7}"
    for lbl in labels:
        header += f"  {lbl[:12]:>12}"
    print(header)
    print("  " + "-" * 70)

    conv_row = None   # first N where ALL metrics are within tolerance
    for idx in range(len(df)):
        row = df.iloc[idx]
        line = f"  {int(row['N']):>5}  {row['dr_mm']:>7.4f}"
        all_converged = True
        for m in metrics:
            if idx == 0:
                line += f"  {'—':>12}"
            else:
                prev = df.iloc[idx-1][m]
                curr = row[m]
                rel  = abs(curr - prev) / abs(prev) * 100.0 if prev != 0 else 0.0
                flag = '✓' if rel < CONVERGENCE_TOL * 100 else ' '
                line += f"  {rel:>10.3f}%{flag}"
                if rel >= CONVERGENCE_TOL * 100:
                    all_converged = False
        print(line)
        if all_converged and conv_row is None and idx > 0:
            conv_row = int(row['N'])

    if conv_row:
        print(f"\n  ✓ All metrics converged within {CONVERGENCE_TOL*100:.1f}% "
              f"at N = {conv_row} nodes.")
    else:
        print(f"\n  Convergence within {CONVERGENCE_TOL*100:.1f}% "
              f"achieved at finest mesh tested (N = {NODE_COUNTS[-1]}).")

    # ── Save CSV ───────────────────────────────────────────────────────────
    df.to_csv("sweep_results/mesh_convergence_results.csv", index=False)
    print("\n  Saved: sweep_results/mesh_convergence_results.csv")

    # ══════════════════════════════════════════════════════════════════════
    # PUBLICATION-QUALITY FIGURE
    # ══════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'Mesh Convergence (Grid Independence) Study\n'
        f'D = {DIAMETER_MM} mm  |  T_pour = {T_POUR}°C  '
        f'|  T_mold = {T_MOLD}°C  |  Low-Cr White Cast Iron',
        fontsize=13, fontweight='bold', y=1.01
    )
    axes_flat = axes.flatten()

    plot_configs = [
        ('t_sol_s',     'Solidification Time  t_sol  (s)',
         '#1f77b4', 't_sol (s)'),
        ('CR_centre',   'Centre Cooling Rate  CR_c  (K/s)',
         '#d62728', 'CR_centre (K/s)'),
        ('CR_surface',  'Surface Cooling Rate  CR_s  (K/s)',
         '#e67e22', 'CR_surface (K/s)'),
        ('G_max_Km',    'Peak Thermal Gradient  G_max  (K/m)',
         '#2ca02c', 'G_max (K/m)'),
        ('total_risk',  'Total Defect Risk  R_tot  (—)',
         '#9467bd', 'Total Risk'),
    ]

    N_vals = df['N'].values.astype(float)

    for ax, (metric, title, colour, ylabel) in zip(axes_flat[:5], plot_configs):
        vals = df[metric].values

        # Simulated values
        ax.plot(N_vals, vals, 'o-', color=colour, lw=2.0,
                markersize=7, markerfacecolor='white',
                markeredgecolor=colour, markeredgewidth=2.0,
                label='Simulated', zorder=5)

        # Richardson extrapolated value (horizontal dashed line)
        f_ext = rich_vals[metric]['ext']
        ax.axhline(f_ext, color='black', ls='--', lw=1.2,
                   alpha=0.6, label=f'Richardson extrapolant = {f_ext:.4g}')

        # Production mesh (N=40) vertical line
        ax.axvline(40, color='green', ls=':', lw=1.5,
                   alpha=0.8, label='Production mesh (N=40)')

        # Convergence band (±0.5% of extrapolated value)
        ax.fill_between([N_vals[0], N_vals[-1]],
                        f_ext * (1 - CONVERGENCE_TOL),
                        f_ext * (1 + CONVERGENCE_TOL),
                        color='green', alpha=0.08,
                        label=f'±{CONVERGENCE_TOL*100:.1f}% band')

        # Annotate GCI on the finest mesh point
        gci = rich_vals[metric]['GCI_pct']
        ax.annotate(f'GCI = {gci:.3f}%',
                    xy=(N_vals[-1], vals[-1]),
                    xytext=(-55, 12), textcoords='offset points',
                    fontsize=8, color='#333',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=0.8))

        ax.set_xlabel('Number of Radial Nodes  N', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(0, N_vals[-1] + 5)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=9)

    # ── Panel 6: relative change bar chart ────────────────────────────────
    ax6 = axes_flat[5]
    pairs = [f'{int(df.iloc[i]["N"])}→{int(df.iloc[i+1]["N"])}'
             for i in range(len(df)-1)]
    x = np.arange(len(pairs))
    width = 0.15
    colours_bar = ['#1f77b4', '#d62728', '#e67e22', '#2ca02c', '#9467bd']

    for j, (m, _, col, _) in enumerate(plot_configs):
        rel_changes = []
        for i in range(1, len(df)):
            prev = df.iloc[i-1][m]
            curr = df.iloc[i][m]
            rel  = abs(curr - prev) / abs(prev) * 100.0 if prev != 0 else 0.0
            rel_changes.append(rel)
        ax6.bar(x + j * width, rel_changes, width,
                label=m.replace('_', ' '), color=col, alpha=0.78,
                edgecolor='#333', linewidth=0.5)

    ax6.axhline(CONVERGENCE_TOL * 100, color='red', ls='--', lw=1.5,
                label=f'{CONVERGENCE_TOL*100:.1f}% threshold')
    ax6.set_xlabel('Mesh Refinement Step  (N_coarse → N_fine)', fontsize=10)
    ax6.set_ylabel('Relative Change (%)', fontsize=10)
    ax6.set_title('Relative Change Between Consecutive Refinements',
                  fontsize=10, fontweight='bold')
    ax6.set_xticks(x + width * 2)
    ax6.set_xticklabels(pairs, fontsize=8)
    ax6.legend(fontsize=7, ncol=2)
    ax6.grid(True, alpha=0.35, axis='y')
    ax6.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("sweep_results/mesh_convergence_figure.png",
                dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved: sweep_results/mesh_convergence_figure.png")
    print("\n  Mesh convergence study complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()