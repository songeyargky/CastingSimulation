import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
import src.materials as mat
from src.grid import create_grid
from src.solver import update_temperature

from src.defects import (
    compute_surface_crack_index,
    compute_misrun_risk,
    compute_cold_shut_risk,
    compute_warpage_index,
)

# ── Simulation settings ──────────────────────────────────────────────────────
DIAMETERS_MM  = config.diameters_mm
T_POUR        = config.base_case["T_pour"]
T_MOLD        = config.base_case["T_mold"]
MAX_SIM_TIME  = getattr(config, 'max_sim_time', 700.0)
SAVE_INTERVAL = 1.0   # seconds between saved snapshots
EQUIL_TOL     = 5.0   # stop when all nodes within this many degrees of T_mold

print("=" * 65)
print("=== Low-Chrome Mill Ball  --  Solidification Simulation ===")
print("=" * 65)
print(f"T_pour={T_POUR}C  T_mold={T_MOLD}C  "
      f"h_initial={config.h_initial}  h_gap={config.h_gap}")
print(f"Diameters : {DIAMETERS_MM} mm")
print()
print("Active defect models:")
print("  [1] Misrun")
print("  [2] Cold Shut")
print("  [3] Surface Cracking")
print("  [4] Warpage")
print()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_label(v):
    if v < 0.25: return 'Low'
    if v < 0.50: return 'Moderate'
    if v < 0.75: return 'High'
    return 'Very High'

def _risk_colour(v):
    if v < 0.25: return '#2ecc71'
    if v < 0.50: return '#f1c40f'
    if v < 0.75: return '#e67e22'
    return '#e74c3c'

all_results  = []
summary_rows = []

# ── Main loop over ball diameters ────────────────────────────────────────────
for diameter_mm in DIAMETERS_MM:

    print(f"\n{'='*58}")
    print(f"  Simulating  {diameter_mm} mm  ball ...")
    print(f"{'='*58}")

    N         = config.N_nodes
    r, dr, dt = create_grid(diameter_mm, N)
    print(f"  Grid: {N} nodes  dr={dr*1000:.3f}mm  dt={dt:.5f}s")

    # Uniform initial temperature equal to pour temperature
    T          = np.ones(N) * T_POUR
    surface_fs = 0.0
    time_val   = 0.0
    next_save  = SAVE_INTERVAL

    time_points        = []
    center_temps       = []
    surface_temps      = []
    surface_fs_history = []
    T_history          = []
    fs_history         = []
    equilibrated       = False

    # ── Time integration ──────────────────────────────────────────────────
    while time_val < MAX_SIM_TIME:
        T, surface_fs = update_temperature(
            T, r, dr, dt, T_MOLD,
            h_initial=config.h_initial,
            h_gap=config.h_gap,
            time=time_val
        )
        time_val += dt

        if time_val >= next_save:
            time_points.append(time_val)
            center_temps.append(T[0])
            surface_temps.append(T[-1])
            surface_fs_history.append(surface_fs)
            T_history.append(T.copy())
            fs_history.append(mat.get_solid_fraction_profile(T))
            next_save += SAVE_INTERVAL

            # Periodic console output: first 10s and every 30s thereafter
            if abs(time_val % 30) < dt * 1.5 or time_val < 11:
                fs_c = mat.get_solid_fraction(T[0])
                fs_m = mat.get_solid_fraction(T[N // 2])
                print(f"  t={time_val:6.1f}s | Ctr={T[0]:6.0f}C (fs={fs_c:.2f})"
                      f" | Mid={T[N//2]:6.0f}C (fs={fs_m:.2f})"
                      f" | Surf={T[-1]:6.0f}C (fs={surface_fs:.2f})")

            # Stop early once ball has equilibrated with the mold boundary condition
            if np.all(T - T_MOLD < EQUIL_TOL):
                print(f"  >> Equilibrium at t={time_val:.1f}s -- stopping.")
                equilibrated = True
                break

        if np.any(np.isnan(T)) or np.any(T < 0) or np.any(T > 3500):
            print(f"  !! Instability at t={time_val:.2f}s -- stopping.")
            break

    print(f"\n  Snapshots saved: {len(T_history)}  "
          f"({'equilibrated' if equilibrated else 'time-limit'})")

    if len(T_history) < 3:
        print("  WARNING: too few snapshots -- skipping defect calculation.")
        continue

    # ── Defect calculations ───────────────────────────────────────────────
    MRI,  superheat,     t_surf_liq,    mri_desc  = compute_misrun_risk(
        T_history, fs_history, time_points, T_POUR, config.h_initial
    )
    CSRI, CSR,           T_surf_early,  csr_desc  = compute_cold_shut_risk(
        T_history, time_points, T_POUR, T_MOLD
    )
    SCI,  t_surf_sol,    max_cool_rate, sci_desc  = compute_surface_crack_index(
        T_history, fs_history, time_points
    )
    WI,   t_full_solid,  dT_post_solid, warp_desc = compute_warpage_index(
        T_history, fs_history, time_points, r
    )

    WI_risk = float(np.clip(WI / 0.010, 0.0, 1.0))

    # ── Console output ────────────────────────────────────────────────────
    print(f"\n  {'='*52}")
    print(f"  DEFECT PREDICTION  --  {diameter_mm} mm ball")
    print(f"  {'='*52}")

    print(f"\n  [1] MISRUN")
    print(f"      Superheat              : {superheat:.1f} K")
    print(f"      t(surface < T_liq)     : {t_surf_liq:.1f} s")
    print(f"      Risk Index             : {MRI:.4f}  ->  {mri_desc}")

    print(f"\n  [2] COLD SHUT")
    print(f"      CSR (superheat/undercool): {CSR:.4f}")
    print(f"      T_surface at t=10s     : {T_surf_early:.1f} C")
    print(f"      Risk Index             : {CSRI:.4f}  ->  {csr_desc}")

    print(f"\n  [3] SURFACE CRACKING")
    if t_surf_sol:
        print(f"      t_surface solidified   : {t_surf_sol:.1f} s")
    else:
        print(f"      t_surface solidified   : not reached in window")
    print(f"      Max post-solid CR      : {max_cool_rate:.3f} K/s")
    print(f"      Risk Index             : {SCI:.4f}  ->  {sci_desc}")

    print(f"\n  [4] WARPAGE")
    if t_full_solid is not None:
        print(f"      t_all solid            : {t_full_solid:.1f} s")
    else:
        print(f"      t_all solid            : not reached in window")
    print(f"      Max post-solid DeltaT  : {dT_post_solid:.1f} K")
    print(f"      Warpage Index (CTE*DT) : {WI:.5f}  ->  {warp_desc}")
    print(f"      OOT estimate           : {WI * diameter_mm:.3f} mm")

    # ===== INSERTED THERMAL PROFILES SECTION =====
    # ── Compute cooling rate and thermal gradient spatial profiles ────────
    #
    # Cooling rate at each node: average |dT/dt| during solidification.
    # Computed using central differences on saved snapshots.
    # Higher values indicate faster local cooling.
    #
    # Thermal gradient at each node: maximum |dT/dr| seen during solidification.
    # High gradient = steep temperature slope = poor feeding conditions.

    dt_save = SAVE_INTERVAL

    # Find snapshot when the centre first becomes fully solid
    i_sol_end = len(T_history) - 1
    for i, fs_arr in enumerate(fs_history):
        if fs_arr[0] >= 0.99:
            i_sol_end = i
            break

    # Cooling rate: central-difference in time, averaged over solidification phase
    R_profile = np.zeros(N)
    n_used = 0
    for i in range(1, i_sol_end):
        R_at_step = np.abs(
            np.array(T_history[i + 1]) - np.array(T_history[i - 1])
        ) / (2.0 * dt_save)
        R_profile += R_at_step
        n_used    += 1
    if n_used > 0:
        R_profile /= n_used

    # Thermal gradient: maximum |dT/dr| at each node during solidification
    G_profile = np.zeros(N)
    for i in range(i_sol_end + 1):
        G_step = np.abs(np.gradient(T_history[i], dr))
        G_profile = np.maximum(G_profile, G_step)

    r_mm = r * 1000.0

    print(f"\n  THERMAL FIELD PROFILES (solidification phase)")
    print(f"      Cooling rate  -- centre: {R_profile[0]:.3f} K/s  "
          f"surface: {R_profile[-1]:.3f} K/s  peak: {np.max(R_profile):.3f} K/s")
    print(f"      Thermal grad  -- centre: {G_profile[0]:.0f} K/m  "
          f"surface: {G_profile[-1]:.0f} K/m  peak: {np.max(G_profile):.0f} K/m")

    # ── Thermal profiles figure (cooling rate + gradient) ─────────────────
    fig_tp, (ax_cr, ax_tg) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig_tp.suptitle(
        f"Thermal Field Profiles  --  {diameter_mm} mm Ball  |  "
        f"T_pour={T_POUR}C  T_mold={T_MOLD}C",
        fontsize=12, fontweight="bold"
    )

    ax_cr.plot(r_mm, R_profile, color="#e74c3c", lw=2.0, marker="o", markersize=3.5)
    ax_cr.fill_between(r_mm, R_profile, alpha=0.12, color="#e74c3c")
    ax_cr.set_xlabel("Radial Position (mm)", fontsize=11)
    ax_cr.set_ylabel("Cooling Rate  R  (K/s)", fontsize=11)
    ax_cr.set_title("Cooling Rate Profile\nRate of temperature decrease at each node",
                    fontsize=10)
    ax_cr.set_xlim(left=0); ax_cr.set_ylim(bottom=0)
    ax_cr.grid(True, alpha=0.35)
    idx_pk = int(np.argmax(R_profile))
    ax_cr.annotate(f"{R_profile[idx_pk]:.2f} K/s",
                   xy=(r_mm[idx_pk], R_profile[idx_pk]),
                   xytext=(r_mm[idx_pk] + r_mm[-1]*0.08, R_profile[idx_pk]*0.88),
                   fontsize=8, color="#c0392b",
                   arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    ax_cr.text(0.02, 0.97,
               f"Centre : {R_profile[0]:.3f} K/s\nSurface: {R_profile[-1]:.3f} K/s",
               transform=ax_cr.transAxes, fontsize=8.5, va="top",
               bbox=dict(boxstyle="round", facecolor="#fff3f3", alpha=0.88))

    G_threshold = 300.0   # K/m — nodes below this have poor feeding conditions
    ax_tg.plot(r_mm, G_profile, color="#2980b9", lw=2.0, marker="o", markersize=3.5)
    ax_tg.fill_between(r_mm, G_profile, alpha=0.12, color="#2980b9")
    ax_tg.axhline(G_threshold, color="#e67e22", ls="--", lw=1.2,
                  label=f"Poor feeding threshold ({G_threshold:.0f} K/m)")
    n_below = int(np.sum(G_profile < G_threshold))
    if n_below > 0:
        ax_tg.fill_between(r_mm, 0, np.minimum(G_profile, G_threshold),
                           where=(G_profile < G_threshold),
                           color="#e67e22", alpha=0.18,
                           label=f"{n_below} nodes below threshold")
    ax_tg.set_xlabel("Radial Position (mm)", fontsize=11)
    ax_tg.set_ylabel("Thermal Gradient  G  (K/m)", fontsize=11)
    ax_tg.set_title("Thermal Gradient Profile\nTemperature difference across casting",
                    fontsize=10)
    ax_tg.set_xlim(left=0); ax_tg.set_ylim(bottom=0)
    ax_tg.grid(True, alpha=0.35); ax_tg.legend(fontsize=8.5)
    ax_tg.text(0.02, 0.97,
               f"Centre : {G_profile[0]:.0f} K/m\nSurface: {G_profile[-1]:.0f} K/m\n"
               f"Peak   : {np.max(G_profile):.0f} K/m",
               transform=ax_tg.transAxes, fontsize=8.5, va="top",
               bbox=dict(boxstyle="round", facecolor="#f0f6ff", alpha=0.88))

    plt.tight_layout()
    plt.savefig(f"thermal_profiles_{diameter_mm}mm.png", dpi=150, bbox_inches="tight")
    plt.close(fig_tp)

    # ── Defect risk bar chart ─────────────────────────────────────────────
    defect_names   = ['Misrun', 'Cold Shut', 'Surface\nCracking', 'Warpage']
    defect_values  = [MRI, CSRI, SCI, WI_risk]
    defect_drivers = ['T_pour, h', 'T_pour, T_mold', 'Post-solid R', 'T_mold, diameter']

    fig_d, ax_d = plt.subplots(figsize=(10, 5))
    fig_d.patch.set_facecolor('#f8f9fa')
    y_pos   = np.arange(len(defect_names))
    colours = [_risk_colour(v) for v in defect_values]
    bars    = ax_d.barh(y_pos, defect_values, color=colours,
                        edgecolor='white', linewidth=0.8, height=0.55)

    for bar, val, drv in zip(bars, defect_values, defect_drivers):
        ax_d.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                  f'{_risk_label(val)}  ({val:.3f})', va='center', fontsize=10)
        ax_d.text(-0.012, bar.get_y() + bar.get_height() / 2,
                  f'[{drv}]', va='center', ha='right', fontsize=8, color='#555')

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(defect_names, fontsize=11)
    ax_d.set_xlim(0, 1.50)
    ax_d.set_xlabel('Risk Index  (0 = safe  ->  1 = very high)', fontsize=10)
    ax_d.set_title(
        f'Defect Risk Summary  --  {diameter_mm} mm Ball  |  '
        f'T_pour={T_POUR}C  T_mold={T_MOLD}C',
        fontsize=11, fontweight='bold'
    )
    for xv, col in [(0.25,'#2ecc71'),(0.50,'#f1c40f'),(0.75,'#e67e22')]:
        ax_d.axvline(xv, color=col, linestyle=':', linewidth=1.2)
    ax_d.set_facecolor('#fdfdfd')
    ax_d.grid(axis='x', alpha=0.3)
    patches = [mpatches.Patch(color=c, label=l) for c, l in [
        ('#2ecc71','Low (<0.25)'), ('#f1c40f','Moderate (0.25-0.50)'),
        ('#e67e22','High (0.50-0.75)'), ('#e74c3c','Very High (>0.75)')]]
    ax_d.legend(handles=patches, fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"defect_summary_{diameter_mm}mm.png", dpi=150, bbox_inches='tight')
    plt.close(fig_d)

    # ── Cooling curve ─────────────────────────────────────────────────────
    fig_cc, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    ax1.plot(time_points, center_temps,  label='Centre',  lw=2)
    ax1.plot(time_points, surface_temps, label='Surface', lw=2)
    ax1.axhline(config.T_liquidus, color='steelblue', ls=':', lw=1,
                label=f'T_liq={config.T_liquidus}C')
    ax1.axhline(config.T_solidus,  color='steelblue', ls='--', lw=1,
                label=f'T_sol={config.T_solidus}C')
    ax1.axhline(T_MOLD, color='grey', ls='-.', lw=1, label=f'T_mold={T_MOLD}C')
    if t_full_solid:
        ax1.axvline(t_full_solid, color='purple', ls=':', lw=1.2,
                    label=f't_solid={t_full_solid:.0f}s')
    ax1.set_ylabel('Temperature (C)')
    ax1.set_title(f'Cooling Curve  --  {diameter_mm} mm Ball')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.4)
    ax2.plot(time_points, surface_fs_history, lw=2, color='#2ca02c')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Surface solid fraction')
    ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"cooling_curve_{diameter_mm}mm.png", dpi=150, bbox_inches='tight')
    plt.close(fig_cc)

    print(f"\n  Saved: defect_summary_{diameter_mm}mm.png  |  "
          f"cooling_curve_{diameter_mm}mm.png")

    all_results.append({
        'diameter_mm' : diameter_mm,
        'MRI'         : MRI,
        'CSRI'        : CSRI,
        'SCI'         : SCI,
        'WI_risk'     : WI_risk,
        'WI'          : WI,
        't_full_solid': t_full_solid,
    })

    summary_rows.append({
        'd'       : diameter_mm,
        'MRI'     : MRI,
        'CSRI'    : CSRI,
        'SCI'     : SCI,
        'WI'      : WI,
        'WI_risk' : WI_risk,
        'OOT_mm'  : WI * diameter_mm,
        't_solid' : t_full_solid if t_full_solid else 0.0,
    })

# ── Multi-diameter comparison (2x2 grid, one panel per defect) ──────────────
if len(all_results) > 1:
    fig_all, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig_all.suptitle(
        f'Defect Comparison -- All Diameters  |  T_pour={T_POUR}C  T_mold={T_MOLD}C',
        fontsize=13, fontweight='bold'
    )
    defect_cfg = [
        (axes[0, 0], 'MRI',     'Misrun'),
        (axes[0, 1], 'CSRI',    'Cold Shut'),
        (axes[1, 0], 'SCI',     'Surface Cracking'),
        (axes[1, 1], 'WI_risk', 'Warpage'),
    ]
    diams = [str(r['diameter_mm']) for r in all_results]
    for ax, key, title in defect_cfg:
        vals     = [r[key] for r in all_results]
        bar_cols = [_risk_colour(v) for v in vals]
        bars     = ax.bar(diams, vals, color=bar_cols, edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for yv, col in [(0.25,'#2ecc71'),(0.50,'#f1c40f'),(0.75,'#e67e22')]:
            ax.axhline(yv, color=col, ls=':', lw=1.0)
        ax.set_xlabel('Diameter (mm)'); ax.set_ylabel('Risk Index')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#fdfdfd')
    plt.tight_layout()
    plt.savefig("defect_comparison_all_diameters.png", dpi=150, bbox_inches='tight')
    plt.close(fig_all)
    print(f"\n  Saved: defect_comparison_all_diameters.png")

# ── Final summary table ───────────────────────────────────────────────────────
if summary_rows:
    print("\n" + "=" * 85)
    print("  FINAL DEFECT SUMMARY")
    print("=" * 85)
    print(f"  {'Diam':>5} | {'Misrun':>8} {'Class':<10} | {'ColdShut':>8} {'Class':<10} | "
          f"{'SurfCrk':>7} {'Class':<10} | {'Warpage':>7} {'Class':<10} | {'OOT(mm)':>8}")
    print("  " + "-" * 82)
    for row in summary_rows:
        print(
            f"  {str(row['d'])+'mm':>5} | "
            f"{row['MRI']:>8.4f} {_risk_label(row['MRI']):<10} | "
            f"{row['CSRI']:>8.4f} {_risk_label(row['CSRI']):<10} | "
            f"{row['SCI']:>7.4f} {_risk_label(row['SCI']):<10} | "
            f"{row['WI_risk']:>7.4f} {_risk_label(row['WI_risk']):<10} | "
            f"{row['OOT_mm']:>8.3f}"
        )
    print()
    print(f"  Full solidification times:")
    for row in summary_rows:
        print(f"    {str(row['d'])+'mm':>5}: {row['t_solid']:.0f} s")
    print()