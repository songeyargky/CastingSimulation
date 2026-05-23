# src/defects.py
# ─────────────────────────────────────────────────────────────────────────────
# DEFECT PREDICTION MODULE  —  Four physics-based defects
#
# Defects included (confirmed working, responding to process parameters):
#   1. Misrun          — pouring temperature + heat extraction speed
#   2. Cold Shut       — superheat-to-undercooling ratio
#   3. Surface Cracking — post-solidification cooling rate at surface
#   4. Warpage          — post-solidification thermal gradient × CTE


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import matplotlib.patches as mpatches # pyright: ignore[reportMissingModuleSource]
import config
import src.materials as mat

# ============================================================================
# 1. MISRUN
# ============================================================================

def compute_misrun_risk(T_history, fs_history, time_points, T_pour, h_initial):
    """
    Misrun: metal solidifies before mold is fully filled.
    Drivers: T_pour (superheat) and h_initial (surface heat extraction speed).

    MRI = 0.60 × superheat_risk + 0.40 × cooling_risk
      superheat_risk = 1 − clip(superheat/200, 0, 1)
      cooling_risk   = 1 − clip(t_surf_liq/60, 0, 1)

    Responds to:
      T_pour ↑  → superheat ↑ → MRI ↓ (safer)
      h ↑       → surface solidifies faster → MRI ↑ (more dangerous)
      T_mold ↑  → slower surface solidification → MRI ↓ (marginal effect)
    """
    T_liq     = config.T_liquidus
    superheat = T_pour - T_liq

    t_surf_liq = None
    for i, T_arr in enumerate(T_history):
        if T_arr[-1] < T_liq:
            t_surf_liq = time_points[i]
            break
    if t_surf_liq is None:
        t_surf_liq = time_points[-1]

    SH_norm        = float(np.clip(superheat / 200.0, 0.0, 1.0))
    superheat_risk = 1.0 - SH_norm
    t_norm         = float(np.clip(t_surf_liq / 60.0, 0.0, 1.0))
    cooling_risk   = 1.0 - t_norm

    MRI = 0.60 * superheat_risk + 0.40 * cooling_risk

    if MRI < 0.25:    desc = "Low misrun risk"
    elif MRI < 0.50:  desc = "Moderate misrun risk"
    elif MRI < 0.75:  desc = "High misrun risk"
    else:             desc = "Very high misrun risk — increase T_pour or reduce h"

    return MRI, float(superheat), float(t_surf_liq), desc


# ============================================================================
# 2. COLD SHUT
# ============================================================================

def compute_cold_shut_risk(T_history, time_points, T_pour, T_mold):
    """
    Cold shut: converging metal streams meet below T_liquidus and fail to fuse.
    Drivers: T_pour (superheat) and T_mold (undercooling).

    CSR = superheat / undercooling
        = (T_pour − T_liq) / (T_liq − T_mold)
    Low CSR → streams arrive below T_liquidus → cold shut.

    CSRI = 0.60 × CSR_risk + 0.40 × surface_fraction
      CSR_risk      = clip(1 − CSR/0.5, 0, 1)
      surface_fraction = fraction of liquidus-to-mold range the surface
                         has already cooled through by t=10s

    Responds to:
      T_pour ↑  → CSR ↑ → CSRI ↓ (safer)
      T_mold ↑  → undercooling ↓ → CSR ↑ → CSRI ↓ (safer)
    """
    T_liq        = config.T_liquidus
    superheat    = max(T_pour  - T_liq, 0.0)
    undercooling = max(T_liq   - T_mold, 1.0)
    CSR          = superheat / undercooling

    T_surf_early = None
    for i, t in enumerate(time_points):
        if t >= 10.0:
            T_surf_early = T_history[i][-1]
            break
    if T_surf_early is None:
        T_surf_early = T_history[-1][-1]

    CSR_risk  = float(np.clip(1.0 - CSR / 0.5, 0.0, 1.0))
    surf_frac = float(np.clip((T_liq - T_surf_early) / undercooling, 0.0, 1.0))
    CSRI      = 0.60 * CSR_risk + 0.40 * surf_frac

    if CSRI < 0.25:    desc = "Low cold-shut risk"
    elif CSRI < 0.50:  desc = "Moderate cold-shut risk"
    elif CSRI < 0.75:  desc = "High cold-shut risk"
    else:              desc = "Very high cold-shut risk — increase T_pour and/or T_mold"

    return CSRI, float(CSR), float(T_surf_early), desc


# ============================================================================
# 3. SURFACE CRACKING
# ============================================================================

def compute_surface_crack_index(T_history, fs_history, time_points):
    """
    Surface cracking: thermal shock after surface solidification.
    Post-solidification cooling rate at the surface node drives crack initiation.

    Thresholds calibrated for permanent mold:
      CR < 20 K/s  → Low    (sand-casting threshold is 5 K/s — much slower)
      CR 20–50 K/s → Moderate
      CR > 50 K/s  → High

    Responds to:
      h_initial ↑  → faster post-solid surface cooling → SCI ↑
      T_mold ↓     → colder mold → faster extraction → SCI ↑
      diameter ↑   → larger thermal mass → slower post-solid CR → SCI ↓
    """
    nt   = len(T_history)
    surf = len(T_history[0]) - 1

    t_surf_sol   = None
    t_surf_sol_i = None
    for i, fs_arr in enumerate(fs_history):
        if np.array(fs_arr)[surf] >= 0.99:
            t_surf_sol   = time_points[i]
            t_surf_sol_i = i
            break

    if t_surf_sol is None:
        return 0.0, None, 0.0, "Surface never fully solidified in sim window"

    max_cr = 0.0
    for i in range(t_surf_sol_i, nt - 1):
        dt_i = time_points[i + 1] - time_points[i]
        if dt_i > 0:
            cr = abs(T_history[i][surf] - T_history[i + 1][surf]) / dt_i
            if cr > max_cr:
                max_cr = cr

    CR_LOW  = getattr(config, 'SCI_cr_low',  20.0)
    CR_HIGH = getattr(config, 'SCI_cr_high', 50.0)

    if max_cr < CR_LOW:
        SCI  = max_cr / CR_LOW * 0.33
        desc = "Low surface crack risk"
    elif max_cr < CR_HIGH:
        SCI  = 0.33 + (max_cr - CR_LOW) / (CR_HIGH - CR_LOW) * 0.34
        desc = "Moderate surface crack risk"
    else:
        SCI  = min(0.67 + (max_cr - CR_HIGH) / CR_HIGH * 0.33, 1.0)
        desc = "High surface crack risk"

    return float(SCI), float(t_surf_sol), float(max_cr), desc


# ============================================================================
# 4. WARPAGE
# ============================================================================

def compute_warpage_index(T_history, fs_history, time_points, r):
    """
    Warpage: differential thermal contraction after full solidification.

    WI = alpha_CTE × max(T_centre − T_surface)_post_solid

    Dimensional: out-of-tolerance estimate = WI × diameter_mm (mm)

    Thresholds (grinding media, ISO 3290-inspired):
      WI < 0.004  → Low      (OOT < 0.024 mm on 60mm ball)
      WI < 0.007  → Moderate
      WI < 0.010  → High
      WI >= 0.010 → Very High

    Responds to:
      T_mold ↓     → larger post-solid gradient → WI ↑
      diameter ↑   → larger absolute ΔT post-solid → WI ↑
      T_pour ↑     → more stored heat → larger gradient persists longer → WI ↑
    """
    alpha = config.alpha_cte

    t_full_solid   = None
    t_full_solid_i = None
    for i, fs_arr in enumerate(fs_history):
        if np.all(np.array(fs_arr) >= 0.99):
            t_full_solid   = time_points[i]
            t_full_solid_i = i
            break

    if t_full_solid_i is None:
        return 0.0, None, 0.0, "Casting not fully solidified in simulation window"

    delta_T_post = 0.0
    for i in range(t_full_solid_i, len(T_history)):
        dT = T_history[i][0] - T_history[i][-1]
        if dT > delta_T_post:
            delta_T_post = dT

    WI      = alpha * delta_T_post
    d_mm    = r[-1] * 2000.0
    OOT_mm  = WI * d_mm

    if WI < 0.004:
        desc = "Low warpage risk"
    elif WI < 0.007:
        desc = "Moderate warpage risk"
    elif WI < 0.010:
        desc = "High warpage risk"
    else:
        desc = f"Very high warpage — estimated OOT {OOT_mm:.3f} mm on {d_mm:.0f} mm ball"

    return float(WI), float(t_full_solid), float(delta_T_post), desc


# ============================================================================
# CHVORINOV'S RULE
# ============================================================================

def compute_chvorinov(diameter_mm, t_simulated):
    """
    Chvorinov's Rule:  t_s = C × (V/A)²

    For a sphere:  V/A = D/6
    With C = 1.488 (from config):  t_s = 1.488 × (D/6)²

    Compares predicted solidification time to the simulation result.
    Flags deviation > Chvorinov_tol (default 15%).

    Returns
    ───────
    t_chvorinov : float — predicted solidification time (s)
    deviation   : float — fractional deviation (t_sim − t_chv) / t_chv
    flag        : bool  — True if |deviation| > tolerance
    """
    C           = config.Chvorinov_C
    V_over_A    = diameter_mm / 6.0          # mm
    t_chvorinov = C * (V_over_A ** 2)        # seconds

    if t_simulated is not None and t_simulated > 0:
        deviation = (t_simulated - t_chvorinov) / t_chvorinov
        flag      = abs(deviation) > config.Chvorinov_tol
    else:
        deviation = float('nan')
        flag      = False

    return float(t_chvorinov), float(deviation) if not np.isnan(deviation) else None, flag


# ============================================================================
# HELPERS AND CLASSIFICATION
# ============================================================================

def _risk_colour(v):
    if v < 0.25: return '#2ecc71'
    if v < 0.50: return '#f1c40f'
    if v < 0.75: return '#e67e22'
    return '#e74c3c'


def _risk_label(v):
    if v < 0.25: return "Low"
    if v < 0.50: return "Moderate"
    if v < 0.75: return "High"
    return "Very High"


# ============================================================================
# DEFECT DASHBOARD  (4 defects + Chvorinov)
# ============================================================================

def plot_defect_dashboard(scores, diameter_mm, label="",
                          save_path=None, show=False):
    """
    Clean four-defect dashboard with Chvorinov validation panel.
    """
    MRI      = scores.get('MRI', 0.0)
    CSRI     = scores.get('CSRI', 0.0)
    SCI      = scores.get('SCI', 0.0)
    WI_raw   = scores.get('WI', 0.0)
    WI_risk  = float(np.clip(WI_raw / 0.010, 0.0, 1.0))

    defects = [
        ("Misrun",          MRI,     "T_pour, h"),
        ("Cold Shut",       CSRI,    "T_pour, T_mold"),
        ("Surface Cracking",SCI,     "post-solid Ṙ"),
        ("Warpage",         WI_risk, "ΔT_post-solid, CTE"),
    ]

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#f8f9fa')
    title = f'Defect Risk Dashboard  —  ∅{diameter_mm} mm Ball'
    if label:
        title += f'  |  {label}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)

    # ── Risk bar chart ────────────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    risks  = [d[1] for d in defects]
    y_pos  = np.arange(len(defects))
    bars   = ax_bar.barh(y_pos, risks,
                         color=[_risk_colour(r) for r in risks],
                         edgecolor='white', linewidth=0.8, height=0.55)

    for bar, risk, drv in zip(bars, risks, [d[2] for d in defects]):
        ax_bar.text(risk + 0.015, bar.get_y() + bar.get_height() / 2,
                    f'{_risk_label(risk)}  ({risk:.3f})',
                    va='center', fontsize=10)
        ax_bar.text(-0.015, bar.get_y() + bar.get_height() / 2,
                    f'[{drv}]', va='center', ha='right',
                    fontsize=8.5, color='#555')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([d[0] for d in defects], fontsize=11)
    ax_bar.set_xlim(0, 1.55)
    ax_bar.set_xlabel('Risk Index  (0 = safe  →  1 = very high)', fontsize=11)
    ax_bar.set_title('Casting Defect Risk Assessment', fontsize=11, fontweight='bold')
    for xv, col in [(0.25,'#2ecc71'),(0.50,'#f1c40f'),(0.75,'#e67e22')]:
        ax_bar.axvline(xv, color=col, ls=':', lw=1.3)
    ax_bar.set_facecolor('#fdfdfd')
    ax_bar.grid(axis='x', alpha=0.3)
    patches = [mpatches.Patch(color=c, label=l) for c, l in [
        ('#2ecc71', 'Low (< 0.25)'),
        ('#f1c40f', 'Moderate (0.25–0.50)'),
        ('#e67e22', 'High (0.50–0.75)'),
        ('#e74c3c', 'Very High (> 0.75)'),
    ]]
    ax_bar.legend(handles=patches, fontsize=9, loc='lower right')

    # ── Numeric summary panel ─────────────────────────────────────────────────
    ax_txt = fig.add_axes([0.63, 0.55, 0.35, 0.38])
    ax_txt.axis('off')
    t_chv    = scores.get('t_chvorinov', 0.0)
    t_sim    = scores.get('t_full_solid', None)
    dev      = scores.get('chv_deviation', None)
    chv_flag = scores.get('chv_flag', False)
    chv_str  = f"{dev*100:+.1f}%  {'⚠ FLAGGED' if chv_flag else '✓ OK'}" if dev is not None else "N/A"

    lines = [
        f"PROCESS CONDITIONS",
        f"  T_pour         : {scores.get('T_pour', '?'):.0f} °C",
        f"  T_mold         : {scores.get('T_mold', '?'):.0f} °C",
        f"  T_ambient      : {config.T_ambient:.0f} °C",
        f"  h_initial      : {scores.get('h_initial', '?'):.0f} W/m²K",
        f"  h_ambient corr : +{scores.get('h_ambient_corr', 0):.1f} W/m²K",
        f"  Superheat      : {scores.get('superheat', '?'):.0f} K",
        "",
        f"SOLIDIFICATION TIMELINE",
        f"  t_surf solidif : {scores.get('t_surf_sol', 0):.0f} s",
        f"  t_all solid    : {t_sim:.0f} s" if t_sim else "  t_all solid    : not reached",
        "",
        f"CHVORINOV VALIDATION",
        f"  t_Chvorinov    : {t_chv:.0f} s",
        f"  t_simulated    : {t_sim:.0f} s" if t_sim else "  t_simulated    : N/A",
        f"  Deviation      : {chv_str}",
        f"  Tolerance      : ±{config.Chvorinov_tol*100:.0f}%",
        "",
        f"DEFECT SUMMARY",
        f"  Misrun         : {MRI:.3f}  [{_risk_label(MRI)}]",
        f"  Cold Shut      : {CSRI:.3f}  [{_risk_label(CSRI)}]",
        f"  Surface Crack  : {SCI:.3f}  [{_risk_label(SCI)}]",
        f"  Warpage OOT    : {WI_raw*diameter_mm:.3f} mm  [{_risk_label(WI_risk)}]",
    ]
    ax_txt.text(0.02, 0.97, '\n'.join(lines),
                transform=ax_txt.transAxes,
                fontsize=8.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f4f8',
                          alpha=0.95, edgecolor='#aaa'))

    # ── Cooling curves (bottom panel) ─────────────────────────────────────────
    ax_cool = fig.add_axes([0.05, 0.07, 0.55, 0.40])
    time_pts = scores.get('time_points', [])
    ctr_T    = scores.get('center_temps', [])
    srf_T    = scores.get('surface_temps', [])

    if time_pts and ctr_T:
        ax_cool.plot(time_pts, ctr_T,  lw=2, color='#e74c3c', label='Centre')
        ax_cool.plot(time_pts, srf_T,  lw=2, color='#3498db', label='Surface')
        ax_cool.axhline(config.T_liquidus, color='grey', ls=':',  lw=1,
                        label=f'T_liq={config.T_liquidus:.0f}°C')
        ax_cool.axhline(config.T_solidus,  color='grey', ls='--', lw=1,
                        label=f'T_sol={config.T_solidus:.0f}°C')
        if t_sim:
            ax_cool.axvline(t_sim, color='purple', ls=':', lw=1.3,
                            label=f't_solid={t_sim:.0f}s')
        ax_cool.set_xlabel('Time (s)', fontsize=10)
        ax_cool.set_ylabel('Temperature (°C)', fontsize=10)
        ax_cool.set_title('Cooling Curves — Centre and Surface', fontsize=10)
        ax_cool.legend(fontsize=8.5)
        ax_cool.grid(True, alpha=0.35)

    # ── Chvorinov comparison (bottom right) ───────────────────────────────────
    ax_chv = fig.add_axes([0.63, 0.07, 0.35, 0.40])
    ax_chv.axis('off')

    chv_colour = '#e74c3c' if chv_flag else '#2ecc71'
    chv_text   = (
        f"Chvorinov Validation\n\n"
        f"  Rule:   t_s = C × (V/A)²\n"
        f"          = {config.Chvorinov_C} × ({diameter_mm}/6)²\n"
        f"          = {t_chv:.1f} s\n\n"
        f"  Simulation: {t_sim:.1f} s\n\n" if t_sim else
        f"  Simulation: not reached\n\n"
    )
    if dev is not None:
        chv_text += f"  Deviation: {dev*100:+.1f}%\n"
        chv_text += f"  Tolerance: ±{config.Chvorinov_tol*100:.0f}%\n\n"
        chv_text += "  ✓ WITHIN TOLERANCE" if not chv_flag else "  ⚠ EXCEEDS TOLERANCE"

    ax_chv.text(0.10, 0.85, chv_text,
                transform=ax_chv.transAxes,
                fontsize=9.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round',
                          facecolor='#d5f5e3' if not chv_flag else '#fde8e8',
                          alpha=0.95, edgecolor=chv_colour, linewidth=2))

    plt.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.03)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved dashboard -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ============================================================================
# MULTI-DIAMETER COOLING CURVE COMPARISON
# ============================================================================

def plot_cooling_curves_comparison(all_cooling_data, save_path=None, show=False):
    """
    Plot centre and surface temperature vs time for all diameters
    on a single figure (two panels: centre / surface).
    """
    colours = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    fig.suptitle('Cooling Curves — All Ball Diameters', fontsize=13, fontweight='bold')

    for idx, data in enumerate(all_cooling_data):
        c   = colours[idx % len(colours)]
        lbl = f"∅{data['diameter_mm']} mm"
        ax1.plot(data['time_points'], data['center_temps'],  lw=2, color=c, label=lbl)
        ax2.plot(data['time_points'], data['surface_temps'], lw=2, color=c, label=lbl)

    for ax, title in [(ax1, 'Centre Temperature'), (ax2, 'Surface Temperature')]:
        ax.axhline(config.T_liquidus, color='grey', ls=':',  lw=1,
                   label=f'T_liq={config.T_liquidus:.0f}°C')
        ax.axhline(config.T_solidus,  color='grey', ls='--', lw=1,
                   label=f'T_sol={config.T_solidus:.0f}°C')
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)

    ax2.set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved cooling comparison -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ============================================================================
# CHVORINOV VALIDATION SUMMARY PLOT
# ============================================================================

def plot_chvorinov_validation(chvorinov_data, save_path=None, show=False):
    """
    Bar chart comparing Chvorinov predicted vs simulated solidification time
    for all scenarios, with flagged deviations highlighted.

    chvorinov_data: list of dicts with keys:
        label, diameter_mm, t_chvorinov, t_simulated, deviation, flag
    """
    if not chvorinov_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Chvorinov's Rule Validation", fontsize=13, fontweight='bold')

    labels   = [d['label'] for d in chvorinov_data]
    t_chv    = [d['t_chvorinov'] for d in chvorinov_data]
    t_sim    = [d['t_simulated'] if d['t_simulated'] else 0 for d in chvorinov_data]
    devs     = [d['deviation']*100 if d['deviation'] is not None else 0 for d in chvorinov_data]
    flags    = [d['flag'] for d in chvorinov_data]

    x = np.arange(len(labels)); w = 0.38
    ax1.bar(x - w/2, t_chv, w, label="Chvorinov", color='#3498db', alpha=0.8)
    ax1.bar(x + w/2, t_sim, w, label="Simulated",
            color=['#e74c3c' if f else '#2ecc71' for f in flags], alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Solidification Time (s)')
    ax1.set_title('Chvorinov vs Simulated Solidification Time')
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)

    bar_colours = ['#e74c3c' if f else '#2ecc71' for f in flags]
    bars = ax2.bar(x, devs, color=bar_colours, edgecolor='white', linewidth=0.5)
    ax2.axhline( config.Chvorinov_tol * 100, color='orange', ls='--', lw=1.5,
                label=f'+{config.Chvorinov_tol*100:.0f}% tolerance')
    ax2.axhline(-config.Chvorinov_tol * 100, color='orange', ls='--', lw=1.5,
                label=f'-{config.Chvorinov_tol*100:.0f}% tolerance')
    ax2.axhline(0, color='grey', ls='-', lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Deviation from Chvorinov (%)')
    ax2.set_title("Deviation  (flagged if |dev| > 15%)")
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

    flagged_count = sum(flags)
    ax2.text(0.99, 0.97, f'Flagged: {flagged_count}/{len(flags)}',
             transform=ax2.transAxes, fontsize=10, ha='right', va='top',
             color='#e74c3c' if flagged_count > 0 else '#2ecc71',
             fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved Chvorinov validation -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig