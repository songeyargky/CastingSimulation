import streamlit as st
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Mill Ball Casting Simulation",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Import simulation modules ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

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

# ── Helpers ───────────────────────────────────────────────────────
def risk_colour(v):
    if v < 0.25: return '#2ecc71'
    if v < 0.50: return '#f1c40f'
    if v < 0.75: return '#e67e22'
    return '#e74c3c'

def risk_label(v):
    if v < 0.25: return 'LOW'
    if v < 0.50: return 'MODERATE'
    if v < 0.75: return 'HIGH'
    return 'VERY HIGH'

# ── Sidebar — inputs ──────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/ios-filled/100/1a3a5c/sphere.png",
    width=60
)
st.sidebar.title("Process Parameters")
st.sidebar.markdown("---")

st.sidebar.subheader("Temperatures")
T_pour = st.sidebar.slider(
    "Pouring Temperature (°C)",
    min_value=1250, max_value=1550,
    value=1450, step=10,
    help="Temperature of molten metal at point of pouring"
)
T_mold = st.sidebar.slider(
    "Mould Temperature (°C)",
    min_value=25, max_value=450,
    value=300, step=25,
    help="Preheat temperature of the permanent metal mould"
)

st.sidebar.subheader("Geometry")
diameter = st.sidebar.selectbox(
    "Ball Diameter (mm)",
    options=[60, 80, 100, 120],
    index=2
)

st.sidebar.subheader("Heat Transfer")
h_initial = st.sidebar.slider(
    "h_initial — metal contact (W/m²K)",
    min_value=200, max_value=2000,
    value=1200, step=100,
    help="HTC while casting is in contact with mould wall"
)
h_gap = st.sidebar.slider(
    "h_gap — air gap (W/m²K)",
    min_value=50, max_value=800,
    value=300, step=50,
    help="HTC after air gap forms on shrinkage (fs ≥ 0.70)"
)

st.sidebar.markdown("---")
run = st.sidebar.button(
    "▶  Run Simulation",
    use_container_width=True,
    type="primary"
)

# ── Presets ───────────────────────────────────────────────────────
st.sidebar.subheader("Quick Presets")
col1, col2 = st.sidebar.columns(2)
sweet_spot = col1.button("★ Sweet Spot", use_container_width=True)
high_risk  = col2.button("⚠ High Risk",  use_container_width=True)

if sweet_spot:
    st.query_params["preset"] = "sweet"
    st.rerun()
if high_risk:
    st.query_params["preset"] = "risk"
    st.rerun()

preset = st.query_params.get("preset", None)
if preset == "sweet":
    T_pour, T_mold, diameter = 1550, 450, 100
elif preset == "risk":
    T_pour, T_mold, diameter = 1300, 25, 100

# ── Main page header ──────────────────────────────────────────────
st.title("⬡  Low-Chrome Mill Ball Solidification Simulation")
st.markdown(
    "**MS** &nbsp;|&nbsp; 1D Radial FDM &nbsp;|&nbsp; "
    "Backward-Euler Implicit Solver &nbsp;|&nbsp; N = 40 Nodes &nbsp;|&nbsp; "
    "Permanent Metal Mould &nbsp;|&nbsp; Low-Cr White Cast Iron"
)
st.markdown("---")

# ── Parameter summary strip ───────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("T_pour", f"{T_pour} °C", delta=f"{T_pour - 1250} K superheat")
c2.metric("T_mold",  f"{T_mold} °C")
c3.metric("Diameter", f"{diameter} mm")
c4.metric("h_initial", f"{h_initial} W/m²K")
c5.metric("h_gap",  f"{h_gap} W/m²K")

st.markdown("---")

# ── Run simulation ────────────────────────────────────────────────
if run or preset:

    with st.spinner(f"Computing solidification for {diameter} mm ball..."):

        # ── Solver ───────────────────────────────────────────────
        N = config.N_nodes
        r, dr, dt = create_grid(diameter, N)
        T = np.ones(N) * T_pour
        time_val  = 0.0
        next_save = 1.0
        SAVE_INTERVAL = 1.0

        time_pts  = []
        T_centre  = []
        T_surface = []
        T_history = []
        fs_history= []
        t_sol     = None

        while time_val < config.max_sim_time:
            T, surface_fs = update_temperature(
                T, r, dr, dt, T_mold,
                h_initial=h_initial,
                h_gap=h_gap,
                time=time_val
            )
            time_val += dt

            if time_val >= next_save:
                time_pts.append(time_val)
                T_centre.append(T[0])
                T_surface.append(T[-1])
                T_history.append(T.copy())
                fs_history.append(mat.get_solid_fraction_profile(T))
                next_save += SAVE_INTERVAL

                if t_sol is None and T[0] <= config.T_solidus:
                    t_sol = time_val

            if np.all(T - T_mold < 5.0):
                break

            if np.any(np.isnan(T)):
                st.error("Numerical instability detected. Try reducing h_initial.")
                st.stop()

        if t_sol is None:
            t_sol = time_val

        # ── Defect models ─────────────────────────────────────────
        MRI,  sh,    t_sl, _ = compute_misrun_risk(
            T_history, fs_history, time_pts, T_pour, h_initial)
        CSRI, csr,   t_s,  _ = compute_cold_shut_risk(
            T_history, time_pts, T_pour, T_mold)
        SCI,  t_ss,  cr,   _ = compute_surface_crack_index(
            T_history, fs_history, time_pts)
        WI,   t_fs,  dT,   _ = compute_warpage_index(
            T_history, fs_history, time_pts, r)
        WI_risk = float(np.clip(WI / 0.010, 0, 1))
        total   = MRI + CSRI + SCI + WI_risk

        # ── Thermal profiles ──────────────────────────────────────
        i_sol_end = len(T_history) - 1
        for i, fs_arr in enumerate(fs_history):
            if fs_arr[0] >= 0.99:
                i_sol_end = i
                break

        R_profile = np.zeros(N)
        n_used = 0
        for i in range(1, min(i_sol_end, len(T_history) - 2)):
            R_step = np.abs(
                np.array(T_history[i+1]) - np.array(T_history[i-1])
            ) / (2.0 * SAVE_INTERVAL)
            R_profile += R_step
            n_used += 1
        if n_used > 0:
            R_profile /= n_used

        G_profile = np.zeros(N)
        for i in range(i_sol_end + 1):
            G_step = np.abs(np.gradient(T_history[i], dr))
            G_profile = np.maximum(G_profile, G_step)

        r_mm = r * 1000.0

        # ── Chvorinov ─────────────────────────────────────────────
        t_chv = config.Chvorinov_C * (diameter / 6.0) ** 2
        dev   = (t_sol - t_chv) / t_chv * 100.0

    # ── Results layout ────────────────────────────────────────────
    st.success(f"Simulation complete — solidification time: **{t_sol:.0f} s**")

    # Row 1 — key metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Solidification Time", f"{t_sol:.0f} s")
    m2.metric("Chvorinov Prediction", f"{t_chv:.1f} s",
              delta=f"{dev:+.1f}% deviation",
              delta_color="inverse" if abs(dev) > 15 else "normal")
    m3.metric("Superheat",     f"{sh:.0f} K")
    m4.metric("Post-solid CR", f"{cr:.2f} K/s")
    m5.metric("Total Risk",    f"{total:.3f} / 4.0")

    st.markdown("---")

    # Row 2 — cooling curve | defect bars
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Cooling Curve")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(time_pts, T_centre,  lw=2.0, color='#1f77b4', label='T_centre')
        ax1.plot(time_pts, T_surface, lw=2.0, color='#ff7f0e', label='T_surface')
        ax1.axhline(config.T_liquidus, color='steelblue', ls=':', lw=1.2,
                    label=f'T_liq = {config.T_liquidus}°C')
        ax1.axhline(config.T_solidus,  color='steelblue', ls='--', lw=1.2,
                    label=f'T_sol = {config.T_solidus}°C')
        ax1.axhline(T_mold, color='grey', ls='-.', lw=1.0,
                    label=f'T_mold = {T_mold}°C')
        ax1.axvline(t_sol, color='purple', ls=':', lw=1.5,
                    label=f't_sol = {t_sol:.0f} s')
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Temperature (°C)')
        ax1.set_title(f'Cooling Curve — {diameter} mm Ball | '
                      f'T_pour={T_pour}°C  T_mold={T_mold}°C')
        ax1.legend(fontsize=8); ax1.grid(alpha=0.35)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col_right:
        st.subheader("Defect Risk Summary")
        defect_names = ['Misrun', 'Cold Shut', 'Surf. Crack', 'Warpage']
        defect_vals  = [MRI, CSRI, SCI, WI_risk]
        colours = [risk_colour(v) for v in defect_vals]

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.barh(defect_names, defect_vals,
                        color=colours, edgecolor='#333', linewidth=0.8)
        for bar, val in zip(bars, defect_vals):
            label = f'{val:.3f}  {risk_label(val)}'
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     label, va='center', fontsize=9)
        ax2.axvline(0.25, color='green',  ls='--', lw=0.8, alpha=0.7)
        ax2.axvline(0.50, color='orange', ls='--', lw=0.8, alpha=0.7)
        ax2.axvline(0.75, color='red',    ls='--', lw=0.8, alpha=0.7)
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel('Risk Index  (0 = safe → 1 = very high)')
        ax2.set_title(f'D={diameter}mm | T_pour={T_pour}°C | T_mold={T_mold}°C')
        ax2.grid(alpha=0.3, axis='x')
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # Risk table
        st.markdown("**Defect risk detail**")
        for name, val in zip(defect_names, defect_vals):
            col_a, col_b = st.columns([2, 1])
            col_a.write(name)
            col_b.write(f"**{val:.3f}** — {risk_label(val)}")

    st.markdown("---")

    # Row 3 — thermal profiles
    st.subheader("Thermal Field Profiles")
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 4))

    ax3.plot(r_mm, R_profile, color='#e74c3c', lw=2.0,
             marker='o', markersize=3)
    ax3.fill_between(r_mm, R_profile, alpha=0.12, color='#e74c3c')
    ax3.set_xlabel('Radial Position (mm)'); ax3.set_ylabel('Cooling Rate R (K/s)')
    ax3.set_title('Cooling Rate Profile\n'
                  'Rate of temperature decrease at each node')
    ax3.set_xlim(left=0); ax3.set_ylim(bottom=0); ax3.grid(alpha=0.35)
    ax3.text(0.02, 0.97,
             f'Centre: {R_profile[0]:.3f} K/s\nSurface: {R_profile[-1]:.3f} K/s',
             transform=ax3.transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#fff3f3', alpha=0.85))

    G_threshold = 300.0
    ax4.plot(r_mm, G_profile, color='#2980b9', lw=2.0,
             marker='o', markersize=3)
    ax4.fill_between(r_mm, G_profile, alpha=0.12, color='#2980b9')
    ax4.axhline(G_threshold, color='#e67e22', ls='--', lw=1.2,
                label=f'Poor feeding threshold ({G_threshold:.0f} K/m)')
    ax4.set_xlabel('Radial Position (mm)')
    ax4.set_ylabel('Thermal Gradient G (K/m)')
    ax4.set_title('Thermal Gradient Profile\n'
                  'Temperature difference across casting')
    ax4.set_xlim(left=0); ax4.set_ylim(bottom=0)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.35)
    ax4.text(0.02, 0.97,
             f'Centre: {G_profile[0]:.0f} K/m\n'
             f'Surface: {G_profile[-1]:.0f} K/m\n'
             f'Peak: {np.max(G_profile):.0f} K/m',
             transform=ax4.transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#f0f6ff', alpha=0.85))

    plt.suptitle(
        f'Thermal Field Profiles — {diameter} mm Ball | '
        f'T_pour={T_pour}°C  T_mold={T_mold}°C',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("---")

    # Row 4 — export
    st.subheader("Export Results")
    import pandas as pd
    results_df = pd.DataFrame({
        'Parameter': ['T_pour','T_mold','Diameter','h_initial','h_gap',
                      'Solidification_time_s','Chvorinov_time_s',
                      'Deviation_pct','MRI','CSRI','SCI','WI_risk','Total_risk'],
        'Value':     [T_pour, T_mold, diameter, h_initial, h_gap,
                      round(t_sol,1), round(t_chv,1),
                      round(dev,1), round(MRI,4), round(CSRI,4),
                      round(SCI,4), round(WI_risk,4), round(total,4)]
    })
    st.dataframe(results_df, use_container_width=True)

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="⬇  Download Results CSV",
        data=csv,
        file_name=f"sim_D{diameter}_Tp{T_pour}_Tm{T_mold}.csv",
        mime='text/csv'
    )

else:
    # ── Landing state ─────────────────────────────────────────────
    st.info(
        "👈  Set your process parameters in the sidebar and click "
        "**▶ Run Simulation** to begin."
    )
    st.markdown("""
    ### What this simulation does

    | Feature | Detail |
    |---|---|
    | **Solver** | Semi-implicit backward-Euler 1D radial FDM |
    | **Spatial nodes** | N = 40 radial nodes |
    | **Alloy** | Low-chrome white cast iron (~3 wt% C, 1–3 wt% Cr) |
    | **Ball diameters** | 60, 80, 100, 120 mm |
    | **Latent heat model** | Two-stage: primary dendrite + eutectic |
    | **IHTC model** | Two-zone: metal contact → air gap at fs = 0.70 |
    | **Defect models** | Misrun, Cold Shut, Surface Cracking, Warpage |
    | **Outputs** | Cooling curves, defect risk indices, thermal profiles |

    ### Risk index scale
    | Band | Index range | Meaning |
    |---|---|---|
    | 🟢 Low | 0.00 – 0.25 | Acceptable — no immediate action required |
    | 🟡 Moderate | 0.25 – 0.50 | Monitor — consider parameter adjustment |
    | 🟠 High | 0.50 – 0.75 | Likely defect in production — adjust before casting |
    | 🔴 Very High | 0.75 – 1.00 | Near-certain defect — do not run at these parameters |
    """)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Low-Chrome Mill Ball Solidification Simulation · Module L99 · "
    "1D Radial FDM · Semi-implicit Backward-Euler · N=40 nodes · "
    "4 defect models · Permanent metal mould"
)
