# analyse_sweep.py
# Run from the project root after executing run_sweep.py.
# Outputs are written to sweep_results/<subfolder>/.

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from itertools import combinations

warnings.filterwarnings("ignore")

# ─── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "#fafafa",
    "axes.facecolor"    : "#f7f7f7",
    "axes.edgecolor"    : "#333333",
    "axes.labelcolor"   : "#222222",
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "legend.fontsize"   : 9,
    "figure.titlesize"  : 14,
    "axes.titlepad"     : 10,
    "axes.labelpad"     : 6,
    "axes.grid"         : True,
    "grid.color"        : "#dddddd",
    "grid.linewidth"    : 0.6,
    "lines.linewidth"   : 2.0,
    "font.family"       : "DejaVu Sans",
})

DIAM_COLOURS = {60: "#1f77b4", 80: "#ff7f0e", 100: "#2ca02c", 120: "#d62728"}
RISK_CMAP    = "RdYlGn_r"    # red = high risk, green = low risk

# ─── Output directories ───────────────────────────────────────────────────────
for d in ["sweep_results", "sweep_results/contours", "sweep_results/radars",
          "sweep_results/gp_surfaces", "sweep_results/heatmaps",
          "sweep_results/tradeoffs", "sweep_results/sensitivity"]:
    os.makedirs(d, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
data_path = "sweep_results/parameter_sweep.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CSV not found at {data_path}. Run run_sweep.py first."
    )
df = pd.read_csv(data_path)

required = ["T_pour", "T_mold", "diameter_mm", "solidification_time_s",
            "MRI", "CSRI", "SCI", "WI_risk"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}'. Check sweep output.")

df["total_risk"] = df[["MRI", "CSRI", "SCI", "WI_risk"]].sum(axis=1)
diameters        = sorted(df["diameter_mm"].unique())

print(f"Loaded {len(df)} rows — diameters: {diameters}")
print(df[required + ['total_risk']].describe().round(3).to_string())
print()

DEFECTS       = ["MRI",     "CSRI",       "SCI",               "WI_risk"]
DEFECT_LABELS = ["Misrun",  "Cold Shut",  "Surface Cracking",  "Warpage"]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def risk_label(v):
    if v < 0.25: return "Low"
    if v < 0.50: return "Moderate"
    if v < 0.75: return "High"
    return "Very High"

def risk_colour(v):
    if v < 0.25: return "#2ecc71"
    if v < 0.50: return "#f1c40f"
    if v < 0.75: return "#e67e22"
    return "#e74c3c"

def _save(path, dpi=180):
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    print(f"  Saved: {os.path.relpath(path)}")

def _annotate_best(ax, df_sub, xcol, ycol, zcol, color="#1a1a2e", zorder=10):
    """Mark the lowest-risk point with a star and label."""
    idx = df_sub[zcol].idxmin()
    row = df_sub.loc[idx]
    ax.scatter(row[xcol], row[ycol], marker="*", s=280, color=color,
               zorder=zorder, edgecolors="white", linewidths=0.7,
               label=f"Optimal: {xcol}={row[xcol]:.0f}, {ycol}={row[ycol]:.0f}")


# ═════════════════════════════════════════════════════════════════════════════
# A.  PARETO TRADE-OFF SCATTER (all defect pairs, one file per pair)
# ═════════════════════════════════════════════════════════════════════════════
print("A. Pareto trade-off plots ...")
for (d1, l1), (d2, l2) in combinations(zip(DEFECTS, DEFECT_LABELS), 2):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for diam in diameters:
        sub = df[df["diameter_mm"] == diam]
        sc  = ax.scatter(sub[d1], sub[d2],
                         c=sub["T_pour"], cmap="plasma",
                         vmin=df["T_pour"].min(), vmax=df["T_pour"].max(),
                         s=60, alpha=0.75, edgecolors="white", linewidths=0.3,
                         label=f"{diam} mm")
    # add threshold lines
    for val, col, ls in [(0.25, "#2ecc71", "--"), (0.50, "#f1c40f", "-."),
                          (0.75, "#e67e22", ":")]:
        ax.axvline(val, color=col, ls=ls, lw=1.0, alpha=0.8)
        ax.axhline(val, color=col, ls=ls, lw=1.0, alpha=0.8)
    # shade the "both low" quadrant
    ax.axvspan(0, 0.25, ymin=0, ymax=0.25/ax.get_ylim()[1] if ax.get_ylim()[1] else 0.25,
               alpha=0.06, color="#2ecc71", zorder=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("T_pour (°C)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # legend by diameter (manual)
    handles = [mpatches.Patch(color=DIAM_COLOURS.get(d, "grey"),
                              label=f"{d} mm") for d in diameters]
    ax.legend(handles=handles, title="Diameter", title_fontsize=9,
              loc="upper right", framealpha=0.9)

    ax.set_xlabel(l1 + " Risk Index")
    ax.set_ylabel(l2 + " Risk Index")
    ax.set_title(f"Trade-off: {l1}  vs  {l2}\n"
                 "Dashed lines mark Low / Moderate / High thresholds")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    _save(f"sweep_results/tradeoffs/tradeoff_{d1}_vs_{d2}.png")


# ═════════════════════════════════════════════════════════════════════════════
# B.  CONTOUR PLOTS — all defects, all diameters
#     Each figure: 2×2 grid (one panel per defect), with:
#       - filled contour  (RdYlGn_r: green=safe, red=high risk)
#       - ISO lines at 0.25, 0.50, 0.75 with labels
#       - scatter of actual data points
#       - star marking the optimal (T_pour, T_mold) combo
# ═════════════════════════════════════════════════════════════════════════════
print("B. Contour plots ...")
for diam in diameters:
    sub = df[df["diameter_mm"] == diam].copy()
    if len(sub) < 5:
        print(f"   Skipping D={diam}mm (too few points)")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Defect Risk Contours  —  Diameter {diam} mm\n"
                 "Green = low risk  |  Red = high risk  |  ★ = optimal",
                 fontsize=14, y=1.01)
    axes_flat = axes.flatten()

    for idx, (defect, label) in enumerate(zip(DEFECTS, DEFECT_LABELS)):
        ax = axes_flat[idx]
        xi = np.linspace(sub["T_pour"].min(), sub["T_pour"].max(), 80)
        yi = np.linspace(sub["T_mold"].min(), sub["T_mold"].max(), 80)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((sub["T_pour"], sub["T_mold"]), sub[defect],
                      (XI, YI), method="cubic")
        ZI = np.clip(ZI, 0, 1)

        # filled contour
        cf = ax.contourf(XI, YI, ZI, levels=np.linspace(0, 1, 25),
                         cmap=RISK_CMAP, alpha=0.88)
        # ISO lines
        cs = ax.contour(XI, YI, ZI,
                        levels=[0.25, 0.50, 0.75],
                        colors=["#2ecc71", "#f1c40f", "#e74c3c"],
                        linewidths=[1.6, 1.6, 1.6])
        ax.clabel(cs, fmt={0.25: "Low", 0.50: "Mod", 0.75: "High"},
                  fontsize=9, inline=True)

        # scatter actual data
        ax.scatter(sub["T_pour"], sub["T_mold"],
                   c=sub[defect], cmap=RISK_CMAP, vmin=0, vmax=1,
                   s=40, edgecolors="#333", linewidths=0.4, zorder=5)

        # optimal star
        opt = sub.loc[sub[defect].idxmin()]
        ax.scatter(opt["T_pour"], opt["T_mold"], marker="*", s=320,
                   color="#1a1a2e", zorder=10, edgecolors="white", linewidths=0.8)
        ax.annotate(f"Best\n({opt['T_pour']:.0f}°C,{opt['T_mold']:.0f}°C)",
                    xy=(opt["T_pour"], opt["T_mold"]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=8, color="#1a1a2e",
                    arrowprops=dict(arrowstyle="->", color="#1a1a2e", lw=0.8))

        ax.set_xlabel("T_pour (°C)")
        ax.set_ylabel("T_mold (°C)")
        ax.set_title(f"{label} Risk", fontweight="bold")
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Risk Index", fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0 (Low)", "0.25", "0.50", "0.75", "1 (V.High)"],
                            fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save(f"sweep_results/contours/contour_all_defects_D{diam}mm.png")


# ═════════════════════════════════════════════════════════════════════════════
# C.  RADAR CHARTS — best + worst + per-diameter summary
#     For each diameter: four charts on one figure (best, worst, median, typical)
# ═════════════════════════════════════════════════════════════════════════════
print("C. Radar charts ...")

def _radar(ax, values, labels, color, title, fill_alpha=0.25):
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals   = values + values[:1]
    angs   = angles  + angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angs, vals, "o-", lw=2, color=color)
    ax.fill(angs, vals, color=color, alpha=fill_alpha)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10, color="#222")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7, color="#777")
    for ring, col in [(0.25, "#2ecc71"), (0.50, "#f1c40f"),
                      (0.75, "#e67e22"), (1.00, "#e74c3c")]:
        ax.plot(angs, [ring] * (n + 1), color=col, lw=0.6, ls="--", alpha=0.5)
    ax.set_title(title, fontsize=10, pad=16, fontweight="bold", color=color)

labels_radar = ["Misrun", "Cold\nShut", "Surface\nCracking", "Warpage"]

# ── One combined radar figure per diameter (best / worst / median) ──────────
for diam in diameters:
    sub  = df[df["diameter_mm"] == diam].copy()
    if sub.empty:
        continue

    best   = sub.loc[sub["total_risk"].idxmin()]
    worst  = sub.loc[sub["total_risk"].idxmax()]
    median_row = sub.iloc[(sub["total_risk"] - sub["total_risk"].median()).abs().argsort().iloc[0]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                             subplot_kw=dict(polar=True))
    fig.suptitle(f"Radar Charts  —  Diameter {diam} mm\n"
                 "Best / Median / Worst parameter sets",
                 fontsize=13, y=1.03)

    datasets = [
        (best,       "#2ecc71", f"BEST\nT_pour={best['T_pour']:.0f}°C, "
                                f"T_mold={best['T_mold']:.0f}°C\n"
                                f"Total risk = {best['total_risk']:.3f}"),
        (median_row, "#f1c40f", f"MEDIAN\nT_pour={median_row['T_pour']:.0f}°C, "
                                f"T_mold={median_row['T_mold']:.0f}°C\n"
                                f"Total risk = {median_row['total_risk']:.3f}"),
        (worst,      "#e74c3c", f"WORST\nT_pour={worst['T_pour']:.0f}°C, "
                                f"T_mold={worst['T_mold']:.0f}°C\n"
                                f"Total risk = {worst['total_risk']:.3f}"),
    ]
    for ax, (row, col, ttl) in zip(axes, datasets):
        vals = [row["MRI"], row["CSRI"], row["SCI"], row["WI_risk"]]
        _radar(ax, vals, labels_radar, col, ttl)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(f"sweep_results/radars/radar_D{diam}mm.png")


# ── All-diameters comparison radar on a single figure (best per diameter) ───
fig, axes = plt.subplots(1, len(diameters), figsize=(5 * len(diameters), 6),
                         subplot_kw=dict(polar=True))
if len(diameters) == 1:
    axes = [axes]
fig.suptitle("Best-parameter Radar Charts  —  All Diameters",
             fontsize=14, y=1.02)
for ax, diam in zip(axes, diameters):
    sub  = df[df["diameter_mm"] == diam]
    best = sub.loc[sub["total_risk"].idxmin()]
    vals = [best["MRI"], best["CSRI"], best["SCI"], best["WI_risk"]]
    col  = DIAM_COLOURS.get(diam, "#555")
    ttl  = (f"{diam} mm\n"
            f"T_pour={best['T_pour']:.0f}°C\n"
            f"T_mold={best['T_mold']:.0f}°C\n"
            f"Risk={best['total_risk']:.3f}")
    _radar(ax, vals, labels_radar, col, ttl)
plt.tight_layout()
_save("sweep_results/radars/radar_all_diameters_best.png")


# ═════════════════════════════════════════════════════════════════════════════
# D.  COMPOSITE RISK HEATMAPS  —  all diameters
#     Improved: gridded interpolation, ISO lines, sweet-spot star, annotations
# ═════════════════════════════════════════════════════════════════════════════
print("D. Composite risk heatmaps ...")

for diam in diameters:
    sub = df[df["diameter_mm"] == diam].copy()
    if len(sub) < 5:
        continue

    fig, ax = plt.subplots(figsize=(11, 9))

    xi = np.linspace(sub["T_pour"].min(), sub["T_pour"].max(), 100)
    yi = np.linspace(sub["T_mold"].min(), sub["T_mold"].max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((sub["T_pour"], sub["T_mold"]), sub["total_risk"],
                  (XI, YI), method="cubic")
    ZI = np.clip(ZI, 0, None)

    cf = ax.contourf(XI, YI, ZI, levels=40, cmap=RISK_CMAP, alpha=0.92)
    cs = ax.contour(XI, YI, ZI,
                    levels=[0.5, 1.0, 1.5, 2.0],
                    colors=["#166534", "#f1c40f", "#e67e22", "#b91c1c"],
                    linewidths=1.8)
    ax.clabel(cs, fmt={0.5: "0.5  Low", 1.0: "1.0  Mod",
                       1.5: "1.5  High", 2.0: "2.0  V.High"},
              fontsize=9, inline=True)

    # actual data points
    sc = ax.scatter(sub["T_pour"], sub["T_mold"],
                    c=sub["total_risk"], cmap=RISK_CMAP,
                    vmin=ZI.min(), vmax=ZI.max(),
                    s=55, edgecolors="#222", linewidths=0.5, zorder=6)

    # sweet-spot star
    opt = sub.loc[sub["total_risk"].idxmin()]
    ax.scatter(opt["T_pour"], opt["T_mold"],
               marker="*", s=420, color="#1a1a2e",
               zorder=12, edgecolors="white", linewidths=1.0)
    ax.annotate(
        f"  Sweet spot\n  T_pour={opt['T_pour']:.0f}°C\n"
        f"  T_mold={opt['T_mold']:.0f}°C\n"
        f"  Total risk={opt['total_risk']:.3f}",
        xy=(opt["T_pour"], opt["T_mold"]),
        xytext=(20, -35), textcoords="offset points",
        fontsize=9, color="#1a1a2e",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1a1a2e", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#1a1a2e", lw=1.0)
    )

    cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Total Risk Score  (sum of 4 defect indices)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0.5, 1.0, 1.5, 2.0, 2.5])

    ax.set_xlabel("T_mold  (°C)")
    ax.set_ylabel("T_pour  (°C)")
    ax.set_title(f"Composite Total Defect Risk  —  Diameter {diam} mm\n"
                 "Lower is better  |  ★ = sweet-spot parameter combination",
                 fontsize=13, fontweight="bold")

    _save(f"sweep_results/heatmaps/composite_risk_heatmap_D{diam}mm.png")


# ═════════════════════════════════════════════════════════════════════════════
# E.  SENSITIVITY ANALYSIS  (Random Forest feature importances)
#     One plot per diameter for solidification time and total risk
# ═════════════════════════════════════════════════════════════════════════════
print("E. Sensitivity analysis ...")

features      = ["T_pour", "T_mold", "diameter_mm"]
feat_labels   = ["T_pour (°C)", "T_mold (°C)", "Diameter (mm)"]

for diam in diameters:
    sub = df[df["diameter_mm"] == diam].copy()
    if len(sub) < 10:
        continue

    X = sub[features].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Random Forest Sensitivity Analysis  —  Diameter {diam} mm",
                 fontsize=13)

    targets = [
        ("solidification_time_s", "Solidification Time (s)", "#2980b9"),
        ("total_risk",            "Total Defect Risk",       "#e74c3c"),
    ]
    for ax, (target, tlabel, col) in zip(axes, targets):
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, sub[target].values)
        imp    = rf.feature_importances_
        idx    = np.argsort(imp)
        colours= [risk_colour(v / imp.sum()) for v in imp[idx]]

        bars = ax.barh([feat_labels[i] for i in idx],
                       imp[idx] * 100,
                       color=col, alpha=0.82, edgecolor="#222", linewidth=0.5)
        # value labels on bars
        for bar, val in zip(bars, imp[idx] * 100):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, color="#222")
        ax.set_xlabel("Feature Importance (%)")
        ax.set_title(f"Driver: {tlabel}", fontweight="bold")
        ax.set_xlim(0, 110)
        most = feat_labels[imp.argmax()]
        ax.text(0.98, 0.05, f"Top driver:\n{most}",
                transform=ax.transAxes, fontsize=9, ha="right",
                bbox=dict(boxstyle="round", fc="white", ec="#aaa", alpha=0.9))

    plt.tight_layout()
    _save(f"sweep_results/sensitivity/sensitivity_D{diam}mm.png")

# ── Global sensitivity (all diameters pooled) ────────────────────────────────
X_all = df[features].values
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Random Forest Sensitivity Analysis  —  All Diameters Combined",
             fontsize=13)
for ax, (target, tlabel, col) in zip(axes, [
        ("solidification_time_s", "Solidification Time (s)", "#2980b9"),
        ("total_risk",            "Total Defect Risk",       "#e74c3c")]):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_all, df[target].values)
    imp = rf.feature_importances_
    idx = np.argsort(imp)
    bars = ax.barh([feat_labels[i] for i in idx], imp[idx] * 100,
                   color=col, alpha=0.82, edgecolor="#222", linewidth=0.5)
    for bar, val in zip(bars, imp[idx] * 100):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title(f"Driver: {tlabel}", fontweight="bold")
    ax.set_xlim(0, 110)
plt.tight_layout()
_save("sweep_results/sensitivity/sensitivity_all_diameters.png")


# ═════════════════════════════════════════════════════════════════════════════
# F.  GAUSSIAN PROCESS RESPONSE SURFACE  —  all diameters
#     3D surface: solidification time ~ f(T_pour, T_mold)
#     + 2D uncertainty map (GP std)
# ═════════════════════════════════════════════════════════════════════════════
print("F. GP response surfaces ...")

scaler_gp = StandardScaler()
kernel    = CK(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)

for diam in diameters:
    sub = df[df["diameter_mm"] == diam].copy()
    if len(sub) < 8:
        continue

    X_d = sub[["T_pour", "T_mold"]].values
    y_d = sub["solidification_time_s"].values

    Xs  = scaler_gp.fit_transform(X_d)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   alpha=0.5, normalize_y=True)
    gpr.fit(Xs, y_d)

    tp_lin = np.linspace(sub["T_pour"].min(), sub["T_pour"].max(), 35)
    tm_lin = np.linspace(sub["T_mold"].min(), sub["T_mold"].max(), 35)
    TP, TM = np.meshgrid(tp_lin, tm_lin)
    X_grid = np.column_stack([TP.ravel(), TM.ravel()])
    X_gs   = scaler_gp.transform(X_grid)

    try:
        y_mu, y_sig = gpr.predict(X_gs, return_std=True)
    except Exception:
        y_mu  = gpr.predict(X_gs)
        y_sig = np.zeros_like(y_mu)

    MU  = y_mu.reshape(TP.shape)
    SIG = y_sig.reshape(TP.shape)

    # ── 3-D surface ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(f"GP Response Surface  —  Solidification Time  —  Diameter {diam} mm",
                 fontsize=13)

    # surface
    ax3 = fig.add_subplot(121, projection="3d")
    surf = ax3.plot_surface(TP, TM, MU, cmap="viridis",
                            edgecolor="none", alpha=0.82)
    ax3.scatter(sub["T_pour"], sub["T_mold"], sub["solidification_time_s"],
                color="red", s=30, zorder=10, label="Simulated")
    ax3.set_xlabel("T_pour (°C)", labelpad=8)
    ax3.set_ylabel("T_mold (°C)", labelpad=8)
    ax3.set_zlabel("t_sol (s)",   labelpad=8)
    ax3.set_title("Predicted Surface", fontsize=11)
    ax3.legend(loc="upper right", fontsize=8)
    fig.colorbar(surf, ax=ax3, shrink=0.5, pad=0.12, label="t_sol (s)")

    # uncertainty
    ax2 = fig.add_subplot(122)
    cf2 = ax2.contourf(TP, TM, SIG, levels=20, cmap="YlOrRd", alpha=0.9)
    ax2.contour(TP, TM, MU,
                levels=8, colors="white", linewidths=0.7, alpha=0.6)
    ax2.scatter(sub["T_pour"], sub["T_mold"],
                c="black", s=30, zorder=5, label="Simulated points")
    # mark minimum predicted solidification time (fastest solidification)
    idx_min = np.unravel_index(np.argmin(MU), MU.shape)
    ax2.scatter(TP[idx_min], TM[idx_min], marker="*", s=280,
                color="#1a1a2e", zorder=12, edgecolors="white")
    ax2.annotate(f"Fastest sol.\n({TP[idx_min]:.0f}°C, {TM[idx_min]:.0f}°C)",
                 xy=(TP[idx_min], TM[idx_min]),
                 xytext=(12, 12), textcoords="offset points", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", ec="#333", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.set_xlabel("T_pour (°C)")
    ax2.set_ylabel("T_mold (°C)")
    ax2.set_title("GP Uncertainty (std dev, s)", fontsize=11)
    cbar2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Std Dev (s)", fontsize=9)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(f"sweep_results/gp_surfaces/gp_response_surface_D{diam}mm.png")


# ═════════════════════════════════════════════════════════════════════════════
# G.  THERMAL FIELD PROFILES  —  heatmaps (cooling rate + thermal gradient)
# ═════════════════════════════════════════════════════════════════════════════
print("G. Thermal field heatmaps ...")

has_cr = "mean_cooling_rate_Ks"  in df.columns
has_tg = "mean_thermal_gradient_Km" in df.columns

if has_cr or has_tg:
    for diam in diameters:
        sub = df[df["diameter_mm"] == diam].copy()
        if len(sub) < 5:
            continue

        n_plots = (2 if has_cr else 0) + (2 if has_tg else 0)
        ncols   = 2
        nrows   = int(np.ceil(n_plots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        if nrows == 1: axes = axes.reshape(1, -1)
        fig.suptitle(f"Thermal Field Heatmaps  —  Diameter {diam} mm",
                     fontsize=13, y=1.01)
        axes_flat = axes.flatten()
        ax_idx = 0

        def _field_heatmap(ax, field, title, cmap, unit):
            xi = np.linspace(sub["T_pour"].min(), sub["T_pour"].max(), 60)
            yi = np.linspace(sub["T_mold"].min(), sub["T_mold"].max(), 60)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((sub["T_pour"], sub["T_mold"]), sub[field],
                          (XI, YI), method="linear")
            cf = ax.contourf(XI, YI, ZI, levels=25, cmap=cmap, alpha=0.9)
            ax.contour(XI, YI, ZI, levels=6, colors="white",
                       linewidths=0.6, alpha=0.5)
            ax.scatter(sub["T_pour"], sub["T_mold"],
                       c=sub[field], cmap=cmap,
                       s=35, edgecolors="#333", linewidths=0.3, zorder=5)
            cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit, fontsize=9)
            ax.set_xlabel("T_pour (°C)"); ax.set_ylabel("T_mold (°C)")
            ax.set_title(title, fontweight="bold")

        if has_cr:
            _field_heatmap(axes_flat[ax_idx],
                           "mean_cooling_rate_Ks",
                           "Mean Cooling Rate", "hot_r", "K/s")
            ax_idx += 1
            _field_heatmap(axes_flat[ax_idx],
                           "max_cooling_rate_Ks",
                           "Peak Cooling Rate", "hot_r", "K/s")
            ax_idx += 1
        if has_tg:
            _field_heatmap(axes_flat[ax_idx],
                           "mean_thermal_gradient_Km",
                           "Mean Thermal Gradient", "plasma", "K/m")
            ax_idx += 1
            _field_heatmap(axes_flat[ax_idx],
                           "max_thermal_gradient_Km",
                           "Peak Thermal Gradient", "plasma", "K/m")
            ax_idx += 1

        for i in range(ax_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()
        _save(f"sweep_results/heatmaps/thermal_fields_D{diam}mm.png")


# ═════════════════════════════════════════════════════════════════════════════
# H.  CORRELATION MATRIX
# ═════════════════════════════════════════════════════════════════════════════
print("H. Correlation matrix ...")

corr_cols = ["T_pour", "T_mold", "diameter_mm", "solidification_time_s",
             "MRI", "CSRI", "SCI", "WI_risk", "total_risk"]
for col in ["mean_cooling_rate_Ks", "max_cooling_rate_Ks",
            "mean_thermal_gradient_Km", "max_thermal_gradient_Km"]:
    if col in df.columns:
        corr_cols.append(col)

short_names = {
    "T_pour": "T_pour", "T_mold": "T_mold", "diameter_mm": "Diameter",
    "solidification_time_s": "t_sol",
    "MRI": "Misrun", "CSRI": "Cold Shut", "SCI": "Surf. Crack",
    "WI_risk": "Warpage", "total_risk": "Total Risk",
    "mean_cooling_rate_Ks": "CR mean", "max_cooling_rate_Ks": "CR max",
    "mean_thermal_gradient_Km": "TG mean", "max_thermal_gradient_Km": "TG max",
}

corr = df[corr_cols].rename(columns=short_names).corr()
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, square=True, vmin=-1, vmax=1,
    annot_kws={"size": 9},
    linewidths=0.5, linecolor="#ddd",
    cbar_kws={"shrink": 0.75, "label": "Pearson r"},
)
ax.set_title("Correlation Matrix — Process Parameters vs Defect Metrics",
             fontsize=13, pad=14)
ax.tick_params(axis="x", labelrotation=40, labelsize=10)
ax.tick_params(axis="y", labelrotation=0,  labelsize=10)
plt.tight_layout()
_save("sweep_results/correlation_matrix.png")


# ═════════════════════════════════════════════════════════════════════════════
# I.  CHVORINOV VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
print("I. Chvorinov validation ...")

C_CH = 1.488
fig, ax = plt.subplots(figsize=(9, 6.5))

# mean simulated solidification time per diameter
grp = df.groupby("diameter_mm")["solidification_time_s"].agg(["mean", "std"])
diam_arr  = np.array(sorted(df["diameter_mm"].unique()), dtype=float)
sim_mean  = grp.loc[diam_arr, "mean"].values
sim_std   = grp.loc[diam_arr, "std"].values
chv_times = C_CH * (diam_arr / 6.0) ** 2

ax.plot(diam_arr, chv_times, "s--", color="#7f7f7f", lw=2.0,
        label="Chvorinov: t = 1.488·(D/6)²")
ax.errorbar(diam_arr, sim_mean, yerr=sim_std, fmt="o-",
            color="#2980b9", lw=2.0, capsize=6, capthick=1.5,
            label="Simulated (mean ± 1 std)")

# deviation annotations
for d, ts, tc in zip(diam_arr, sim_mean, chv_times):
    dev = (ts - tc) / tc * 100
    flag = "FLAG" if abs(dev) > 15 else "OK"
    colour = "#e74c3c" if abs(dev) > 15 else "#2ecc71"
    ax.annotate(f"{dev:+.0f}%\n({flag})",
                xy=(d, ts), xytext=(0, 18), textcoords="offset points",
                ha="center", fontsize=8.5, color=colour,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=colour, alpha=0.9))

ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] else 800,
           alpha=0, zorder=0)   # ensure limits set
# ±15% bands around Chvorinov
ax.fill_between(diam_arr, chv_times * 0.85, chv_times * 1.15,
                alpha=0.12, color="#2980b9", label="±15% Chvorinov band")

ax.set_xlabel("Ball Diameter (mm)")
ax.set_ylabel("Solidification Time (s)")
ax.set_title("Chvorinov's Rule Validation\n"
             "C = 1.488 s/mm²  |  t = C·(D/6)²  |  >15% deviation = FLAG",
             fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(diam_arr.min() - 5, diam_arr.max() + 5)
_save("sweep_results/chvorinov_validation.png")


# ═════════════════════════════════════════════════════════════════════════════
# J.  SWEET-SPOT ANALYSIS  (no excessive temperature)
# ═════════════════════════════════════════════════════════════════════════════
print("J. Sweet-spot analysis ...")

T_POUR_LIMIT = 1450   # °C — avoid excessive temperature above this
RISK_CEILING = 0.35   # each individual defect risk must be below this

sweet = df[
    (df["T_pour"]  <= T_POUR_LIMIT) &
    (df["MRI"]     <= RISK_CEILING) &
    (df["CSRI"]    <= RISK_CEILING) &
    (df["SCI"]     <= RISK_CEILING) &
    (df["WI_risk"] <= RISK_CEILING)
].copy()
sweet = sweet.sort_values("total_risk")

# ── Sweet-spot process window scatter ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
# background: all runs as light grey
ax.scatter(df["T_pour"], df["T_mold"],
           c="#cccccc", s=25, alpha=0.4, label="All runs", zorder=1)
# sweet-spot runs coloured by diameter
for diam in diameters:
    sub_s = sweet[sweet["diameter_mm"] == diam]
    if not sub_s.empty:
        ax.scatter(sub_s["T_pour"], sub_s["T_mold"],
                   c=DIAM_COLOURS.get(diam, "blue"),
                   s=70, alpha=0.9, edgecolors="white", linewidths=0.4,
                   label=f"Sweet spot — {diam} mm", zorder=5)

# mark global optimum
if not sweet.empty:
    opt = sweet.iloc[0]
    ax.scatter(opt["T_pour"], opt["T_mold"],
               marker="*", s=440, color="#1a1a2e", zorder=12,
               edgecolors="white", linewidths=1.0)
    ax.annotate(
        f"  Global optimum\n  T_pour={opt['T_pour']:.0f}°C, T_mold={opt['T_mold']:.0f}°C\n"
        f"  Total risk = {opt['total_risk']:.3f}",
        xy=(opt["T_pour"], opt["T_mold"]),
        xytext=(20, 25), textcoords="offset points", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1a1a2e", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#1a1a2e", lw=1.0)
    )

ax.axvline(T_POUR_LIMIT, color="#e74c3c", ls="--", lw=1.5,
           label=f"T_pour limit = {T_POUR_LIMIT}°C")
ax.set_xlabel("T_pour (°C)")
ax.set_ylabel("T_mold (°C)")
ax.set_title(f"Sweet-Spot Process Window\n"
             f"T_pour ≤ {T_POUR_LIMIT}°C  |  All defect risks ≤ {RISK_CEILING}",
             fontsize=12)
ax.legend(fontsize=9, loc="upper left")
_save("sweep_results/process_window_sweet_spot.png")

sweet.to_csv("sweep_results/sweet_spot_runs.csv", index=False)


# ── Console recommendation report ────────────────────────────────────────────
print()
print("=" * 65)
print("  SWEET-SPOT RECOMMENDATIONS")
print("=" * 65)
print(f"  Criteria: T_pour ≤ {T_POUR_LIMIT}°C | all defect risks ≤ {RISK_CEILING}")
print(f"  Qualifying runs: {len(sweet)} / {len(df)}")
print()

if not sweet.empty:
    print("  TOP 5 LOWEST-RISK COMBINATIONS")
    print("  " + "-" * 60)
    print(f"  {'T_pour':>7} {'T_mold':>7} {'D(mm)':>6} | "
          f"{'Misrun':>7} {'ColdShut':>9} {'SurfCrk':>8} {'Warpage':>8} | "
          f"{'Total':>7}")
    print("  " + "-" * 60)
    for _, row in sweet.head(5).iterrows():
        print(f"  {row['T_pour']:>7.0f} {row['T_mold']:>7.0f} "
              f"{row['diameter_mm']:>6.0f} | "
              f"{row['MRI']:>7.3f} {row['CSRI']:>9.3f} "
              f"{row['SCI']:>8.3f} {row['WI_risk']:>8.3f} | "
              f"{row['total_risk']:>7.3f}")
    print()

    for diam in diameters:
        best_d = sweet[sweet["diameter_mm"] == diam]
        if best_d.empty:
            print(f"  {diam} mm: no sweet-spot found within constraints")
            continue
        b = best_d.iloc[0]
        print(f"  {diam} mm ball — recommended parameters:")
        print(f"    T_pour = {b['T_pour']:.0f} °C   T_mold = {b['T_mold']:.0f} °C")
        print(f"    Misrun={b['MRI']:.3f}  ColdShut={b['CSRI']:.3f}  "
              f"SurfCrack={b['SCI']:.3f}  Warpage={b['WI_risk']:.3f}")
        print(f"    Total risk = {b['total_risk']:.3f}  "
              f"({risk_label(b['total_risk'] / 4)})")
        print()
else:
    print("  No runs met all constraints. Consider raising RISK_CEILING or T_POUR_LIMIT.")

print()
print("All outputs saved in sweep_results/")
print("=" * 65)