# src/ml_defects.py
#
# ML DEFECT PREDICTOR  —  Shrinkage Porosity + Hot Tearing
# ─────────────────────────────────────────────────────────────────────────────
#
# TRAINING DATA SOURCE
# ────────────────────
# A 592-row balanced synthetic dataset generated from the casting defect
# tool (balanced_casting_defects HTML). The dataset was produced using
# physics-informed heuristic equations derived from casting literature:
#
#   Ny_min  ∝  D^1.3 · (T_mold/250)^0.9 / [(h_init/300)^0.8 · (h_gap/100)^0.3]
#   TSI_max ∝  (T_pour − T_mold) · (D/80)^0.6 · √(h_init·h_gap) / 300
#
# These relationships encode:
#   Ny_min  — Niyama criterion (shrinkage): increases with larger D and warmer
#             mold, decreases with aggressive heat extraction (high h)
#   TSI_max — Thermal Stress Index (hot tearing): increases with superheat,
#             diameter, and heat-extraction intensity
#
# The dataset is stratified across all 16 combinations of
# (Ny_class × TSI_class), each with 37 rows → exactly 148 per class.
# This eliminates the class-imbalance problem that caused the previous
# version to always predict "Low" for hot tearing.
#
# FEATURES (5 process parameters only)
# ─────────────────────────────────────
#   T_pour    : pouring temperature (°C)
#   T_mold    : mold temperature (°C)
#   D         : ball diameter (mm)
#   h_initial : initial heat transfer coefficient (W/m²K)
#   h_gap     : air-gap heat transfer coefficient (W/m²K)
#
# WHY PROCESS PARAMETERS, NOT SIMULATION FEATURES
# ─────────────────────────────────────────────────
# The previous approach extracted 11 simulation-derived features (cooling
# rates, DAS, MZRTI, etc.) from the thermal history. This had two problems:
#   1. The training sweep took 3312 seconds (not the estimated 20s) because
#      Python loop overhead dominated at N=40 nodes.
#   2. Features like MZRTI and DAS still produced imbalanced class labels
#      under permanent mold conditions.
#
# Using process parameters as features:
#   - Training takes < 2 seconds (no simulation sweep)
#   - The model is trained on a balanced 592-row dataset with guaranteed
#     coverage of all four risk classes
#   - The five parameters are directly controllable by the foundry operator
#   - Feature importances tell you which process lever matters most
#
# PREDICTION WORKFLOW
# ────────────────────
# For each simulation run, the model receives the five process parameters
# and outputs:
#   - Class label (Low / Moderate / High / Very High)
#   - Class probabilities for all four classes
#   - Confidence (probability of the predicted class)
#
# The physics-reference metrics (DAS, HCS, MZRTI) are still computed from
# the thermal history and printed alongside ML predictions for transparency.

import os
import sys
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ── Constants ──────────────────────────────────────────────────────────────
_CLASS_NAMES   = ['Low', 'Moderate', 'High', 'Very High']
_CLASS_COLOURS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

_FEATURE_NAMES = [
    'T_pour (°C)',
    'T_mold (°C)',
    'Diameter (mm)',
    'h_initial (W/m²K)',
    'h_gap (W/m²K)',
]

# Dataset label thresholds (from the HTML tool)
_NY_THRESHOLDS  = [0.4, 0.7, 1.0]   # Ny_min:  Low<0.4, Moderate 0.4-0.7, High 0.7-1.0, VH>1.0
_TSI_THRESHOLDS = [250, 450, 600]   # TSI MPa: Low<250, Moderate 250-450, High 450-600, VH>600

# Cache
_MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ml_cache')
_SHRINK_PATH  = os.path.join(_MODEL_DIR, 'shrinkage_rf.pkl')
_HOTTEAR_PATH = os.path.join(_MODEL_DIR, 'hottear_rf.pkl')
_SCALER_PATH  = os.path.join(_MODEL_DIR, 'scaler.pkl')


# ============================================================================
# DATASET GENERATION
# Reproduces the balanced_casting_defects HTML tool logic in Python.
# ============================================================================

def _generate_balanced_dataset(n_per_combo=37, seed=42):
    """
    Generate the balanced 592-row casting defect dataset.

    Reproduces the physics-informed heuristic equations from the HTML tool:
        Ny_min  ∝  D^1.3 · (T_mold/250)^0.9 / [(h_init/300)^0.8 · (h_gap/100)^0.3]
        TSI_max ∝  (T_pour−T_mold) · (D/80)^0.6 · √(h_init·h_gap) / 300

    Returns
    ───────
    X        : ndarray (N, 5)  — [T_pour, T_mold, D, h_init, h_gap]
    y_shrink : ndarray (N,)    — 0-3 Ny class
    y_tear   : ndarray (N,)    — 0-3 TSI class
    metadata : list[dict]      — full row info including raw Ny/TSI values
    """
    rng = np.random.default_rng(seed)

    ny_param_ranges = {
        0: dict(h_init=(100,350),  h_gap=(20,80),   T_mold=(300,450), D=(90,150)),   # Low
        1: dict(h_init=(300,700),  h_gap=(50,150),  T_mold=(250,350), D=(70,130)),   # Moderate
        2: dict(h_init=(600,1100), h_gap=(100,300), T_mold=(180,280), D=(60,100)),   # High
        3: dict(h_init=(900,1500), h_gap=(200,500), T_mold=(150,220), D=(50,80)),    # Very High
    }
    tsi_pour_ranges = {
        0: (1250, 1350),   # Low
        1: (1330, 1430),   # Moderate
        2: (1400, 1480),   # High
        3: (1450, 1550),   # Very High
    }
    ny_value_ranges  = {0:(1.0,2.2), 1:(0.7,1.0), 2:(0.4,0.7), 3:(0.0005,0.4)}
    tsi_value_ranges = {0:(50,250),  1:(250,450), 2:(450,600), 3:(600,1800)}

    X_rows = []
    y_s    = []
    y_t    = []
    meta   = []

    for ny_cls in range(4):
        for tsi_cls in range(4):
            for _ in range(n_per_combo):
                pr = ny_param_ranges[ny_cls]
                h_init = rng.uniform(*pr['h_init'])
                h_gap  = rng.uniform(*pr['h_gap'])
                T_mold = rng.uniform(*pr['T_mold'])
                D      = rng.uniform(*pr['D'])
                T_pour = rng.uniform(*tsi_pour_ranges[tsi_cls])
                if T_pour <= T_mold:
                    T_pour = T_mold + rng.uniform(50, 200)

                T_liq     = 1200.0
                superheat = max(0.0, T_pour - T_liq)
                cooling   = (h_init/300)**0.8 * (h_gap/100)**0.3
                Ny_min    = 1.2 * (D/100)**1.3 * (T_mold/250)**0.9 / (cooling + 0.3)
                Ny_min   /= (1 + 0.002 * superheat)
                Ny_min   *= (0.8 + rng.random() * 0.7)
                Ny_min    = float(np.clip(Ny_min, *ny_value_ranges[ny_cls]))

                dT      = max(20.0, T_pour - T_mold)
                h_eff   = (h_init * h_gap)**0.5
                TSI_max = 2.4 * dT * (D/80)**0.6 * (h_eff/300) * (1 + 0.003*superheat)
                TSI_max *= (0.7 + rng.random() * 0.8)
                TSI_max  = float(np.clip(TSI_max, *tsi_value_ranges[tsi_cls]))

                X_rows.append([T_pour, T_mold, D, h_init, h_gap])
                y_s.append(ny_cls)
                y_t.append(tsi_cls)
                meta.append({'T_pour':T_pour,'T_mold':T_mold,'D':D,
                             'h_init':h_init,'h_gap':h_gap,
                             'Ny_min':Ny_min,'TSI_max':TSI_max,
                             'ny_cls':ny_cls,'tsi_cls':tsi_cls})

    X  = np.array(X_rows, dtype=float)
    ys = np.array(y_s)
    yt = np.array(y_t)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], ys[idx], yt[idx], [meta[i] for i in idx]


# ============================================================================
# TRAINING
# ============================================================================

def train_models(verbose=True):
    """
    Generate the balanced dataset and train two Random Forest classifiers.

    Returns
    ───────
    rf_shrink, rf_tear : trained classifiers
    scaler             : fitted StandardScaler
    metadata           : full dataset row info
    cv_s, cv_t         : cross-validation accuracy arrays
    """
    X, y_shrink, y_tear, metadata = _generate_balanced_dataset()

    if verbose:
        print(f"  Dataset: {len(X)} rows  |  5 features  |  4 classes each")
        print(f"  Class distribution:")
        for i, name in enumerate(_CLASS_NAMES):
            print(f"    Shrinkage {name}: {np.sum(y_shrink==i):3d}  |  "
                  f"Hot Tear {name}: {np.sum(y_tear==i):3d}")
        print()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_shrink = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf_tear = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_s = cross_val_score(rf_shrink, X_scaled, y_shrink, cv=cv, scoring='accuracy')
    cv_t = cross_val_score(rf_tear,   X_scaled, y_tear,   cv=cv, scoring='accuracy')

    rf_shrink.fit(X_scaled, y_shrink)
    rf_tear.fit(X_scaled, y_tear)

    if verbose:
        print(f"  Shrinkage RF  CV: {cv_s.mean():.3f} ± {cv_s.std():.3f}")
        print(f"  Hot Tear  RF  CV: {cv_t.mean():.3f} ± {cv_t.std():.3f}")

    return rf_shrink, rf_tear, scaler, metadata, cv_s, cv_t


# ============================================================================
# SAVE / LOAD
# ============================================================================

def save_models(rf_shrink, rf_tear, scaler):
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for path, obj in [(_SHRINK_PATH, rf_shrink),
                      (_HOTTEAR_PATH, rf_tear),
                      (_SCALER_PATH,  scaler)]:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


def load_models():
    try:
        with open(_SHRINK_PATH,  'rb') as f: rf_shrink = pickle.load(f)
        with open(_HOTTEAR_PATH, 'rb') as f: rf_tear   = pickle.load(f)
        with open(_SCALER_PATH,  'rb') as f: scaler    = pickle.load(f)
        return rf_shrink, rf_tear, scaler
    except FileNotFoundError:
        return None, None, None


def models_are_cached():
    return all(os.path.exists(p) for p in
               [_SHRINK_PATH, _HOTTEAR_PATH, _SCALER_PATH])


# ============================================================================
# PREDICTION
# ============================================================================

def predict_defects(T_pour, T_mold, diameter_mm, h_initial, h_gap,
                    rf_shrink, rf_tear, scaler):
    """
    Predict shrinkage and hot tear risk from process parameters alone.

    Parameters
    ──────────
    T_pour, T_mold : process temperatures (°C)
    diameter_mm    : ball diameter (mm)
    h_initial      : initial HTC (W/m²K)
    h_gap          : air-gap HTC (W/m²K)

    Returns
    ───────
    dict with:
        shrink_class, shrink_label, shrink_proba, shrink_confidence
        tear_class, tear_label, tear_proba, tear_confidence
        features, feature_names
    """
    features = np.array([[T_pour, T_mold, diameter_mm, h_initial, h_gap]])
    X        = scaler.transform(features)

    s_cls = int(rf_shrink.predict(X)[0])
    t_cls = int(rf_tear.predict(X)[0])

    n_cls = 4
    s_proba_raw = rf_shrink.predict_proba(X)[0]
    t_proba_raw = rf_tear.predict_proba(X)[0]

    s_proba = np.zeros(n_cls)
    t_proba = np.zeros(n_cls)
    for i, c in enumerate(rf_shrink.classes_):
        s_proba[int(c)] = s_proba_raw[i]
    for i, c in enumerate(rf_tear.classes_):
        t_proba[int(c)] = t_proba_raw[i]

    return {
        'shrink_class'      : s_cls,
        'shrink_label'      : _CLASS_NAMES[s_cls],
        'shrink_proba'      : s_proba,
        'shrink_confidence' : float(s_proba[s_cls]),
        'tear_class'        : t_cls,
        'tear_label'        : _CLASS_NAMES[t_cls],
        'tear_proba'        : t_proba,
        'tear_confidence'   : float(t_proba[t_cls]),
        'features'          : features[0],
        'feature_names'     : _FEATURE_NAMES,
    }


# ============================================================================
# VISUALISATION
# ============================================================================

def plot_ml_results(pred, rf_shrink, rf_tear, diameter_mm,
                    label="", save_path=None, show=False):
    """
    Four-panel ML results figure.
    Top: probability bar charts for shrinkage + hot tearing.
    Bottom: feature importances for each model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#f8f9fa')
    title = f'ML Defect Prediction  —  {diameter_mm} mm Ball'
    if label:
        title += f'  |  {label}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)

    def _prob_panel(ax, proba, pred_cls, title_str, metric_note):
        edge = ['black' if i == pred_cls else 'white' for i in range(4)]
        lw   = [2.5    if i == pred_cls else 0.5     for i in range(4)]
        bars = ax.bar(_CLASS_NAMES, proba * 100,
                      color=_CLASS_COLOURS, edgecolor=edge, linewidth=lw)
        for bar, p in zip(bars, proba):
            if p > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.5,
                        f'{p*100:.0f}%', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability (%)', fontsize=11)
        ax.set_title(
            f'{title_str}\n'
            f'Prediction: {_CLASS_NAMES[pred_cls]}   '
            f'(confidence: {proba[pred_cls]*100:.0f}%)',
            fontsize=10, fontweight='bold',
            color=_CLASS_COLOURS[pred_cls]
        )
        ax.set_ylim(0, 125)
        ax.set_facecolor('#fdfdfd')
        ax.grid(axis='y', alpha=0.3)
        ax.text(0.98, 0.96, metric_note, transform=ax.transAxes,
                fontsize=8, ha='right', va='top', color='#555', style='italic')

    _prob_panel(
        axes[0, 0], pred['shrink_proba'], pred['shrink_class'],
        'Shrinkage Porosity',
        'Trained on Ny criterion thresholds\n<0.4 / 0.4-0.7 / 0.7-1.0 / >1.0'
    )
    _prob_panel(
        axes[0, 1], pred['tear_proba'], pred['tear_class'],
        'Hot Tearing',
        'Trained on TSI thresholds (MPa)\n<250 / 250-450 / 450-600 / >600'
    )

    def _imp_panel(ax, rf, title_str):
        imp   = rf.feature_importances_
        idx   = np.argsort(imp)
        vals  = imp[idx]
        names = [_FEATURE_NAMES[i] for i in idx]
        feats = pred['features']
        cols  = ['#e74c3c' if v > 0.25 else '#e67e22' if v > 0.15
                 else '#f1c40f' if v > 0.08 else '#2ecc71' for v in vals]
        ax.barh(range(len(names)), vals * 100, color=cols, edgecolor='white',
                linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Feature Importance (%)', fontsize=10)
        ax.set_title(title_str, fontsize=10, fontweight='bold')
        ax.set_facecolor('#fdfdfd')
        ax.grid(axis='x', alpha=0.3)
        for i, j in enumerate(idx):
            ax.text(vals[i] * 100 + 0.5, i,
                    f'  current: {feats[j]:.1f}',
                    va='center', fontsize=9, color='#333')
        # Colour legend
        patches = [mpatches.Patch(color=c, label=l) for c, l in [
            ('#e74c3c', 'Very strong (>25%)'),
            ('#e67e22', 'Strong (15-25%)'),
            ('#f1c40f', 'Moderate (8-15%)'),
            ('#2ecc71', 'Weak (<8%)'),
        ]]
        ax.legend(handles=patches, fontsize=7.5, loc='lower right')

    _imp_panel(axes[1, 0], rf_shrink,
               'Shrinkage — Feature Importances\n'
               '(which process parameter drives shrinkage risk most)')
    _imp_panel(axes[1, 1], rf_tear,
               'Hot Tearing — Feature Importances\n'
               '(which process parameter drives hot tear risk most)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved ML results  -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_training_overview(metadata, save_path=None, show=False):
    """
    Six-panel training data overview: class distributions, Ny vs parameters,
    TSI vs parameters, and the 2D class separation maps.
    """
    if not metadata:
        return

    T_pours  = np.array([m['T_pour']  for m in metadata])
    T_molds  = np.array([m['T_mold']  for m in metadata])
    Ds       = np.array([m['D']        for m in metadata])
    h_inits  = np.array([m['h_init']  for m in metadata])
    Ny_vals  = np.array([m['Ny_min']  for m in metadata])
    TSI_vals = np.array([m['TSI_max'] for m in metadata])
    ny_cls   = np.array([m['ny_cls']  for m in metadata])
    tsi_cls  = np.array([m['tsi_cls'] for m in metadata])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('ML Training Data  —  Balanced Synthetic Dataset (592 rows)',
                 fontsize=13, fontweight='bold')

    # 1. Class distribution bar chart
    ax = axes[0, 0]
    x = np.arange(4); w = 0.38
    ax.bar(x - w/2, [np.sum(ny_cls==i)  for i in range(4)],
           w, color=_CLASS_COLOURS, label='Shrinkage (Ny)', alpha=0.85)
    ax.bar(x + w/2, [np.sum(tsi_cls==i) for i in range(4)],
           w, color=_CLASS_COLOURS, label='Hot Tear (TSI)',
           edgecolor='black', lw=0.6, alpha=0.65)
    ax.set_xticks(x); ax.set_xticklabels(_CLASS_NAMES, fontsize=10)
    ax.set_ylabel('Count'); ax.set_title('Class Distribution (perfectly balanced)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # 2. Ny_min vs h_initial, coloured by class
    ax = axes[0, 1]
    for i in range(4):
        mask = ny_cls == i
        ax.scatter(h_inits[mask], Ny_vals[mask], c=_CLASS_COLOURS[i],
                   label=_CLASS_NAMES[i], s=20, alpha=0.6)
    for thr in _NY_THRESHOLDS:
        ax.axhline(thr, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('h_initial (W/m²K)'); ax.set_ylabel('Ny_min')
    ax.set_title('Shrinkage: Ny_min vs h_initial\n(high h → low Ny → high risk)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. TSI_max vs T_pour, coloured by class
    ax = axes[0, 2]
    for i in range(4):
        mask = tsi_cls == i
        ax.scatter(T_pours[mask], TSI_vals[mask], c=_CLASS_COLOURS[i],
                   label=_CLASS_NAMES[i], s=20, alpha=0.6)
    for thr in _TSI_THRESHOLDS:
        ax.axhline(thr, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('T_pour (°C)'); ax.set_ylabel('TSI_max (MPa)')
    ax.set_title('Hot Tearing: TSI_max vs T_pour\n(high T_pour → high TSI → high risk)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Ny class map: h_initial vs D
    ax = axes[1, 0]
    for i in range(4):
        mask = ny_cls == i
        ax.scatter(Ds[mask], h_inits[mask], c=_CLASS_COLOURS[i],
                   label=_CLASS_NAMES[i], s=20, alpha=0.7)
    ax.set_xlabel('Diameter (mm)'); ax.set_ylabel('h_initial (W/m²K)')
    ax.set_title('Shrinkage Class Map\n(D vs h_initial)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. TSI class map: T_pour vs T_mold
    ax = axes[1, 1]
    for i in range(4):
        mask = tsi_cls == i
        ax.scatter(T_pours[mask], T_molds[mask], c=_CLASS_COLOURS[i],
                   label=_CLASS_NAMES[i], s=20, alpha=0.7)
    ax.set_xlabel('T_pour (°C)'); ax.set_ylabel('T_mold (°C)')
    ax.set_title('Hot Tearing Class Map\n(T_pour vs T_mold)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Combined risk: Ny class vs TSI class heatmap
    ax = axes[1, 2]
    grid = np.zeros((4, 4))
    for ns, nt in zip(ny_cls, tsi_cls):
        grid[ns, nt] += 1
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(_CLASS_NAMES, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(_CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Hot Tear class'); ax.set_ylabel('Shrinkage class')
    ax.set_title('Co-occurrence of Defect Classes\n(how often they appear together)')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{int(grid[i,j])}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if grid[i,j] > 25 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved training overview -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# TOP-LEVEL ENTRY POINT
# ============================================================================

def train_and_predict(T_pour, T_mold, diameter_mm,
                      h_initial=None, h_gap=None,
                      force_retrain=False, verbose=True):
    """
    Train (or load cached) RF models on the balanced synthetic dataset,
    then predict for the given process parameters.

    Parameters
    ──────────
    T_pour, T_mold  : process temperatures (°C)
    diameter_mm     : ball diameter (mm)
    h_initial       : HTC — defaults to config.h_initial if None
    h_gap           : air-gap HTC — defaults to config.h_gap if None
    force_retrain   : ignore cached models and retrain
    verbose         : print training progress

    Returns
    ───────
    pred       : prediction dict
    rf_shrink  : trained shrinkage RF
    rf_tear    : trained hot-tear RF
    metadata   : training dataset rows
    """
    if h_initial is None:
        h_initial = config.h_initial
    if h_gap is None:
        h_gap = config.h_gap

    if not force_retrain and models_are_cached():
        if verbose:
            print("  Loading cached ML models (trained on balanced synthetic dataset) ...")
        rf_shrink, rf_tear, scaler = load_models()
        metadata = []
    else:
        if verbose:
            print("  Training ML models on balanced 592-row synthetic dataset ...")
            print("  (No simulation sweep required — training takes ~2 seconds)")
        rf_shrink, rf_tear, scaler, metadata, cv_s, cv_t = train_models(verbose=verbose)
        save_models(rf_shrink, rf_tear, scaler)
        if verbose:
            print(f"  Models cached to {_MODEL_DIR}")

    pred = predict_defects(T_pour, T_mold, diameter_mm, h_initial, h_gap,
                           rf_shrink, rf_tear, scaler)
    return pred, rf_shrink, rf_tear, metadata