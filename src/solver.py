# src/solver.py
# THERMAL SOLVER  --  Semi-implicit backward-Euler, full N x N system
#
# ROOT CAUSE OF SURFACE OVER-SOLIDIFICATION (diagnosed and fixed here)
#
# THE BUG (previous version):
#   The surface node (i=N-1) was updated EXPLICITLY and INDEPENDENTLY:
#       T_new[surf] = T_old[surf] - 2*dt*q_mold / (rho*cp*dr)
#
#   This omits the conductive heat flux from the interior INTO the surface.
#   During the liquid and early-mushy phase, the hot interior drives enormous
#   conduction toward the surface:
#       q_cond_in = k * (T[N-2] - T_surf) / dr
#
#   Quantified for 60mm ball at t=5s (T_surf=1500C, T[N-2]=1545C):
#       q_cond_in  ~  2,047,500 W/m2   <- interior heat flowing INTO surface
#       q_mold     ~     50,000 W/m2   <- heat extracted to mold
#       Result: interior HEATS the surface; surface should barely cool at all.
#
#   The old code ignored q_cond_in entirely, letting the surface cool at
#   ~24 K/s from pure mold extraction, making it reach solidus temperature
#   while the centre is still at 1500C. Surface solidification in < 21 s
#   for a 60mm ball is a direct consequence of this omission.
#
# THE FIX (this version):
#   Include the surface node as the Nth equation in the tridiagonal system.
#   It is treated IMPLICITLY for diffusion (conduction) and EXPLICITLY only
#   for the nonlinear mold-flux term (radiation + convection).
#
#   Surface half-control-volume energy balance:
#       rho*Cp*(dr/2)*dT/dt = k*(T[N-2]-T[N-1])/dr  -  q_mold(T_old)
#
#   Discretised:
#       a[N-1]*T_new[N-2]  +  b[N-1]*T_new[N-1]  =  d[N-1]
#       a[N-1] = -2*alpha*dt/dr^2
#       b[N-1] =  1 + 2*alpha*dt/dr^2
#       d[N-1] =  T_old[N-1]  -  2*dt*q_mold / (rho*Cp*dr)
#
#   Physical effect: surface temperature is constrained by the thermal mass
#   of the interior. It cannot race ahead of the thermal wave. Expected
#   surface solidification: ~30-60 s for 60mm ball (vs ~21 s previously).

import numpy as np
import config
import src.materials as mat


def compute_properties(T):
    rho = np.array([mat.get_rho(t) for t in T])
    cp  = np.array([mat.get_cp(t)  for t in T])
    k   = np.array([mat.get_k(t)   for t in T])
    return rho, cp, k


def compute_surface_flux(T_surf, T_mold, fs_surf, h_initial, h_gap, time):
    """
    Heat flux leaving the casting surface (W/m2).

    Two-zone contact resistance:
      fs < fs_gap_onset  ->  R_contact_metal  (metal-mold contact, low resistance)
      fs >= fs_gap_onset ->  R_contact_gap    (air gap after shrinkage, high resistance)
    """
    h_raw = max(1.0, h_initial * (1.0 - fs_surf) + h_gap * fs_surf)

    fs_gap_onset = getattr(config, 'fs_gap_onset',    0.70)
    R_metal      = getattr(config, 'R_contact_metal', 0.001)
    R_gap        = getattr(config, 'R_contact_gap',   0.020)

    if fs_surf < fs_gap_onset:
        blend     = fs_surf / fs_gap_onset
        R_contact = R_metal + (R_gap - R_metal) * blend
    else:
        R_contact = R_gap

    h_eff  = 1.0 / (1.0 / h_raw + R_contact)
    q_conv = h_eff * (T_surf - T_mold)

    q_rad = 0.0
    if config.emissivity > 0:
        T_s_K = T_surf + 273.15
        T_m_K = T_mold + 273.15
        q_rad = config.emissivity * config.sigma * (T_s_K**4 - T_m_K**4)

    q_total = q_conv + q_rad
    q_cap   = getattr(config, 'q_total_cap', 150_000)
    return min(q_total, q_cap)


def thomas_algorithm(a, b, c, d):
    """
    Thomas algorithm for tridiagonal system:
        a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    Requires a[0]=0, c[-1]=0.
    """
    n  = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x  = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i]  / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def update_temperature(T, r, dr, dt, T_mold, h_initial, h_gap, time,
                       surface_fs=None, debug=False):
    """
    Full N x N semi-implicit temperature update.

    All nodes (centre through surface) solved simultaneously in one
    tridiagonal system. Diffusion is implicit (backward Euler). The
    nonlinear mold flux at the surface is treated explicitly.
    """
    N          = len(T)
    rho, cp, k = compute_properties(T)
    alpha      = k / (rho * cp)

    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    # -- Centre node (i=0): spherical symmetry, L'Hopital limit ---------------
    coeff = 3.0 * alpha[0] * dt / (dr * dr)
    b[0]  = 1.0 + coeff
    c[0]  = -coeff
    d[0]  = T[0]

    # -- Interior nodes (i=1 to N-2): spherical Laplacian ---------------------
    for i in range(1, N - 1):
        ri  = r[i]
        ai  = alpha[i]
        aL  = -ai * dt / (dr * dr) + ai * dt / (ri * dr)
        aC  =  1.0 + 2.0 * ai * dt / (dr * dr)
        aR  = -ai * dt / (dr * dr) - ai * dt / (ri * dr)
        a[i] = aL
        b[i] = aC
        c[i] = aR
        d[i] = T[i]

    # -- Surface node (i=N-1): HALF control-volume, implicit diffusion ---------
    #
    # Energy balance for shell of thickness dr/2:
    #   rho*Cp*(dr/2)*dT/dt = k*(T[N-2]-T[N-1])/dr  -  q_mold
    #
    # Rearranged (q_mold explicit from T_old, diffusion implicit):
    #   -(2*alpha*dt/dr^2)*T_new[N-2] + (1 + 2*alpha*dt/dr^2)*T_new[N-1]
    #       = T_old[N-1]  -  2*dt*q_mold / (rho*Cp*dr)
    #
    # c[N-1] = 0 satisfies Thomas algorithm's last-row requirement.
    # -------------------------------------------------------------------------
    fs_old  = mat.get_solid_fraction(T[N - 1])
    q_surf  = compute_surface_flux(T[N - 1], T_mold, fs_old, h_initial, h_gap, time)
    coeff_s = 2.0 * alpha[N - 1] * dt / (dr * dr)
    a[N - 1] = -coeff_s
    b[N - 1] = 1.0 + coeff_s
    c[N - 1] = 0.0
    d[N - 1] = T[N - 1] - 2.0 * dt * q_surf / (rho[N - 1] * cp[N - 1] * dr)

    # -- Solve full N x N system -----------------------------------------------
    T_new       = thomas_algorithm(a, b, c, d)
    fs_surf_new = mat.get_solid_fraction(T_new[N - 1])

    if debug:
        print(f"  [dbg] Ctr={T_new[0]:.1f}C  Surf={T_new[N-1]:.1f}C  "
              f"fs={fs_surf_new:.3f}  q_mold={q_surf:.0f} W/m2")

    return T_new, fs_surf_new


def compute_solid_fraction_profile(T):
    return np.array([mat.get_solid_fraction(t) for t in T])
