"""
Alaska SKI Path Effect — Python port of Alaska_SKIP_Path_Effect_All_Sites.m

Predicts non-ergodic within-path residuals across Alaska using a
Hadamard-product Matérn-1/2 GP with SKI (Structured Kernel Interpolation).

Outputs
-------
median_path_prediction.txt      : (n_Pred x n_sites) tab-delimited
epistemic_path_prediction.txt   : (n_Pred x n_sites) tab-delimited
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Ensure imports and paths work regardless of Spyder's working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ski_utils import (
    km2deg, deg2km,
    matern12,
    bilinear_interp_weights,
    build_kron_grid,
    reckon,
)
from cg_solver import cg_solver
from fast_svd_symmetric import fast_svd_symmetric
from scipy.sparse.linalg import LinearOperator
from sklearn.utils.extmath import randomized_svd


# =============================================================================
# Paths (always relative to this script file, not the working directory)
# =============================================================================

DATA_DIR = os.path.join(SCRIPT_DIR, "07_Alaska_data_Maxime_20260324")
SVD_FILE  = os.path.join(SCRIPT_DIR, "SVD_Alaska_inter_PSA_T_0.2.npz")

# =============================================================================
# Load observations
# =============================================================================

csv_path = os.path.join(DATA_DIR, "Alaska_inter_PSA_T_0.2_new_Maxime_20260324.csv")
df = pd.read_csv(csv_path)

Site_Vector_Total     = df[["sta_lat", "sta_long"]].values
Source_Vector_Total   = df[["eq_lat", "eq_long"]].values
Length_Vector_Total   = df["R_rup"].values
Within_Site_Residuals = df["lnWS_nlme3"].values

n_path = len(Within_Site_Residuals)
y_WP   = Within_Site_Residuals.copy()

# =============================================================================
# Prediction site coordinates
# =============================================================================

site_input = np.loadtxt(os.path.join(SCRIPT_DIR, "prediction_sites.txt"))
site_lat_grid = site_input[:, 0]
site_lon_grid = site_input[:, 1]
n_sites_grid  = len(site_lat_grid)

print(f"Loaded {n_sites_grid} prediction sites from prediction_sites.txt")

# =============================================================================
# Source grid geometry (re-centred per site inside the loop)
# =============================================================================

radius_grid     = 300.0   # km
grid_spacing_km = 5.0     # km

x_vector = np.arange(-radius_grid, radius_grid + grid_spacing_km, grid_spacing_km)
y_vector = np.arange(-radius_grid, radius_grid + grid_spacing_km, grid_spacing_km)

x_grid, y_grid = np.meshgrid(x_vector, y_vector)

dist_grid_km  = np.sqrt(x_grid**2 + y_grid**2).ravel()
azimuth_grid  = np.mod(np.degrees(np.arctan2(x_grid, y_grid)), 360).ravel()

idx_circle    = dist_grid_km <= radius_grid
dist_grid_deg = km2deg(dist_grid_km)

# Precompute masked arrays — reckon only needs the within-circle points
dist_circle_deg = dist_grid_deg[idx_circle]
azimuth_circle  = azimuth_grid[idx_circle]

n_Pred = int(idx_circle.sum())

print(f"Source grid: {n_Pred} locations per site ({radius_grid:.0f} km radius, {grid_spacing_km:.0f} km spacing)")

# =============================================================================
# Hyperparameters
# =============================================================================

theta_factor   = 0.3
rho_path_Karen = deg2km(0.6)   # length scale in km  (~66.7 km)
sn             = 0.1

num_evals_SKIP               = 1000
tol_eval_truncation_SKIP     = 1e-2
tol_CG                       = 1e-6
maxit_number_CG              = int(1e4)

# =============================================================================
# SKI grids and interpolation weights
# =============================================================================

rho_in_deg = km2deg(rho_path_Karen)   # length scale in degrees (matches GPML hyp units)
sf2        = theta_factor             # signal variance

# ---- Site SKI grid ----

tlat_min_site = Site_Vector_Total[:, 0].min() - 0.5
tlat_max_site = Site_Vector_Total[:, 0].max() + 0.5
tlon_min_site = Site_Vector_Total[:, 1].min() - 0.5
tlon_max_site = Site_Vector_Total[:, 1].max() + 0.5

delta_deg_site          = rho_in_deg / 4.0
tlat_vector_path_site   = np.arange(tlat_min_site, tlat_max_site + delta_deg_site, delta_deg_site)
tlon_vector_path_site   = np.arange(tlon_min_site, tlon_max_site + delta_deg_site, delta_deg_site)

x_path_site  = Site_Vector_Total[:, :2]
Mg_path_site = bilinear_interp_weights(x_path_site, tlat_vector_path_site, tlon_vector_path_site)
covg_path_site = build_kron_grid(tlat_vector_path_site, tlon_vector_path_site, rho_in_deg, sf2)

print(f"Site SKI grid: {len(tlat_vector_path_site)} x {len(tlon_vector_path_site)} = "
      f"{len(tlat_vector_path_site)*len(tlon_vector_path_site)} points")

# ---- Source SKI grid (must cover training + all per-site prediction grids) ----

max_lat_offset = km2deg(radius_grid)
max_lon_offset = km2deg(radius_grid)

tlat_min_source = min(Source_Vector_Total[:, 0].min(), (site_lat_grid - max_lat_offset).min()) - 0.5
tlat_max_source = max(Source_Vector_Total[:, 0].max(), (site_lat_grid + max_lat_offset).max()) + 0.5
tlon_min_source = min(Source_Vector_Total[:, 1].min(), (site_lon_grid - max_lon_offset).min()) - 0.5
tlon_max_source = max(Source_Vector_Total[:, 1].max(), (site_lon_grid + max_lon_offset).max()) + 0.5

delta_deg_source        = rho_in_deg / 4.0
tlat_vector_path_source = np.arange(tlat_min_source, tlat_max_source + delta_deg_source, delta_deg_source)
tlon_vector_path_source = np.arange(tlon_min_source, tlon_max_source + delta_deg_source, delta_deg_source)

x_path_source  = Source_Vector_Total[:, :2]
Mg_path_source = bilinear_interp_weights(x_path_source, tlat_vector_path_source, tlon_vector_path_source)
covg_path_source = build_kron_grid(tlat_vector_path_source, tlon_vector_path_source, rho_in_deg, sf2)

print(f"Source SKI grid: {len(tlat_vector_path_source)} x {len(tlon_vector_path_source)} = "
      f"{len(tlat_vector_path_source)*len(tlon_vector_path_source)} points")

# =============================================================================
# SVD of K_path and alpha vector  (auto-detect: load .npz if present, else compute)
# =============================================================================

if os.path.isfile(SVD_FILE):

    print(f"Loading precomputed SVD and alpha from {SVD_FILE} ...")
    data         = np.load(SVD_FILE)
    U_path       = data["U_path"]
    D_path       = data["D_path"]
    alpha_vector = data["alpha_vector"]
    print(f"Loaded {len(D_path)} eigenvalues (min = {D_path.min():.6f})")

else:

    print(f"{SVD_FILE} not found — computing SVD and alpha from scratch ...")

    K_site_Full   = matern12(x_path_site,   x_path_site,   rho_in_deg, sf2)
    K_source_Full = matern12(x_path_source, x_path_source, rho_in_deg, sf2)
    K_path        = K_site_Full * K_source_Full   # Hadamard product

    print(f"K_path shape: {K_path.shape}")

    U_path, D_path = fast_svd_symmetric(K_path, n_path, 7 * num_evals_SKIP, number_of_passes=1)

    keep           = D_path >= tol_eval_truncation_SKIP
    U_path, D_path = U_path[:, keep], D_path[keep]

    print(f"Retained {len(D_path)} eigenvalues (min = {D_path.min():.6f})")

    UD_path       = U_path * D_path[np.newaxis, :]
    K_fn_with_sn2 = lambda x: sn**2 * x + UD_path @ (U_path.T @ x)

    print("Computing alpha vector via CG ...")
    alpha_vector, n_iter = cg_solver(K_fn_with_sn2, y_WP, tol=tol_CG / 100, maxit=maxit_number_CG)
    print(f"CG converged in {n_iter} iterations")

    np.savez(SVD_FILE, U_path=U_path, D_path=D_path, alpha_vector=alpha_vector)
    print(f"Saved SVD and alpha to {SVD_FILE}")

# =============================================================================
# Shared prediction quantities (always rebuilt from loaded/computed U, D)
# =============================================================================

UD_path         = U_path * D_path[np.newaxis, :]
K_fn_with_sn2   = lambda x: sn**2 * x + UD_path @ (U_path.T @ x)
D_svd_B         = D_path + sn**2   # regularised eigenvalues used in posterior variance

# =============================================================================
# Shared prediction parameters
# =============================================================================

kss_vector             = theta_factor**2 * np.ones(n_Pred)  # prior variance
num_rho_to_keep_cross  = 5
cutoff_km_cross        = num_rho_to_keep_cross * rho_path_Karen
EARTH_R                = 6371.0   # km
num_evals_taken_source = 200

# =============================================================================
# Preallocate output maps
# =============================================================================

mu_map  = np.full((n_sites_grid, n_Pred), np.nan)
Psi_map = np.full((n_sites_grid, n_Pred), np.nan)

# =============================================================================
# LinearOperator wrapper for sklearn's randomized_svd
# =============================================================================

class _BatchLinearOperator(LinearOperator):
    """Wraps two MVM callables into a LinearOperator with efficient batch matmat.

    fwd(V): (n_cols, k) -> (n_rows, k)   — computes M @ V
    bwd(V): (n_rows, k) -> (n_cols, k)   — computes M.T @ V
    """
    def __init__(self, shape, fwd, bwd):
        super().__init__(dtype=np.float64, shape=shape)
        self._fwd = fwd
        self._bwd = bwd

    def _matvec(self, v):
        return self._fwd(v[:, np.newaxis]).ravel()

    def _matmat(self, V):
        return self._fwd(V)

    def _rmatvec(self, v):
        return self._bwd(v[:, np.newaxis]).ravel()

    def _rmatmat(self, V):
        return self._bwd(V)


# =============================================================================
# Site prediction loop
# =============================================================================

print("Starting site grid prediction loop ...")

i_site_count = 0

_PROFILE = True   # set False to silence per-step breakdown after tuning

for i_site in range(n_sites_grid):

    site_lat_loop = site_lat_grid[i_site]
    site_lon_loop = site_lon_grid[i_site]

    _t = time.perf_counter()

    # ---- Build per-site source grid (only within-circle points) ----
    source_lat_loop, source_lon_loop = reckon(
        site_lat_loop, site_lon_loop,
        dist_circle_deg, azimuth_circle
    )
    Source_Pred_loop = np.column_stack([source_lat_loop, source_lon_loop])  # (n_Pred, 2)
    _t_reckon = time.perf_counter() - _t; _t = time.perf_counter()

    # ---- Site cross-covariance (rank-1, exact) ----
    # k_site_cross(i) = k_site(training_site_i, prediction_site*)
    k_site_cross_loop = matern12(
        x_path_site,
        np.array([[site_lat_loop, site_lon_loop]]),
        rho_in_deg, sf2
    ).ravel()  # (n_path,)
    _t_matern = time.perf_counter() - _t; _t = time.perf_counter()

    # ---- Subset training records within (radius_grid + cutoff) of prediction site ----
    dlat = np.deg2rad(Source_Vector_Total[:, 0] - site_lat_loop)
    dlon = np.deg2rad(Source_Vector_Total[:, 1] - site_lon_loop)
    lat1 = np.deg2rad(Source_Vector_Total[:, 0])
    lat2 = np.deg2rad(site_lat_loop)
    a    = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    dist_source_to_site = 2 * EARTH_R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    idx_cross = dist_source_to_site <= (radius_grid + cutoff_km_cross)
    n_path_cross = int(idx_cross.sum())
    _t_haversine = time.perf_counter() - _t; _t = time.perf_counter()

    if n_path_cross == 0:
        print(f"  Site {i_site+1}  (lat={site_lat_loop:.3f}, lon={site_lon_loop:.3f}) — "
              "no training data, assigning prior")
        mu_map[i_site, :]  = 0.0
        Psi_map[i_site, :] = theta_factor**2
        continue

    i_site_count += 1
    print(f"Site {i_site_count} / {n_sites_grid}  "
          f"(lat={site_lat_loop:.3f}, lon={site_lon_loop:.3f})  n_obs={n_path_cross}")

    n_cols_cross = max(n_path_cross, n_Pred)

    Mg_source_cross = Mg_path_source[idx_cross, :]          # (n_path_cross, m_grid)
    k_site_cross_sub = k_site_cross_loop[idx_cross]         # (n_path_cross,)

    # ---- Interpolation weights for this site's prediction source grid ----
    Mg_source_pred = bilinear_interp_weights(
        Source_Pred_loop,
        tlat_vector_path_source,
        tlon_vector_path_source
    )  # (n_Pred, m_grid)
    _t_bilinear = time.perf_counter() - _t; _t = time.perf_counter()

    # ---- MVM functions for source cross-covariance K_source(training, pred) ----
    # Convention matches MATLAB: LHS multiplies K from the left (n_path_cross output),
    # RHS multiplies K^T from the left (n_Pred output).
    # Swap when n_Pred > n_path_cross so SVD acts on the smaller output space first.

    def _mvm_cross(Mg_left, Mg_right, v):
        # Mg_left @ K_grid @ Mg_right.T @ v
        return Mg_left @ covg_path_source.mvm(Mg_right.T @ v)

    if n_Pred <= n_path_cross:
        # K_RHS(v): takes n_path_cross rows → output n_Pred  (= K_source.T @ v)
        # K_LHS(v): takes n_Pred rows → output n_path_cross  (= K_source @ v)
        K_RHS = lambda v: _mvm_cross(Mg_source_pred,  Mg_source_cross, v)
        K_LHS = lambda v: _mvm_cross(Mg_source_cross, Mg_source_pred,  v)
    else:
        # K_RHS(v): takes n_Pred rows → output n_path_cross  (= K_source @ v)
        # K_LHS(v): takes n_path_cross rows → output n_Pred  (= K_source.T @ v)
        K_RHS = lambda v: _mvm_cross(Mg_source_cross, Mg_source_pred,  v)
        K_LHS = lambda v: _mvm_cross(Mg_source_pred,  Mg_source_cross, v)

    # ---- Rectangular SVD of source cross-covariance via sklearn randomized_svd ----
    k_svd    = min(num_evals_taken_source, n_Pred + 10)
    n_rows_op = min(n_Pred, n_path_cross)
    op = _BatchLinearOperator((n_rows_op, n_cols_cross), K_RHS, K_LHS)
    U_src, D_src, Vt_src = randomized_svd(
        op, n_components=k_svd, n_oversamples=10, n_iter=1,
        power_iteration_normalizer="QR", random_state=None
    )
    V_src = Vt_src.T
    _t_svd = time.perf_counter() - _t; _t = time.perf_counter()

    # Truncate small singular values
    keep_src = D_src >= tol_eval_truncation_SKIP
    D_src    = D_src[keep_src]
    U_src    = U_src[:, keep_src]
    V_src    = V_src[:, keep_src]

    # ---- Merge site and source cross-covariance ----
    # K_{X,X*}(i,j) = k_site(site_i, site*) * K_source(source_i, source*_j)
    # = diag(k_site_cross_sub) * K_source_cross
    # => scale U rows by k_site_cross values
    U_merged = U_src * k_site_cross_sub[:, np.newaxis]   # (n_path_cross, k)
    D_merged = D_src                                       # (k,)
    V_merged = V_src                                       # (n_Pred, k)

    # Assign LHS/RHS such that K_{X,X*} = LHS @ diag(D) @ RHS.T
    if n_Pred <= n_path_cross:
        LHS = V_merged                                       # (n_path_cross, k)
        RHS = U_merged * D_merged[np.newaxis, :]            # (n_Pred, k)
    else:
        LHS = U_merged * D_merged[np.newaxis, :]            # (n_path_cross, k)
        RHS = V_merged                                       # (n_Pred, k)

    # ---- Posterior mean ----
    alpha_cross  = alpha_vector[idx_cross]                 # (n_path_cross,)
    mu_loop      = RHS @ (LHS.T @ alpha_cross)             # (n_Pred,)
    _t_mean = time.perf_counter() - _t; _t = time.perf_counter()

    # ---- Posterior variance ----
    # Psi = diag(K_{X*,X*}) - diag(K_{X*,X} @ (K + sn^2 I)^{-1} @ K_{X,X*})
    # (K + sn^2 I)^{-1} ≈ U_path @ diag(1/D_svd_B) @ U_path.T

    U_cross = U_path[idx_cross, :]                         # (n_path_cross, k_path)
    # B = P @ diag(1/D_svd_B) @ P.T  where  P = LHS.T @ U_cross  (k x k_path)
    P       = LHS.T @ U_cross                             # (k, k_path)
    B       = (P / D_svd_B[np.newaxis, :]) @ P.T          # (k, k)
    ks_diag = np.sum((RHS @ B) * RHS, axis=1)             # (n_Pred,)
    _t_var = time.perf_counter() - _t

    Psi_loop = kss_vector - ks_diag                        # (n_Pred,)

    mu_map[i_site, :]  = mu_loop
    Psi_map[i_site, :] = Psi_loop

    elapsed = _t_reckon + _t_matern + _t_haversine + _t_bilinear + _t_svd + _t_mean + _t_var
    if _PROFILE:
        print(f"  reckon={_t_reckon:.3f}  matern={_t_matern:.3f}  "
              f"haversine={_t_haversine:.3f}  bilinear={_t_bilinear:.3f}  "
              f"svd={_t_svd:.3f}  mean={_t_mean:.3f}  var={_t_var:.3f}  "
              f"total={elapsed:.2f}s")

print(f"Completed all {n_sites_grid} sites.")

# =============================================================================
# Write output files
# =============================================================================
# Rows = source locations (n_Pred), Columns = prediction sites (n_sites_grid)

pd.DataFrame(mu_map.T).to_csv(
    os.path.join(SCRIPT_DIR, "median_path_prediction.txt"),
    sep="\t", header=False, index=False)
pd.DataFrame(Psi_map.T).to_csv(
    os.path.join(SCRIPT_DIR, "epistemic_path_prediction.txt"),
    sep="\t", header=False, index=False)

print(f"Wrote median_path_prediction.txt    ({n_Pred} sources x {n_sites_grid} sites)")
print(f"Wrote epistemic_path_prediction.txt ({n_Pred} sources x {n_sites_grid} sites)")
