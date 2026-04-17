"""
Randomized rectangular SVD — Python port of Fast_SVD_Rectangular_Improved_function.m.

Computes a rank-k approximation  K ≈ U @ diag(D) @ V.T
where K is never formed explicitly; only two MVM handles are provided:
    K_RHS(x)  computes  K  @ x   (output has n_rows  rows)
    K_LHS(x)  computes  K.T @ x  (output has n_cols rows)

Algorithm: randomized SVD via subspace iteration (Halko, Martinsson & Tropp, §5.1 + §4.4).
"""

import numpy as np
from scipy.linalg import qr, svd as dense_svd


def fast_svd_rectangular(K_RHS, K_LHS, n, k, number_of_passes=1):
    """
    Randomized SVD of a rectangular matrix defined by two MVM handles.

    Parameters
    ----------
    K_RHS : callable v -> K @ v   (output shape: n_rows x ncols_v)
              OR (n_rows, n_cols) ndarray
    K_LHS : callable v -> K.T @ v (output shape: n_cols x ncols_v)
              OR (n_cols, n_rows) ndarray — i.e. the transpose
    n     : number of columns of K  (used to size the random sketch)
    k     : target rank
    number_of_passes : 1 or 2 subspace iteration passes (default 1)

    Returns
    -------
    U : (n_rows, k) left singular vectors
    D : (k,)        singular values, descending
    V : (n_cols, k) right singular vectors

    Such that  K ≈ U @ np.diag(D) @ V.T
    """
    p       = 10          # oversampling parameter
    l_value = min(k + p, n)

    def _apply_RHS(x):
        return K_RHS @ x if isinstance(K_RHS, np.ndarray) else K_RHS(x)

    def _apply_LHS(x):
        return K_LHS @ x if isinstance(K_LHS, np.ndarray) else K_LHS(x)

    # ------------------------------------------------------------------ Stage A
    # Build an orthonormal basis Q for range(K) via subspace iteration.

    Omega   = np.random.randn(n, l_value)
    Y0      = _apply_RHS(Omega)
    Q0, _   = qr(Y0, mode='economic')

    # Pass 1
    Y1t     = _apply_LHS(Q0)
    Q1t, _  = qr(Y1t, mode='economic')
    Y1      = _apply_RHS(Q1t)
    Q1, _   = qr(Y1, mode='economic')

    if number_of_passes == 2:
        Y2t    = _apply_LHS(Q1)
        Q2t, _ = qr(Y2t, mode='economic')
        Y2     = _apply_RHS(Q2t)
        Q2, _  = qr(Y2, mode='economic')
        Q_matrix = Q2
    else:
        Q_matrix = Q1

    # ------------------------------------------------------------------ Stage B
    B = _apply_LHS(Q_matrix).T          # (l_value, n_cols)
    U_tild, D_small, Vt = dense_svd(B, full_matrices=False)

    n_final       = min(k, D_small.shape[0])
    D_full        = D_small[:n_final]
    order         = np.argsort(D_full)[::-1]

    D_sorted = D_full[order]
    U_sorted = (Q_matrix @ U_tild[:, :n_final])[:, order]
    V_sorted = Vt[:n_final, :].T[:, order]

    return U_sorted, D_sorted, V_sorted
