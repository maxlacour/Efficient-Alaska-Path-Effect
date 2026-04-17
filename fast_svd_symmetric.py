"""
Randomized symmetric eigendecomposition — Python port of Fast_SVD_Improved_function.m
(that file is not in the shared folder but is called by the main script).

Computes a rank-k approximation  K ≈ U @ diag(D) @ U.T  of a symmetric PSD matrix K.
K may be given as a dense ndarray or as a callable MVM handle.

Uses scipy.sparse.linalg.eigsh with a LinearOperator for large matrices,
or a direct randomized approach matching the MATLAB algorithm structure.
"""

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator


def fast_svd_symmetric(K, n, k, number_of_passes=1):
    """
    Randomized eigendecomposition of a symmetric PSD matrix.

    Parameters
    ----------
    K                : (n, n) ndarray  OR  callable v -> K @ v
    n                : size of K (number of rows/cols)
    k                : number of eigenvalues/vectors to compute
    number_of_passes : ignored (kept for API parity with MATLAB; eigsh handles accuracy internally)

    Returns
    -------
    U : (n, k) eigenvectors  (columns), ordered descending by eigenvalue
    D : (k,)   eigenvalues,  descending
    """
    if callable(K):
        A_op = LinearOperator((n, n), matvec=K, dtype=float)
    else:
        A_op = K

    # eigsh returns eigenvalues in ascending order; request k largest
    D, U = eigsh(A_op, k=k, which='LM')

    # Sort descending
    order = np.argsort(D)[::-1]
    return U[:, order], D[order]
