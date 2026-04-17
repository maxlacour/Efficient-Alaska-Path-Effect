"""
Conjugate gradient solver — Python port of CG_Solver_function.m.

Solves  A(x) = b  where A is given as a callable MVM function handle
(or a numpy matrix). Handles a single RHS vector.

Uses scipy.sparse.linalg.cg internally, wrapped so the interface matches
the MATLAB original: returns (alpha_vector, num_iterations).
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator


def cg_solver(A_mvm, b, tol=1e-6, maxit=10000, x0=None):
    """
    Solve A @ x = b via conjugate gradient.

    Parameters
    ----------
    A_mvm : callable (v -> A @ v) or (n, n) ndarray
    b     : (n,) array
    tol   : relative residual tolerance
    maxit : maximum iterations
    x0    : initial guess (default: zeros)

    Returns
    -------
    x    : (n,) solution vector
    info : int — 0 if converged, >0 = iteration count at exit without convergence
    """
    n = b.shape[0]

    if callable(A_mvm):
        A_op = LinearOperator((n, n), matvec=A_mvm, dtype=float)
    else:
        A_op = A_mvm

    x0 = np.zeros(n) if x0 is None else x0

    # scipy.cg callback to count iterations
    iters = [0]
    def _callback(_):
        iters[0] += 1

    x, info = cg(A_op, b, x0=x0, rtol=tol, maxiter=maxit, callback=_callback)

    return x, iters[0]
