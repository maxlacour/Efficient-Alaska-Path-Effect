# Efficient Alaska Path Effect (Python)

Python port of the Alaska non-ergodic within-path residual prediction code
(`Alaska_SKIP_Path_Effect_All_Sites.m`), optimised for speed.

## What it does

Predicts spatially-varying within-path residuals across Alaska using a
**Structured Kernel Interpolation (SKI / KISS-GP)** Gaussian Process with a
Hadamard-product Matérn-1/2 kernel.  For each prediction site the code:

1. Builds a per-site source grid (300 km radius, 5 km spacing).
2. Computes the site and source cross-covariances via bilinear interpolation
   onto precomputed SKI grids (Block-Toeplitz with Toeplitz Blocks structure,
   evaluated efficiently with 2-D FFTs).
3. Approximates the cross-covariance matrix with a randomised truncated SVD
   (`sklearn.utils.extmath.randomized_svd`).
4. Returns the posterior **mean** (median path prediction) and **variance**
   (epistemic uncertainty) at every source location in the grid.

Outputs are written as tab-delimited text files with one column per prediction
site and one row per source grid location.

## Repository contents

| File | Description |
|------|-------------|
| `alaska_skip_path_effect.py` | Main prediction script |
| `ski_utils.py` | SKI/BTTB kernel utilities (Matérn-1/2, bilinear weights, FFT MVM) |
| `cg_solver.py` | Conjugate-gradient solver for the alpha vector |
| `fast_svd_symmetric.py` | Randomised symmetric eigendecomposition (uses `scipy.sparse.linalg.eigsh`) |
| `fast_svd_rectangular.py` | Legacy custom rectangular SVD (kept for reference; replaced by sklearn in the main script) |
| `convert_svd_mat_to_npz.py` | Utility to convert a MATLAB `.mat` SVD file to `.npz` |
| `environment.yml` | Conda environment specification |
| `prediction_sites.txt` | Lat/lon coordinates of the prediction grid |
| `07_Alaska_data_Maxime_20260324/` | Observed within-path residuals (inter- and intra-event, multiple periods) |

## Installation

```bash
conda env create -f environment.yml
conda activate alaska_skip
```

## Usage

The script expects a precomputed SVD file `SVD_Alaska_inter_PSA_T_0.2.npz` in
the same directory.  If it is absent the SVD is computed from scratch (slow,
~minutes) and saved for subsequent runs.

```bash
python alaska_skip_path_effect.py
```

Outputs written to the script directory:

- `median_path_prediction.txt` — posterior mean, shape (n\_sources × n\_sites)
- `epistemic_path_prediction.txt` — posterior variance, same shape

## Performance optimisations

This port achieves roughly the same per-site runtime as the original MATLAB
code (~1 s/site on a 10-core laptop) through several changes:

- **sklearn randomized SVD** — `sklearn.utils.extmath.randomized_svd` wrapped
  in a `LinearOperator` replaces the hand-rolled rectangular SVD, giving better
  BLAS utilisation.
- **float32 FFT** — the BTTB matrix-vector multiply casts inputs to `float32`
  before the batched `rfft2`/`irfft2`, halving the memory traffic per MVM call
  (412 MB → 206 MB) and improving cache efficiency.
- **Multi-threaded FFT** — `scipy.fft` called with `workers=-1` to use all
  available cores.
- **Reformulated posterior variance** — avoids materialising a large
  `(n_path_cross × k_path)` intermediate by reordering the matrix products.
- **Minimal reckon calls** — spherical reckon is evaluated only for the
  within-circle grid points rather than the full rectangular mesh.

## References

- Wilson, A. G., & Nickisch, H. (2015). Kernel interpolation for scalable
  structured Gaussian processes (KISS-GP). *ICML*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
- Abrahamson, N., et al. — Alaska non-ergodic path effect model (internal
  report).
