"""
Utility functions for the Alaska SKI/KISS-GP path effect model.
Replaces MATLAB GPML toolbox calls: covMaterniso, apxGrid, km2deg/deg2km, reckon.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.fft import next_fast_len

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fft_mod
    pyfftw.interfaces.cache.enable()
    def rfft2(a, s=None, **kw):  return _fft_mod.rfft2(a, s=s)
    def irfft2(a, s=None, **kw): return _fft_mod.irfft2(a, s=s)
except ImportError:
    from scipy.fft import rfft2, irfft2


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG = np.pi * EARTH_RADIUS_KM / 180.0   # ~111.195 km / deg


def km2deg(km):
    return np.asarray(km) / KM_PER_DEG


def deg2km(deg):
    return np.asarray(deg) * KM_PER_DEG


# ---------------------------------------------------------------------------
# Spherical reckon  (replaces MATLAB Mapping Toolbox reckon)
# ---------------------------------------------------------------------------

def reckon(lat_deg, lon_deg, dist_deg, az_deg):
    """
    Compute destination point(s) given start (lat, lon), angular distance, and azimuth.

    Parameters
    ----------
    lat_deg, lon_deg : float — start point in degrees
    dist_deg         : array-like — angular distance(s) in degrees
    az_deg           : array-like — azimuth(s) in degrees (0 = North, clockwise)

    Returns
    -------
    lat2, lon2 : ndarrays in degrees
    """
    lat1 = np.deg2rad(lat_deg)
    lon1 = np.deg2rad(lon_deg)
    d    = np.deg2rad(np.asarray(dist_deg, dtype=float))
    az   = np.deg2rad(np.asarray(az_deg,   dtype=float))

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d) +
        np.cos(lat1) * np.sin(d) * np.cos(az)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(az) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2)
    )
    return np.rad2deg(lat2), np.rad2deg(lon2)


# ---------------------------------------------------------------------------
# Matérn-1/2 kernel  (replaces covMaterniso with d=1)
# ---------------------------------------------------------------------------

def matern12(X1, X2, rho, sf2):
    """
    Matérn-1/2 (exponential) covariance matrix.

    k(x, x') = sf2 * exp(-||x - x'|| / rho)

    Parameters
    ----------
    X1 : (n, d) array
    X2 : (m, d) array
    rho : length scale (same units as X)
    sf2 : signal variance

    Returns
    -------
    K : (n, m) array
    """
    R = cdist(X1, X2, metric='euclidean')
    return sf2 * np.exp(-R / rho)


# ---------------------------------------------------------------------------
# Bilinear interpolation weight matrix  (replaces apxGrid / GPML)
# ---------------------------------------------------------------------------

def bilinear_interp_weights(x, lat_grid, lon_grid):
    """
    Sparse bilinear interpolation weight matrix Mg (N x n_lat*n_lon).

    Each row has ≤ 4 nonzero entries (the four surrounding grid points).
    Uses row-major (C) ordering: flat index = i_lat * n_lon + i_lon.

    Parameters
    ----------
    x        : (N, 2) array of [lat, lon] data points
    lat_grid : (n_lat,) sorted 1-D array of grid latitudes
    lon_grid : (n_lon,) sorted 1-D array of grid longitudes

    Returns
    -------
    Mg : scipy CSR sparse matrix, shape (N, n_lat * n_lon)
    """
    N     = x.shape[0]
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)

    # Vectorised: process all N points at once with numpy
    lat_i = np.clip(x[:, 0], lat_grid[0], lat_grid[-1])
    lon_i = np.clip(x[:, 1], lon_grid[0], lon_grid[-1])

    i0 = np.clip(np.searchsorted(lat_grid, lat_i, side='right') - 1, 0, n_lat - 2)
    j0 = np.clip(np.searchsorted(lon_grid, lon_i, side='right') - 1, 0, n_lon - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    t = (lat_i - lat_grid[i0]) / (lat_grid[i1] - lat_grid[i0])
    u = (lon_i - lon_grid[j0]) / (lon_grid[j1] - lon_grid[j0])

    rows = np.repeat(np.arange(N), 4)
    cols = np.concatenate([i0 * n_lon + j0,
                           i0 * n_lon + j1,
                           i1 * n_lon + j0,
                           i1 * n_lon + j1])
    data = np.concatenate([(1 - t) * (1 - u),
                           (1 - t) * u,
                           t       * (1 - u),
                           t       * u])

    return csr_matrix((data, (rows, cols)), shape=(N, n_lat * n_lon))


# ---------------------------------------------------------------------------
# 2D BTTB grid covariance via FFT  (matches GPML bttbmvmsymfft / apxGrid)
# ---------------------------------------------------------------------------

class KronGrid:
    """
    2D Block-Toeplitz with Toeplitz Blocks (BTTB) covariance MVM via 2D FFT.

    Implements the GPML `bttbmvmsymfft` approach for the full 2D Matérn-1/2
    kernel with Euclidean distance:

        k(Δlat, Δlon) = sf2 * exp(-sqrt(Δlat² + Δlon²) / rho)

    Uses Strang circulant embedding of size (2*n_lat-1) × (2*n_lon-1) and
    scipy.fft.rfft2 / irfft2 for efficient MVM.

    Row-major convention: flat index = i_lat * n_lon + i_lon.
    Interface is identical to the old KronGrid so the rest of the code is unchanged.
    """

    def __init__(self, lat_grid, lon_grid, rho, sf2):
        n_lat = len(lat_grid)
        n_lon = len(lon_grid)
        self.n_lat = n_lat
        self.n_lon = n_lon

        delta_lat = (lat_grid[-1] - lat_grid[0]) / (n_lat - 1) if n_lat > 1 else 1.0
        delta_lon = (lon_grid[-1] - lon_grid[0]) / (n_lon - 1) if n_lon > 1 else 1.0

        # Strang circulant embedding indices (0-based, matching MATLAB 1-based offset)
        # Rows:    [0, 1, ..., n_lat-1,  -(n_lat-1), ..., -1]   (size 2*n_lat-1)
        # Columns: [0, 1, ..., n_lon-1,  -(n_lon-1), ..., -1]   (size 2*n_lon-1)
        i_lat = np.concatenate([np.arange(n_lat), np.arange(-(n_lat - 1), 0)])
        i_lon = np.concatenate([np.arange(n_lon), np.arange(-(n_lon - 1), 0)])

        I_lat, I_lon = np.meshgrid(i_lat, i_lon, indexing='ij')
        d_lat = I_lat * delta_lat
        d_lon = I_lon * delta_lon

        ci = sf2 * np.exp(-np.sqrt(d_lat ** 2 + d_lon ** 2) / rho)

        # Pad to next fast FFT size
        self.fft_lat = next_fast_len(2 * n_lat - 1)
        self.fft_lon = next_fast_len(2 * n_lon - 1)

        # Precompute FFT of the circulant first column (float64 and float32 versions)
        self.fi    = rfft2(ci, s=(self.fft_lat, self.fft_lon))
        self.fi_f32 = self.fi.astype(np.complex64)

    def mvm(self, v):
        """Compute K_grid @ v via 2D FFT convolution.

        v : (n_lat * n_lon, p) or (n_lat * n_lon,)
        """
        squeeze = v.ndim == 1
        if squeeze:
            v = v[:, np.newaxis]
        p     = v.shape[1]
        n_lat = self.n_lat
        n_lon = self.n_lon

        # Reshape to (p, n_lat, n_lon) for batch FFT.
        # Cast to float32 to halve memory traffic (412 MB → 206 MB per call);
        # precision is sufficient for the randomised sketching inside the SVD.
        b = v.reshape(n_lat, n_lon, p).transpose(2, 0, 1).astype(np.float32)

        B      = rfft2(b, s=(self.fft_lat, self.fft_lon), workers=-1)   # complex64
        B      = self.fi_f32[np.newaxis, :, :] * B
        result = irfft2(B, s=(self.fft_lat, self.fft_lon), workers=-1)  # float32

        # Truncate to (p, n_lat, n_lon), flatten, and return as float64
        out = result[:, :n_lat, :n_lon].transpose(1, 2, 0).reshape(n_lat * n_lon, p).astype(np.float64)
        return out[:, 0] if squeeze else out


def build_kron_grid(lat_grid, lon_grid, rho, sf2):
    """
    Build a KronGrid (2D BTTB FFT) from 1-D lat/lon grids and kernel hyperparameters.

    Parameters
    ----------
    lat_grid, lon_grid : 1-D arrays (uniform spacing assumed)
    rho  : length scale in degrees
    sf2  : signal variance

    Returns
    -------
    KronGrid instance
    """
    return KronGrid(lat_grid, lon_grid, rho, sf2)
