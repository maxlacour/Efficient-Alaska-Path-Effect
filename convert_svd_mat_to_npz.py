"""
One-time conversion of SVD_Alaska_inter_PSA_T_0.2.mat -> SVD_Alaska_inter_PSA_T_0.2.npz
Run this once from the Python folder before running the main script.
"""

import os
import numpy as np
import scipy.io as sio

MAT_FILE = "../To_Share_SKIP_Alaska_Path_Effect/SVD_Alaska_inter_PSA_T_0.2.mat"
NPZ_FILE = "SVD_Alaska_inter_PSA_T_0.2.npz"

if os.path.isfile(NPZ_FILE):
    print(f"{NPZ_FILE} already exists — nothing to do.")
else:
    print(f"Loading {MAT_FILE} ...")
    d = sio.loadmat(MAT_FILE)
    U_path       = d["U_path"]
    D_path       = d["D_path"].ravel()
    alpha_vector = d["alpha_vector"].ravel()
    print(f"  U_path:       {U_path.shape}")
    print(f"  D_path:       {D_path.shape}  (min={D_path.min():.6f})")
    print(f"  alpha_vector: {alpha_vector.shape}")

    np.savez(NPZ_FILE, U_path=U_path, D_path=D_path, alpha_vector=alpha_vector)
    print(f"Saved {NPZ_FILE}")
