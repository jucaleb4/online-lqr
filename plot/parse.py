import os

import numpy as np
import pandas as pd

def get_alg_perfs(folder):
    """
    Date: 2024 August 29th
    Commit: 2828ccf672e9021009ffbdcc4eeb2b719ed8e192

    We will look into folder/* for csv files with the following column structure:

        iter  samples Jgap J_K rho.

    :param folder: contains all algorithm performance files
    :return sample_arr: each row is a seed's sampling complexity at each checkpoint
    :return Jgap_arr: each row is a seed's progress of Jgap:=J(K_t)-J^*
    :return J_star: optimal value
    """
    file_arr = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    Jgap_arr = None
    sample_arr = None
    J_star = np.inf

    for i, fname in enumerate(file_arr):
        fullpath_fname = os.path.join(folder, fname)
        df = pd.read_csv(fullpath_fname, header="infer")

        if i == 0:
            # create array
            n_iter = df.shape[0]
            Jgap_arr = np.zeros((len(file_arr), n_iter), dtype=float)
            sample_arr = np.zeros((len(file_arr), n_iter), dtype=float)

            # get optimal value
            Jgap = df["Jgap"][0]
            J_K  = df["J_K"][0]
            J_star = J_K - Jgap

        sample_arr[i,:] = df["samples"].to_numpy()
        Jgap_arr[i,:] = df["Jgap"].to_numpy() 
        rhos = df["rho"].to_numpy()
        unstable_idxs = np.where(rhos >= 1)[0]
        if len(unstable_idxs) > 0:
            Jgap_arr[i, unstable_idxs] = np.inf

    return sample_arr, Jgap_arr, J_star

if __name__ == "__main__":
    (sample_arr, Jgap_arr, J_star) = get_alg_perfs('/home/jc4/Code/github/online-lqr/logs/npg_boeing')
