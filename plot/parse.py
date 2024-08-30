import os

import numpy as np
import pandas as pd

def get_alg_perfs(folder):
    """
    Date: 2024 August 29th
    Commit: 2828ccf672e9021009ffbdcc4eeb2b719ed8e192

    We will look into <folder>/* for csv files with the following column structure:

        iter  samples Jgap J_K rho.

    :param folder: contains all algorithm performance files
    :return sample_arr: each row is a seed's sampling complexity at each checkpoint
    :return Jgap_arr: each row is a seed's progress of Jgap:=J(K_t)-J^*
    :return iter_arr: number of valid iterations for each seed's performance, i.e., use Jgap_arr[i,:iter_arr[i]] 
    :return J_star: optimal value
    """
    file_arr = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    Jgap_arr = None
    sample_arr = None
    iter_arr = np.zeros(len(file_arr), dtype=int)
    J_star = np.inf

    for i, fname in enumerate(file_arr):
        fullpath_fname = os.path.join(folder, fname)
        df = pd.read_csv(fullpath_fname, header="infer")

        iter_arr[i] = n_iter = df.shape[0]

        # create array or expand the array
        if Jgap_arr is None or Jgap_arr.shape[1] < n_iter:
            new_Jgap_arr = np.zeros((len(file_arr), n_iter), dtype=float)
            new_sample_arr = np.zeros((len(file_arr), n_iter), dtype=float)

            # copy old data
            for j in range(i):
                new_Jgap_arr[j,:iter_arr[j]] = Jgap_arr[j,:iter_arr[j]]
                new_sample_arr[j,:iter_arr[j]] = sample_arr[j,:iter_arr[j]]

            Jgap_arr = new_Jgap_arr
            sample_arr = new_sample_arr

        Jgap_arr[i,:iter_arr[i]] = df["Jgap"].to_numpy() 
        sample_arr[i,:iter_arr[i]] = df["samples"].to_numpy()
        rhos = df["rho"].to_numpy()
        if np.max(rhos) >= 1:
            first_unstable_idx = np.argmax(rhos >= 1)
            iter_arr[i] = first_unstable_idx
            # zero out invalid
            Jgap_arr[i,iter_arr[i]:] = 0

        if i == 0:
            # get optimal value
            Jgap = df["Jgap"][0]
            J_K  = df["J_K"][0]
            J_star = J_K - Jgap

    return sample_arr, Jgap_arr, iter_arr, J_star

if __name__ == "__main__":
    (sample_arr, Jgap_arr, J_star) = get_alg_perfs('/home/jc4/Code/github/online-lqr/logs/npg_boeing')
