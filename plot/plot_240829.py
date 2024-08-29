import os

import numpy as np
import matplotlib.pyplot as plt

from parse import get_alg_perfs

def plot_lqr_comparison_experiments(env, log_folder, ylog=False, sname=None):
    """
    """
    method_arr = ["npg_rst", "npg", "tts_ac"]
    method_name = ["Multi-Epoch NPG", "NPG", "Two-time AC"]

    colors=["red","green","blue","blue","purple"]
    mkrs=["s","o",".",r'$\clubsuit$','$\Phi$']
    lss=["solid","dashed","dotted","dashdot",(1,(3,5,1,5,1,5))]

    plt.style.use("ggplot")
    _, ax = plt.subplots()
    min_val = np.inf
    max_val = 0

    for i, method in enumerate(method_arr):
        folder = os.path.join(log_folder, "%s_%s" % (method, env))
        (sample_arr, Jgap_arr, J_star) = get_alg_perfs(folder)

        Jgap_median = np.zeros(Jgap_arr.shape[1], dtype=float)
        Jgap_std = np.zeros(Jgap_arr.shape[1], dtype=float)
        # last time step where at least one finite J(K_t)
        last_t = len(Jgap_median)
        for t in range(len(Jgap_median)):
            Jgap_t = Jgap_arr[:,t]
            Jgap_t_finite = Jgap_t[np.isfinite(Jgap_t)]
            # HACK: can never be exactly 0
            Jgap_t_finite_and_pos = Jgap_t_finite[Jgap_t_finite > 0]
            if len(Jgap_t_finite_and_pos) == 0:
                last_t = t
                break
            Jgap_median[t] = np.median(Jgap_t_finite_and_pos)
            Jgap_std[t] = np.std(Jgap_t_finite_and_pos)

        Jgap_median = Jgap_median[:last_t]
        Jgap_std = Jgap_std[:last_t]
        xs = np.mean(sample_arr, axis=0)
        xs = xs[:last_t]

        J_T_median = Jgap_median[-1] + J_star
        print("Median perf of %s on %s achieved J(K_T)=%.2e (J_star=%.2e)" % (method, env, J_T_median, J_star))

        ax.plot(xs, Jgap_median, color=colors[i], linestyle=lss[i], label=method_name[i])
        ax.fill_between(xs, np.maximum(1e-16, Jgap_median-Jgap_std), Jgap_median+Jgap_std, color=colors[i], alpha=0.25)

        if np.min(Jgap_median-Jgap_std) > 1e-16:
            min_val = min(min_val, 0.9*np.min(Jgap_median-Jgap_std))
        max_val = max(max_val, 1.1*np.max(Jgap_median+Jgap_std))

    ax.legend()
    ax.set(
        yscale="log" if ylog else "linear",
        xlabel="Total samples", 
        ylabel=r"Gap = $J(K_t)-J(K^*)$", 
        title="Performance on %s" % env,
        ylim=(0.95*min_val, 1.05*max_val),
    )
    assert min_val < max_val
    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    yticks = np.exp(np.linspace(np.log(min_val), np.log(max_val), num=5, endpoint=True))
    ax.set_yticks(yticks, labels=["%.1e" % ytick for ytick in yticks])
    plt.tight_layout()

    if sname is None:
        plt.show()
    else:
        plt.savefig(sname, dpi=240)
    plt.close()

if __name__ == "__main__":
    env_arr = ["simple", "large_simple", "boeing"]
    env = env_arr[2]
    root = "/home/jc4/Code/github/online-lqr"
    log_folder = os.path.join(root, "logs")
    save_folder = os.path.join(root, "plot")

    for env in env_arr:
        print("Plotting env %s" % env)
        sname = os.path.join(save_folder, "%s_perfs.png" % env)
        plot_lqr_comparison_experiments(env, log_folder, ylog=True, sname=sname)
