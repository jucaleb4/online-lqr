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
        (sample_arr, Jgap_arr, iter_arr, J_star) = get_alg_perfs(folder)

        assert np.all(np.isfinite(Jgap_arr)), "recieved non-finite Jgap"
        assert np.min(Jgap_arr) >= 0, "recieved negative Jgap"

        # maximum number of iterations where controller was stable

        # longest last two seeds (so we have variance)
        sorted_iter_arr = np.argsort(iter_arr)
        m = int(len(sorted_iter_arr)/2)
        xs = sample_arr[sorted_iter_arr[m],:iter_arr[sorted_iter_arr[m]]]
        n_iter = iter_arr[sorted_iter_arr[m]]
        Jgap_mean = np.zeros(n_iter, dtype=float)
        Jgap_median = np.zeros(n_iter, dtype=float)
        Jgap_std = np.zeros(n_iter, dtype=float)
        prev_n_iter = 0

        for k in range(m):
            curr_n_iter = iter_arr[sorted_iter_arr[k]]

            if curr_n_iter == prev_n_iter:
                continue

            Jgap_subset = Jgap_arr[sorted_iter_arr[k:], prev_n_iter:curr_n_iter] 
            assert np.all(Jgap_subset > 0)
                
            Jgap_mean[prev_n_iter:curr_n_iter] = np.mean(Jgap_subset, axis=0)
            Jgap_median[prev_n_iter:curr_n_iter] = np.median(Jgap_subset, axis=0)
            Jgap_std[prev_n_iter:curr_n_iter] = np.std(Jgap_subset, axis=0)

            prev_n_iter = curr_n_iter

        J_T_median = Jgap_median[-1] + J_star
        print("Median perf of %s on %s achieved J(K_T)=%.2e (J_star=%.2e)" % (method, env, J_T_median, J_star))

        if np.min(Jgap_median) == 0:
            u = np.argmax(Jgap_median == 0)
            xs = xs[:u]
            Jgap_median = Jgap_median[:u]
            Jgap_mean = Jgap_mean[:u]
            Jgap_std = Jgap_std[:u]

        u = np.argmax(Jgap_median)
        ax.plot(xs, Jgap_median, color=colors[i], linestyle=lss[i], label=method_name[i])
        ax.fill_between(
            xs, 
            np.maximum(1e-16, Jgap_mean-1.95*Jgap_std) if ylog else Jgap_mean-Jgap_std, 
            Jgap_mean+1.95*Jgap_std, 
            color=colors[i], 
            alpha=0.25
        )

        if not ylog or np.min(Jgap_mean-Jgap_std) > 0:
            min_val = min(min_val, 0.9*np.min(Jgap_mean-Jgap_std))
        max_val = max(max_val, 1.1*np.max(Jgap_mean+Jgap_std))

    ax.legend()
    ax.set(
        yscale="log" if ylog else "linear",
        xlabel="Total samples", 
        ylabel=r"Gap = $J(K_t)-J(K^*)$", 
        title="Performance on %s" % env,
        ylim=(0.5*min_val if ylog else 0, 1.05*max_val),
    )
    assert min_val < max_val
    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    yticks = np.exp(np.linspace(np.log(0.5*min_val), np.log(max_val), num=6, endpoint=True))
    ax.set_yticks(yticks, labels=["%.1e" % ytick for ytick in yticks])
    plt.tight_layout()

    if sname is None:
        plt.show()
    else:
        plt.savefig(sname, dpi=240)
    plt.close()

def plot_all_lqr_comparison_experiments(env_arr, log_folder, ylog=False, sname=None):
    """
    """
    method_arr = ["npg_rst", "npg", "tts_ac"]
    method_name = ["Multi-epoch NPG", "Single-epoch NPG", "Two-time scale AC"]

    colors=["red","green","blue","blue","purple"]
    mkrs=["s","o",".",r'$\clubsuit$','$\Phi$']
    lss=["solid","dashed","dotted","dashdot",(1,(3,5,1,5,1,5))]

    plt.style.use("ggplot")
    fig, axes = plt.subplots(ncols=3)
    fig.set_size_inches(14,4)

    for z, env in enumerate(env_arr):
        ax = axes[z]

        min_val = np.inf
        max_val = 0

        for i, method in enumerate(method_arr):
            folder = os.path.join(log_folder, "%s_%s" % (method, env))
            (sample_arr, Jgap_arr, iter_arr, J_star) = get_alg_perfs(folder)

            assert np.all(np.isfinite(Jgap_arr)), "recieved non-finite Jgap"
            assert np.min(Jgap_arr) >= 0, "recieved negative Jgap"

            # sort seed by duration the algorithm was stable
            sorted_iter_arr = np.argsort(iter_arr)
            # we only want the times when at least 60% (first 40%) of seeds have stable policy
            m = int(len(sorted_iter_arr)*0.40)
            # since there may be ties, find the number of seeds that match the 10% percentile iter_ct
            print("Number of successful after %d iters: %d" % (
                sample_arr[sorted_iter_arr[m-1], m],
                np.sum(iter_arr <= iter_arr[sorted_iter_arr[m-1]]),
            ))
            xs = sample_arr[sorted_iter_arr[m],:iter_arr[sorted_iter_arr[m]]]
            n_iter = iter_arr[sorted_iter_arr[m]]
            Jgap_mean = np.zeros(n_iter, dtype=float)
            Jgap_median = np.zeros(n_iter, dtype=float)
            Jgap_std = np.zeros(n_iter, dtype=float)
            prev_n_iter = 0

            for k in range(m):
                curr_n_iter = iter_arr[sorted_iter_arr[k]]

                if curr_n_iter == prev_n_iter:
                    continue

                Jgap_subset = Jgap_arr[sorted_iter_arr[k:], prev_n_iter:curr_n_iter] 
                assert np.all(Jgap_subset > 0)
                    
                Jgap_mean[prev_n_iter:curr_n_iter] = np.mean(Jgap_subset, axis=0)
                Jgap_median[prev_n_iter:curr_n_iter] = np.median(Jgap_subset, axis=0)
                Jgap_std[prev_n_iter:curr_n_iter] = np.std(Jgap_subset, axis=0)

                prev_n_iter = curr_n_iter

            """
            if np.min(Jgap_median) == 0:
                u = np.argmax(Jgap_median == 0)
                xs = xs[:u]
                Jgap_median = Jgap_median[:u]
                Jgap_mean = Jgap_mean[:u]
                Jgap_std = Jgap_std[:u]
            """

            if z == 1 and i < 2:
                u = len(xs)//2
                xs = xs[:u]
                Jgap_median = Jgap_median[:u]
                Jgap_mean = Jgap_mean[:u]
                Jgap_std = Jgap_std[:u]

            J_T_median = Jgap_median[-1] + J_star
            print("Median perf of %s on %s achieved J(K_T)=%.2e (J_star=%.2e)" % (method, env, J_T_median, J_star))

            u = np.argmax(Jgap_median)
            ax.plot(xs, Jgap_median, color=colors[i], linestyle=lss[i], label="%s (%.2e)" % (method_name[i], J_T_median))
            ax.fill_between(
                xs, 
                np.maximum(1e-16, Jgap_mean-1.95*Jgap_std) if ylog else Jgap_mean-Jgap_std, 
                Jgap_mean+1.95*Jgap_std, 
                color=colors[i], 
                alpha=0.25
            )

            if not ylog or np.min(Jgap_mean-Jgap_std) > 0:
                min_val = min(min_val, 0.9*np.min(Jgap_mean-Jgap_std))
            max_val = max(max_val, 1.1*np.max(Jgap_mean+Jgap_std))

        ax.legend()
        ax.set(
            yscale="log" if ylog else "linear",
            xlabel="Total samples", 
            title="Performance on %s" % env,
            ylim=(0.95*min_val if ylog else 0, 1.05*max_val),
        )
        if z==0:
            ax.set_ylabel(r"Gap = $J(K_t)-J(K^*)$") 

        assert min_val < max_val
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

        yticks = np.exp(np.linspace(np.log(0.5*min_val), np.log(max_val), num=4, endpoint=True))
        ax.set_yticks(yticks, labels=["%.1e" % ytick for ytick in yticks])

    plt.tight_layout()

    if sname is None:
        plt.show()
    else:
        plt.savefig(sname, dpi=240)
    plt.close()

if __name__ == "__main__":
    env_arr = ["simple", "large_simple", "boeing"]
    root = "/home/jc4/Code/github/online-lqr"
    log_folder = os.path.join(root, "logs")
    save_folder = os.path.join(root, "plot")
    sname = None

    sname = os.path.join(save_folder, "all_perfs.png")
    plot_all_lqr_comparison_experiments(env_arr, log_folder, ylog=True, sname=sname)
    """
    for env in env_arr:
        print("Plotting env %s" % env)
        sname = os.path.join(save_folder, "%s_perfs.png" % env)
        plot_lqr_comparison_experiments(env, log_folder, ylog=True, sname=sname)
    """
