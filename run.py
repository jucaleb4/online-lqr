import os
import time
import argparse

import numpy as np
import numpy.linalg as la
import scipy.linalg as spla

from lqr import tune
from lqr import po
from lqr import lqr_logger
from lqr import example_envs

import multiprocessing as mp

def tune_stepsize_simple_env(setup_env):
    """
    With seed=10, 

        {"eta":100, "lamda":100, "tau":5, "D": 10} # using variance reduction (minibatch=1)
        {"eta":50, "lambda=50", "tau":20, "D" :10} # using vr and simple cspd

    produces the best accuracy
    """
    (env, K_0) = setup_env(seed=10)

    params = dict({"total_iters": 100, "minibatch": 30})

    tune.tune_stepsize_lqr_policy_eval(env, K_0, params, vr=True)

def tune_iters_simple_env(setup_env):
    """
    With seed=20 (different seed then stepsize),  

        {"total_iters": 100, "total_epochs": 2}

    produces the best accuracy.
    """
    (env, K_0) = setup_env(seed=20)

    # obtained from `tune_stepsize_simple_env`
    # params = dict({"eta": 100, "lambda": 100, "tau": 5, "D": 10})
    params = dict({"eta": 50, "lambda": 50, "tau": 20, "D": 10})

    tune.tune_iters_lqr_policy_eval(env, K_0, params, vr=True)

def tune_po_simple_env(setup_env):
    """
    With seed=30,  

        {"po_eta": 0.1}

    produces the best accuracy.
    """
    (env, K_0) = setup_env(seed=30)

    # obtained from `tune_stepsize_simple_env`
    params = dict({"eta": 100, "lambda": 100, "tau": 5, "D": 10, "total_iters": 100, "total_epochs": 2, "vr": True})

    tune.tune_lqr_policy_opt(env, K_0, params)

def po_experiment(K_0, env, fname=None, args={}):
    """ Runs simple environment """

    # whether to use our implemented NPG
    use_npg = True

    if args["alg"] == "npg":
        total_iters = 300
        total_epochs = 1
    elif args["alg"] == "npg_rst":
        total_iters = 100
        total_epochs = 2
    elif args["alg"] == "tts_ac":
        pe_total_iters = 30000
        total_iters = pe_total_iters*args["po_total_iters"]
        use_npg = False
    else:
        print("Unknown alg=%s" % args["alg"])
        return 

    logger = lqr_logger.SimpleRunLogger(env, fname, silent=args.get("parallel", False))
    if use_npg:
        params = dict({
            "po_eta": vargs["po_eta"],
            "po_total_iters": vargs["po_total_iters"],
            "eta": 10, # 10
            "lambda": 50, 
            "tau": 5, 
            "D": 20 if args.get("dynamic", False) else 10, # allow slightly larger diameter initially if we dynamically decrease
            "dynamic_D": args.get("dynamic", False),
            "total_iters": total_iters, 
            "total_epochs": total_epochs,
            "vr": True,
            "minibatch": 10,
        })
        po.npg(K_0, env, params, logger)
    else:
        params = dict( {
            "a_power": args["a_power"],
            "b_power": args["b_power"], 
            "alpha_0": args["po_alpha"],
            "beta_0": args["po_beta"],
            "log_n_iter": pe_total_iters,
            "total_iters": total_iters, 
        })
        po.tts_actor_critic(K_0, env, params, logger)
    logger.save()

def run_po_experiments(setup_env, num_runs, env_name, args):
    seed_0 = args.get("seed", 1000)
    alg = args['alg']
    folder = os.path.join("logs", "%s_%s" % (alg, env_name))
    if not os.path.exists(folder):
        os.makedirs(folder)

    def run_worker_po_experiment(seed):
        fname = os.path.join(folder, f"alg={alg}_{env_name}_env_seed={seed}.csv")
        (env, K_0) = setup_env(seed=seed)
        po_experiment(K_0, env, fname, args)

    if not args["parallel"]:
        for seed in range(seed_0, seed_0+num_runs):
            run_worker_po_experiment(seed)
        return
    
    num_cpu = mp.cpu_count()
    print("Parallel PO experiements with %d workers" % (num_cpu-1))
    worker_queue = []
    for seed in range(seed_0, seed_0+num_runs):
        if len(worker_queue) == num_cpu-1:
            # wait for all workers to finish
            for p in worker_queue:
                p.join()
            worker_queue = []
        p = mp.Process(target=run_worker_po_experiment, args=(seed,))
        p.start()
        worker_queue.append(p)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", 
                        choices=["simple", "cartpole", "boeing"], 
                        default="simple",
                        help="Which environment to run on")
    parser.add_argument("--tune", 
                        choices=["none", "stepsize", "iter", "po_stepsize", "tts_ac"], 
                        default="none",
                        help="Which tuning mode. If none specified, runs experiment")
    parser.add_argument("--num_runs", 
                        type=int, 
                        default=1, 
                        help="Number of experiments to run (ignored if fine tuning)")
    parser.add_argument("--dynamic", 
                        action="store_true",
                        help="Dynamic updates the diameter")
    parser.add_argument("--alg", 
                        choices=["npg", "npg_rst", "tts_ac"],
                        default="npg_rst", 
                        help="Algorithm")
    parser.add_argument("--seed", 
                        type=int, 
                        default=0, 
                        help="Seed counter")
    parser.add_argument("--parallel", 
                        action="store_true",
                        help="Use multiprocessing")

    args = parser.parse_args()
    vargs = vars(args)


    setup_env = lambda : (None, None)
    if args.env == "simple":
        # npg
        vargs["po_eta"] = 0.05
        # two-time scale (see theorem 2 from https://arxiv.org/pdf/2109.14756#page=14)
        vargs["a_power"] = 1
        vargs["b_power"] = 2./3
        vargs["po_alpha"] = 1e-1
        vargs["po_beta"] = 1e-4
        vargs["po_total_iters"] = 30
        setup_env = example_envs.setup_simple_env
        env_name = "simple"
    elif args.env == "cartpole":
        vargs["po_eta"] = 0.05
        vargs["po_total_iters"] = 30
        setup_env = example_envs.setup_cartpole_env
        env_name = "cartpole"
    else:
        # npg
        vargs["po_eta"] = 0.00025
        # two-time scale
        vargs["a_power"] = 1
        vargs["b_power"] = 2./3
        vargs["po_alpha"] = 1e-10
        vargs["po_beta"] = 1e-10 # 1e-9
        vargs["po_total_iters"] = 120
        setup_env = example_envs.setup_boeing_env
        env_name = "boeing"

    if args.tune == "stepsize":
        tune_stepsize_simple_env(setup_env)
    elif args.tune == "iter":
        tune_iters_simple_env(setup_env)
    elif args.tune == "po_stepsize":
        tune_po_simple_env(setup_env)
    elif args.tune == "tts_ac":
        tune.tune_tts_ac(setup_env)
    else:
        run_po_experiments(setup_env, args.num_runs, env_name=env_name, args=vargs)
