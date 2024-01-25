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

    params = dict({
        "po_eta": vargs["po_eta"],
        "po_total_iters": vargs["po_total_iters"],
        "eta": 10, # 10
        "lambda": 50, 
        "tau": 5, 
        "D": 20 if args.get("dynamic", False) else 10, # allow slightly larger diameter initially if we dynamically decrease
        "dynamic_D": args.get("dynamic", False),
        "total_iters": 100, 
        "total_epochs": 2,
        "vr": True,
        "minibatch": 10,
    })

    logger = lqr_logger.SimpleRunLogger(env, fname)
    po.npg(K_0, env, params, logger)
    logger.save()

def run_po_experiments(setup_env, num_runs, env_name, args):
    # TEMP
    for seed in range(1000, 1000+num_runs):
        s_time = time.time()
        fname = os.path.join("logs", f"{env_name}_env_seed={seed}.csv")
        (env, K_0) = setup_env(seed=seed)
        po_experiment(K_0, env, fname, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", 
                        choices=["simple", "cartpole", "boeing"], 
                        default="simple",
                        help="Which environment to run on")
    parser.add_argument("--tune", 
                        choices=["none", "stepsize", "iter", "po_stepsize"], 
                        default="none",
                        help="Which tuning mode. If none specified, runs experiment")
    parser.add_argument("--num_runs", 
                        type=int, 
                        default=1, 
                        help="Number of experiments to run (ignored if fine tuning)")
    parser.add_argument("--dynamic", 
                        action="store_true",
                        help="Dynamic updates the diameter")

    args = parser.parse_args()
    vargs = vars(args)

    setup_env = lambda : (None, None)
    if args.env == "simple":
        vargs["po_eta"] = 0.05
        vargs["po_total_iters"] = 30
        setup_env = example_envs.setup_simple_env
        env_name = "simple"
    elif args.env == "cartpole":
        vargs["po_eta"] = 0.05
        vargs["po_total_iters"] = 30
        setup_env = example_envs.setup_cartpole_env
        env_name = "cartpole"
    else:
        vargs["po_eta"] = 0.00025
        vargs["po_total_iters"] = 60
        setup_env = example_envs.setup_boeing_env
        env_name = "boeing"

    if args.tune == "stepsize":
        tune_stepsize_simple_env(setup_env)
    elif args.tune == "iter":
        tune_iters_simple_env(setup_env)
    elif args.tune == "po_stepsize":
        tune_po_simple_env(setup_env)
    else:
        run_po_experiments(setup_env, args.num_runs, env_name=env_name, args=vargs)
