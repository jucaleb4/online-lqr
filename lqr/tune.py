"""
For finetuning CSPD
"""
import logging

import numpy as np
import numpy.linalg as la
import itertools

from lqr import po
from lqr import pe
from lqr import tools
from lqr import lqr_logger

def tune_stepsize_lqr_policy_eval(env, K, params, vr):
    """
    Tunes policy evaluation. Assumes `K` is stable
    """
    (n,k) = env.get_system_dimension()
    m = n+k
    nn = int(m*(m+1)/2)

    tau = 20
    N = 2000
    avg_H = np.zeros((nn+1, nn+1))
    avg_b = np.zeros(nn+1)

    logging.debug("Starting function evaluation estimation")
    # """
    for t in range(1,N+1):
        (H,b) = env.form_stochastic_Hb(K, tau)
        theta = 1./float(t)
        avg_H = (1.-theta)*avg_H + theta*H
        avg_b = (1.-theta)*avg_b + theta*b
    logging.debug("Finished function evaluation estimation")
    est_err = lambda p : la.norm(avg_H@p-avg_b)
    # """

    # TEMP (exact tuning)
    """
    [J,E] = pe.exact_policy_eval(K, env)
    def est_err(p):
        J_K = p[0]
        theta_K = p[1:]
        Theta_K = tools.smat(theta_K)
        T_22 = Theta_K[n:,n:]
        T_21 = Theta_K[n:,:n]
        E_K  = T_22@K - T_21
        return ((J-J_K)**2 + la.norm(E_K - E, ord='fro')**2)**0.5
    """

    params["total_iters"] = params.get("total_iters", 100)
    logger = lqr_logger.Logger()

    best_res = float("inf")

    rranges = [
        [10, 50, 100], # eta
        [10, 50, 100], # lambda
        [1, 5, 20],    # tau
        [10, 100],   # D
    ]

    best_res = float("inf")
    for tuned_param in itertools.product(*rranges):
        # to decorrelate
        for _ in range(10*tau):
            env.step(K)

        (eta, lam, tau, D) = tuned_param
        curr_params = dict({"eta": eta, "lambda": lam, "tau": tau, "D": D})
        params.update(curr_params)
        params["total_epochs"] = 1
        if vr:
            (_, _, p) = pe.policy_eval_vr(K, env, params, logger)
        else:
            (_, _, p) = pe.policy_eval(K, env, params, logger)
        res = est_err(p)

        if res < best_res:
            best_res = res
            mark = "<-- ***"
        else:
            mark = ""

        print(f"res={res:.2f} (eta={eta}, lam={lam} tau={tau} D={D}) {mark}")

def tune_iters_lqr_policy_eval(env, K, params, vr):
    """
    Tunes policy evaluation. Assumes `K` is stable
    """
    (n,k) = env.get_system_dimension()
    m = n+k
    nn = int(m*(m+1)/2)

    tau = 20
    N = 2000
    avg_H = np.zeros((nn+1, nn+1))
    avg_b = np.zeros(nn+1)

    logging.debug("Starting function evaluation estimation")
    for t in range(1,N+1):
        (H,b) = env.form_stochastic_Hb(K, tau)
        theta = 1./float(t)
        avg_H = (1.-theta)*avg_H + theta*H
        avg_b = (1.-theta)*avg_b + theta*b
    logging.debug("Finished function evaluation estimation")
    est_err = lambda p : la.norm(avg_H@p-avg_b)

    """
    [J,E] = pe.exact_policy_eval(K, env)
    def est_err(p):
        J_K = p[0]
        theta_K = p[1:]
        Theta_K = tools.smat(theta_K)
        T_22 = Theta_K[n:,n:]
        T_21 = Theta_K[n:,:n]
        E_K  = T_22@K - T_21
        return ((J-J_K)**2 * la.norm(E_K - E, ord='fro')**2)**0.5
    """

    rranges = [
        [100, 200, 500], # iters
        [1, 2, 4],       # epochs
    ]
    # """

    logger = lqr_logger.Logger()
    best_res = float("inf")
    for tuned_param in itertools.product(*rranges):
        # to decorrelate
        for _ in range(10*tau):
            env.step(K)

        (total_iters, total_epochs) = tuned_param
        curr_params = dict({"total_iters": total_iters, "total_epochs": total_epochs})
        params.update(curr_params)
        params["total_epochs"] = 1
        if vr:
            (_, _, p) = pe.policy_eval_vr(K, env, params, logger)
        else:
            (_, _, p) = pe.policy_eval(K, env, params, logger)
        res = est_err(p)

        if res < best_res:
            best_res = res
            mark = "**"
        else:
            mark = ""

        print(f"res={res:.2f} (total_iters={total_iters}, total_epochs={total_epochs}) {mark}")

def tune_lqr_policy_opt(env, K_0, params):
    """
    Tunes policy optimiztion using a (presumably) tunred policy evaluation under
    the hood. Assumes `K_0` is stable.
    """
    (n,k) = env.get_system_dimension()
    def est_J_K(K):
        tau = 5
        avg_cost = 0
        for t in range(1,1+30*(n+k)):
            for _ in range(tau): # de-correlate
                env.step(K)
            (_, cost) = env.step(K)
            theta = 1./float(t)
            avg_cost = (1.-theta)*avg_cost + theta*cost
        return avg_cost

    rranges = [
        [0.01, 0.1, 1., 10, 100], # po_eta
        [10],                     # po_total_iters
    ]

    logger = lqr_logger.Logger()
    best_J_K = float("inf")
    for tuned_param in itertools.product(*rranges):
        (po_eta, po_total_iters) = tuned_param
        curr_params = dict({"po_eta": po_eta, "po_total_iters": po_total_iters})
        params.update(curr_params)
        K = np.copy(K_0)

        # cheat about resetting
        env.reset()
        K = po.npg(K, env, params, logger)
        J_K = est_J_K(K)

        if J_K < best_J_K:
            best_J_K = J_K
            mark = "**"
        else:
            mark = ""

        print(f"J_K={J_K:.4e} (po_eta={po_eta}, po_total_iters={po_total_iters}) {mark}")
