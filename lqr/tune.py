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

    po_total_iters = 10

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

def tune_tts_ac(setup_env):
    """
    Tunes two-time scale actor-critic method (https://arxiv.org/pdf/2109.14756)
    """
    params = dict({"total_iters": 2000})
    logger = lqr_logger.Logger()

    rranges = [
        [10.**p for p in range(-10, 4)], # alpha_0
        [10.**p for p in range(-10, 4)], # beta_0
    ]
    params["a_power"] = 1
    params["b_power"] = 2./3
    best_J_K = float("inf")

    for tuned_param in itertools.product(*rranges):
        (env, K_0) = setup_env(seed=30)
        (alpha_0, beta_0) = tuned_param
        curr_params = dict({"alpha_0": alpha_0, "beta_0": beta_0})
        params.update(curr_params)
        K = po.tts_actor_critic(K_0, env, params, logger)
        if np.all(np.isfinite(K)) and la.norm(K) < 1e10:
            (J_K, _, rho) = pe.exact_policy_eval(K, env)
            if rho >= 1:
                J_K = float("inf")
        else:
            J_K = float("inf")
        print("J(K)=%.4e (alpha_0=%.2e beta_0=%.2e)%s" % (J_K, alpha_0, beta_0, " <-- ***" if J_K < best_J_K else ""))
        if J_K < best_J_K:
            best_J_K = J_K
