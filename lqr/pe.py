from typing import Optional

import numpy as np
import numpy.linalg as la
import scipy.linalg as spla
import logging

from lqr import tools

""" Policy Evaluation """
def exact_policy_eval(K: np.ndarray, env):
    """ Exact policy evaluation via Lyapunov equation

    :param K: matrix gain/policy (assumed to be stabilizing)
    :param env: environment 
    :param X: current state
    :return J_K: cost of current controller
    :return E_K: natural gradient
    """
    (A, B, Q, R, Cov, sigma) = env.get_system_parameters()

    P_K = spla.solve_discrete_lyapunov((A-B@K).T, Q+K.T@R@K)
    Cov2 = Cov + sigma**2 * B@B.T
    J_K = np.trace(P_K@Cov2)
    rho = la.norm(A-B@K, ord=2)
    if rho >= 1:
        logging.debug("Given unstable matrix gain")
        J_K = np.inf

    T_22 = R + B.T@P_K@B
    T_21 = B.T@P_K@A
    E_K = T_22@K- T_21

    return (J_K, E_K)

def policy_eval_params_checker(params):
    assert params.__contains__("tau"), "Missing mixing time"

def policy_eval(K: np.ndarray, env, params, logger, p=None):
    """ Basic policy evaluation. See `_policy_eval` for more details """
    if params.get("vr", False):
        return policy_eval_vr(K, env, params, logger, p)
    else:
        get_stochastic_linsys = lambda : env.form_stochastic_Hb(K, params["tau"])
        return _policy_eval(K, env, params, logger, get_stochastic_linsys, p)

def policy_eval_vr(K: np.ndarray, env, params, logger, p=None):
    """ 
    Policy evaluation with variance reduction. See `_policy_eval` for more
    details 
    """
    def get_minibatch_stochastic_linsys(): 
        (avg_H, avg_b) = env.form_stochastic_Hb(K, params["tau"])

        for t in range(2, params.get("minibatch", 1)+1):
            # no more mixing
            (H, b) = env.form_stochastic_Hb(K, 1)
            theta = 1./float(t)
            avg_H = (1.-theta)*avg_H + theta*H
            avg_b = (1.-theta)*avg_b + theta*b

        return (avg_H, avg_b)

    # TODO: Why is covariate not working
    params["covariate"] = False

    def get_new_covariate():
        old_minibatch = params.get("minibatch", 1)
        (n,k) = env.get_system_dimension()
        params["minibatch"] = 32*(n+k)
        (H_0, b_0) = get_minibatch_stochastic_linsys()
        params["H_0"] = H_0
        params["b_0"] = b_0
        params["minibatch"] = old_minibatch

    params["get_new_covariate"] = get_new_covariate

    return _policy_eval(K, env, params, logger, get_minibatch_stochastic_linsys, p)

def _policy_eval(K: np.ndarray, env, params, logger, get_stochastic_linsys, p_0=None):
    """ Template policy evaluation. See `cspd()` for remaining parameters
    :param K: matrix gain:
    :param get_stochastic_linsys: function to return estimated linear system
    :param p_0: initial solution
    :return J_K: estimated objective function
    :return E_K: estimated knatural gradient
    :return p: primal solution
    """
    policy_eval_params_checker(params)

    (n,k) = env.get_system_dimension()
    m = n+k 
    nn = int(m*(m+1)/2)

    # TODO: Warm-start primal solution
    p_0 = np.zeros(nn+1, dtype=float) if p_0 is None else p_0
    (p, _) = sme_cspd(get_stochastic_linsys, p_0, params, logger)

    J_K = p[0]
    theta_K = p[1:]
    Theta_K = tools.smat(theta_K)
    T_22 = Theta_K[n:,n:]
    T_21 = Theta_K[n:,:n]
    E_K  = T_22@K - T_21

    return (J_K, E_K, p)

def sme_cspd_param_check(params):
    """ Input checker for `cspd` """
    cspd_param_check(params)
    assert params.__contains__("total_epochs"), "Missing number of epochs"

def sme_cspd(
        get_stochastic_linsys, 
        p_0: np.ndarray, 
        params: dict,
        logger: object,
    ):
    """ 
    Shrinking multi-epoch conditional stochastic primal dual (CSPD). Check
    `cspd` for info about inputs and returns.
    """
    sme_cspd_param_check(params)

    p = p_0
    y = np.zeros(len(p))

    for t in range(params["total_epochs"]):
        # logging.debug(f"Epoch {t+1}/{params['total_epochs']}")

        if params.get("covariate", False):
            # refresh covariate
            params["get_new_covariate"]()

        params_copy = dict(params)
        params_copy["total_iters"] = params["total_iters"] * 2**t
        params_copy["eta"] = params["eta"] * 2**t
        params_copy["D"] = params["D"] * 2**(-t)

        (p, y) = cspd(get_stochastic_linsys, p, y, params_copy, logger)

        logger.log(t, p, label="sme_cspd")

    return (p, y)

def cspd_param_check(params):
    """ Input checker for `cspd` """
    assert params.__contains__("eta"), "Missing initial primal step size"
    assert params.__contains__("lambda"), "Missing initial dual step size"
    assert params.__contains__("total_iters"), "Missing total number of iterations"
    assert params.__contains__("D"), "Missing diameter"

def cspd(
        get_stochastic_linsys, 
        p_0: np.ndarray, 
        y_0: Optional[np.ndarray], 
        params: dict,
        logger: object,
    ):
    """
    Conditional stochastic primal dual (CSPD)
    :param get_stochastic_linsys: function that returns noisy [H,b] for solving Hx=b
    :param p_0: initial primal solution, and also the center
    :param y_0: initial dual solution (if None, then use zero)
    :param params: dictionary of parameters
    :param logger: object with `.log()` function to save progress
    :return p: estimated primal solution
    :return y: estimated dual solution
    """
    cspd_param_check(params)

    p = np.copy(p_0) 
    p_prev = np.copy(p)
    p_sol = np.zeros(len(p))
    y = y_0 if y_0 is not None else np.zeros(len(p))

    # for variance reduction
    if params.get("covariate", False):
        H_0 = params["H_0"]
        b_0 = params["b_0"]
        h_0_p = H_0@p_0 - b_0
        y_0 = H_0@p_0 - b_0 # initial residual 
        h_0_y = H_0.T@y_0

    k = params["total_iters"]
    for t in range(1, k+1):
        theta_t  = (t-1.0)/t
        eta_t    = params["eta"] * (t**0.5)
        lambda_t = params["lambda"] * (t**0.5)

        # primal extrapolation
        z = p + theta_t*(p-p_prev)
        p_prev = p

        # dual update
        (H,b) = get_stochastic_linsys()
        grad = -H@z + b # -H@p + b
        if params.get("covariate", False):
            grad += (H@p_0 - b) - h_0_p

        y = y - (1./lambda_t)*grad
        y /= la.norm(y)

        # primal update
        (H,_) = get_stochastic_linsys()
        grad = H.T@y
        if params.get("covariate", False):
            grad += -H.T@y_0 + h_0_y
        p = primal_update(p, p_0, grad, eta_t, params["D"])

        # weighted ergodic iterate
        theta = 2./(float(t)+1) # 1./float(t)
        p_sol = (1.-theta)*p_sol + theta*p

        logger.log(t, p_sol, label="cspd")

    return (p_sol, y)

def primal_update(x_curr, x_0, grad, eta, D):
    """ 
    (Projected) primal update. If projection's dual solution is too large, we
    print.

    :param x_curr: current solution (center of Bregman)
    :param x_0: center of feasible region
    :param grad: gradient of primal
    :param eta: step size
    :param D: diameter
    :return x: estimated primal update
    """
    niters_of_binary_search = 0
    lam = 0 # lagrange multiplier

    # unconstrained optimal solution
    x = x_curr - (1./eta)*grad

    # exponential search for Lagrange multiplier 
    while la.norm(x-x_0) > D:
        niters_of_binary_search += 1
        lam = max(1, 10*lam)
        x = (eta*x_curr + lam*x_0)/(eta+lam) - 1./(eta+lam)*grad

        if lam >= 1e8: # in case we do not terminate
            # TODO: Make sure this is logging
            # logging.warn("Unstable KKT condition, skipping primal update (consider terminating and debugging...)")
            return x_curr

    lo, hi = lam/2, lam
    if niters_of_binary_search > 0:
        niters_of_binary_search += 4
    # binary search for Lagrange multiplier (if needed)
    for _ in range(2*niters_of_binary_search):
        lam = (lo+hi)/2
        x = (eta*x_curr + lam*x_0)/(eta+lam) - 1./(eta+lam)*grad
        if la.norm(x-x_0) > D:
            lo = lam
        else:
            hi = lam

    lam = hi
    x = (eta*x_curr + lam*x_0)/(eta+lam) - 1./(eta+lam)*grad

    return x