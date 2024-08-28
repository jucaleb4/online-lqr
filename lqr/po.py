from lqr import pe

import logging

import numpy as np
import numpy.linalg as la

# Policy Optimization
def npg(K_0, env, params, logger):
    """
    Natural policy gradient (policy optimization).

    :param K_0: initial matrix gain (assumed stable)
    :param env: environemtn to simulate
    :param params: dictionary of parameters
    :param logger: Logger file
    :return K: final matrix gain
    """
    K = K_0
    eta = params["po_eta"]

    rho = env.get_spectrum(K)
    logger.log(0, K, label="npg", rho=rho)
    for t in range(1, 1+params["po_total_iters"]):

        (J_K, E_K, p) = pe.policy_eval(K, env, params, logger) 

        new_K = K - 2*eta * E_K

        if np.all(np.isfinite(new_K)) and la.norm(new_K) < 1e8:
            K = new_K

        # (_, E_K_star) = pe.exact_policy_eval(K, env) 
        # print(f"E_K=\n{E_K}\nE_K_star=\n{E_K_star}\n===================")

        rho = env.get_spectrum(K)
        logger.log(t, K, label="npg", rho=rho)

        if params.get("dynamic_D", False):
            params["D"] = min(max(1, la.norm(p)), params["D"])
            # params["D"] = params["D"] * t/(t+1.)
        # print(f"D={params['D']}")

        params["minibatch"] = min(params["minibatch"]+1, 30)

    return K

def tts_actor_critic(K_0, env, params, logger):
    """
    Two-time scale actor-critic method (https://arxiv.org/pdf/2109.14756)
    """
    K = K_0
    (n,m) = env.get_system_dimension()
    J_k   = 0
    # TODO: Is this correct?
    Phi_k = np.zeros((n+m,n+m), dtype=float)

    # See Theorem 2 from 
    a = params['a_power'] # 1 or 3./5
    b = params['b_power'] # 2./3 or 2./5

    for k in range(params["total_iters"]):

        alpha_k = params['alpha_0']/((k+1)**a)
        beta_k  = params['beta_0']/((k+1)**b)

        (x_k, u_k, cost_k) = env._step(K)
        z_k = np.append(x_k, u_k)

        # actor step
        Phi_k_21 = Phi_k[n:,:n]
        Phi_k_22 = Phi_k[n:,n:]
        new_K = K - alpha_k*(Phi_k_22@K - Phi_k_21)
        if np.all(np.isfinite(new_K)) and la.norm(new_K) < 1e10:
            K = new_K

        # critic step 
        J_k = J_k - beta_k*(J_k-cost_k)
        Phi_k = Phi_k - beta_k*np.outer(z_k, np.dot(z_k, z_k)*Phi_k@z_k + z_k*(J_k-cost_k))

        # Log every n_log steps
        if "log_n_iter" in params and (k % params["log_n_iter"] == 0):
            t = int(k/params["log_n_iter"])
            logger.log(t, K, label="npg", rho=env.get_spectrum(K))

    return K
