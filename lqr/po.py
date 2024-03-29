from lqr import pe

import logging

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

        K = K - 2*eta * E_K
        print(f"Estimated cost J(K)={J_K}")

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
