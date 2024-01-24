import numpy as np
import numpy.linalg as la

from lqr import lqr_env

def setup_simple_env(seed):
    """ Borrwed from 'On the sample complexity of the linear quadratic regulator'
    by Dean, Mania, Matni, Recht, and Tu.
    """
    n = k = 3
    A = np.array([
        [1.01, 0.01, 0.00],
        [0.01, 1.01, 0.01],
        [0.00, 0.01, 1.01],
    ])
    B = np.eye(k)
    Q = 1e-3 * np.eye(n)
    R = np.eye(k)
    Cov = np.eye(n)
    sigma = 1

    env = lqr_env.LQREnv(A, B, Q, R, Cov, sigma, seed=seed)
    K_0 = np.eye(k)
    # K_0 = np.array([
    #     [0.04458641, 0.01197829, 0.00123005],
    #     [0.01197829, 0.04581646, 0.01197829],
    #     [0.00123005, 0.01197829, 0.04458641]
    # ])

    print(f"Initial spectrum: {la.norm(A-B@K_0, ord=2)}")
    return (env, K_0)

def setup_cartpole_env(seed):
    """ Borrowed from 'A Comparison of LQR and MPC Control Algorithms of an
    Inverted Pendulum' by Jezierski, Mozaryn, and Suski.
    We scale the system dynamic matrices A,B by a factor of 0.8 to ensure
    stability.
    """
    A = np.array([
        [1., 0.0099, 2e-5, 9e-8],
        [0, 0.9840, 0.0052, 2e-5],
        [0, -0.0002, 1.0011, 0.0100],
        [0, -0.0329, 0.2120, 1.0011]
    ])
    A[1:,1:] *= 0.8
    B = 0.8*np.array([[0, 0.0246, 0.0003, 0.0506]]).T
    (n,k) = B.shape
    Q = np.eye(n)
    R = np.eye(k)
    U = np.array([[1, 0.5, 0.01, 0.01],[0, 1, 0.75, 0.25], [0, 0, 1, 0.5], [0,0,0,1]])
    Cov = U.T@U
    sigma = 1

    K_0 = np.array([[1,1,1,1]])
    env = lqr_env.LQREnv(A, B, Q, R, Cov, sigma, seed=seed)

    print(f"Initial spectrum: {la.norm(A-B@K_0, ord=2)}")
    return (env, K_0)
