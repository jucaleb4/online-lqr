import numpy as np
import numpy.linalg as la
import scipy.linalg as scla

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

    # print(f"Initial spectrum: {env.get_spectrum(K_0)}")
    return (env, K_0)

def setup_medium_simple_env(seed):
    return _setup_simple_env(seed, 10)

def setup_large_simple_env(seed):
    return _setup_simple_env(seed, 100)

def _setup_simple_env(seed, n):
    """ Borrwed from 'On the sample complexity of the linear quadratic regulator'
    by Dean, Mania, Matni, Recht, and Tu.
    """
    n = k = 10 
    col = np.append([1.01, 0.01], np.zeros(n-2))
    A = scla.toeplitz(col, col)
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

    # print(f"Initial spectrum: {env.get_spectrum(K_0)}")
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

    # print(f"Initial spectrum: {env.get_spectrum(K_0)}")
    return (env, K_0)

def setup_boeing_env(seed):
    """ Control of Boeing 747 """
    A = np.array([
        [0, 0, -1, 0, 0],
        [0, -0.003, 0.039, 0, -0.322],
        [0,-0.065,-0.319, 7.74, 0],
        [0, 0.020,-0.101,-0.429, 0],
        [0,0,0,1,0],
    ])
    B = np.array([
        [0,0],
        [0.01,1],
        [-0.18,-0.04],
        [-1.16,0.598],
        [0,0],
    ])

    A = np.array([
        [1,-1.1267,-0.6528,-8.0749, 1.5890],
        [0, 0.7741, 0.3176,-0.9772,-2.9690],
        [0, 0.1157, 0.0201,-0.0005,-0.3628],
        [0, 0.0111, 0.0033,-0.0349,-0.0447],
        [0, 0.1388,-0.0862, 0.2935, 0.7579],
    ])
    # print(f"rho(A)={np.abs(la.eig(A)[0])}")
    B = np.array([
        [89.1973,-50.1685, 1.1267,-19.3472],
        [ 5.2231,  6.3614, 0.2259, -0.3176],
        [-9.4731,  5.9294,-0.1157,  0.9799],
        [-0.3236,  0.3178,-0.0111, -0.0033],
        [-4.5318,  3.2146,-0.1388,  0.0862],
    ])

    (n,k) = B.shape
    Q = np.eye(n)
    R = np.eye(k)
    # Cov = 0.1 * np.eye(n)
    U = np.array([
        [1, -0.01, 0.5, -0.5, -0.5],
        [0, 1, 0.1, -0.01, -0.01],
        [0, 0, 1, -0.5, -0.5],
        [0, 0, 0, 1, 0.5],
        [0,0,0,0,1]
    ])
    Cov = U@U.T
    sigma = 1

    K_0 = 0.005*np.ones(shape=(k,n))
    env = lqr_env.LQREnv(A, B, Q, R, Cov, sigma, seed=seed)
    
    # print(f"Initial spectrum: {env.get_spectrum(K_0)}")
    return (env, K_0)
