from typing import Optional

import logging

import numpy as np
import numpy.linalg as la
import scipy.linalg as spla

from lqr import tools

logging.basicConfig(level=logging.DEBUG)

"""
Environment class
"""

class LQREnv():
    def __init__(self, A, B, Q, R, Cov, sigma, x_0:Optional[np.ndarray]=None, seed:Optional[int]=None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Cov = Cov
        self.sigma = sigma
        self.n = B.shape[0]
        self.k = B.shape[1]

        self.zero_n = np.zeros(self.n)
        self.zero_k = np.zeros(self.k)
        self.I_k = np.eye(self.k)

        # matrices for linear system (reduce memory footprint since Python passes by reference)
        m = self.n+self.k
        nn = int(m*(m+1)/2)
        self.H = np.zeros((nn+1,nn+1), dtype=float)
        self.b = np.zeros(nn+1, dtype=float)

        self.rng = np.random.default_rng(seed)

        if x_0 is None:
            self.x = self.rng.normal(loc=0, scale=1.0, size=self.n) # (# np.zeros(self.n, dtype=float)
        else:
            self.x = x_0

        self.sample_ct = 0

    def get_system_parameters(self):
        """ Return all parameters for the system """
        return (self.A, self.B, self.Q, self.R, self.Cov, self.sigma)

    def get_system_dimension(self):
        """ Return dimension of state and action, respectively """
        return (self.n, self.k)

    @property
    def num_samples(self):
        return self.sample_ct

    @property
    def state(self):
        return self.x

    def step(self, K):
        """ Simulates a step in the environment (hides action). See `_step` for description """
        (x_next, _, cost) = self._step(K)
        return (x_next, cost)

    def get_spectrum(self, K):
        return np.max(np.abs(la.eig(self.A-self.B@K)[0]))

    def _step(self, K):
        """ Simulates a step in the environment.

        :param K: matrix gain
        :return x_next: next state
        :return u: action
        :return cost: simulated cost
        """
        (A, B, Q, R, Cov, sigma, x) = self.A, self.B, self.Q, self.R, self.Cov, self.sigma, self.x
        (zero_k, zero_n, I_k) = self.zero_k, self.zero_n, self.I_k

        # controller 
        v = self.rng.multivariate_normal(mean=zero_k, cov=sigma**2 * I_k)
        u = -K@x + v

        # cost
        cost = np.dot(x, np.dot(Q,x)) + np.dot(u, np.dot(R,u))

        # next state 
        w = self.rng.multivariate_normal(mean=zero_n, cov=Cov)
        x_next = A@x + B@u + w
        self.x = x_next

        self.sample_ct += 1
        return (x_next, u, cost)

    def reset(self):
        """ Reset the state and sample count back to zero """
        self.sample_ct = 0
        self.x = np.zeros(self.n)

    def form_stochastic_Hb(self, K, tau):
        """ 
        Estimates H,b, first by mixing `tau-1` times and selecting the last two samples to form H,b

        :param K: matrix gain
        :param tau: number of mixes
        :return H,b: matrix vector pair to solve
        """
        for _ in range(tau-1):
            self.step(K)

        x_1 = self.state
        (x_2, u_1, cost_1) = self._step(K)
        (_,   u_2, _)      = self._step(K)

        phi_vec_1 = np.append(x_1, u_1)
        phi_vec_2 = np.append(x_2, u_2)
        phi_1 = tools.svec(np.outer(phi_vec_1, phi_vec_1))
        phi_2 = tools.svec(np.outer(phi_vec_2, phi_vec_2))

        self.H[1:,0] = phi_1
        self.H[1:,1:] = np.outer(phi_1, phi_1-phi_2) 
        self.b[0] = cost_1
        self.b[1:] = cost_1 * phi_1

        return (self.H, self.b)

    def get_optimal_val(self):
        """ Get J(K*) """
        P_star = spla.solve_discrete_are(self.A, self.B, self.Q, self.R)
        Cov2 = self.Cov + self.sigma**2 * self.B@self.B.T
        J_star = np.trace((P_star@Cov2))
        return J_star
