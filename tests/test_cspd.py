import unittest

import numpy as np
import numpy.linalg as la

from lqr import pe 
from lqr import lqr_logger

# run `python -m unittest discover tests` in parent directory

class TestCSPD(unittest.TestCase):

    def test_cspd_easy_linsys(self):
        """
        Tests simple Ax=b residual error when A and b have iid normal elements
        with CSPD.
        """
        n = 10
        x = np.arange(n)
        params = dict({
            "eta": 1,
            "lambda": 1, 
            "total_iters": 200, 
            "D": 4*la.norm(x),
        })

        logger = lqr_logger.Logger()

        for seed in range(1000, 1030):
            rng = np.random.default_rng(seed)

            A = rng.normal(scale=4.0, size=(n,n))
            b = A@x
            x_0 = np.zeros(n)

            get_stochastic_linsys = lambda : (A + rng.normal(scale=1.0, size=(n,n)), b + rng.normal(scale=1.0, size=n))

            (p,_) = pe.cspd(get_stochastic_linsys, x_0, None, params, logger)

            tol = la.norm(x)*la.cond(A)*la.norm(b)*2e-4

            self.assertLess(la.norm(p-x), tol) 

    def test_sme_cspd_easy_linsys(self):
        """
        Tests simple Ax=b residual error when A and b have iid normal elements
        using shrinking multi-epoch CSPD.
        """
        n = 10
        x = np.arange(n)
        params = dict({
            "eta": 1,
            "lambda": 1, 
            "total_iters": 200, 
            "D": 4*la.norm(x),
            "total_epochs": 3, 
        })

        logger = lqr_logger.Logger()

        for seed in range(1000, 1030):
            rng = np.random.default_rng(seed)

            A = rng.normal(scale=4.0, size=(n,n))
            b = A@x
            x_0 = np.zeros(n)

            get_stochastic_linsys = lambda : (A + rng.normal(scale=1.0, size=(n,n)), b + rng.normal(scale=1.0, size=n))

            (p,_) = pe.sme_cspd(get_stochastic_linsys, x_0, params, logger)

            tol = la.norm(x)*la.cond(A)*la.norm(b)*1e-4 # decrease error 2x

            self.assertLess(la.norm(p-x), tol) 
