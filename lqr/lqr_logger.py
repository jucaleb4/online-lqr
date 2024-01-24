from abc import ABC

import logging

import numpy as np
import numpy.linalg as la

class Logger(ABC):
    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def save(self):
        raise NotImplemented

class TestLogger(Logger):
    def __init__(self, A, x_sol, b):
        super().__init__()
        self.A = A
        self.b = b
        self.x_sol = x_sol

        print("TestLogger turning logging on")
        logging.basicConfig(level=logging.DEBUG)

    def log(self, t, x, label):
        if label=="cspd":
            logging.debug(f"[{t}]: res={la.norm(self.A@x-self.b): .2e} x={x}, x*={self.x_sol}")

class SimpleRunLogger(Logger):
    def __init__(self, env, fname=None):
        super().__init__()
        self.env = env
        self.J_star = env.get_optimal_val()
        self.fname = fname

        self.num_logs = 0
        self.sample_arr = np.zeros(32, dtype=int)
        self.Jgap_arr = np.zeros(32, dtype=float)
        self.J_arr = np.zeros(32, dtype=float)
        self.rho_arr = np.zeros(32, dtype=float)

        print("SimpleRunLogger turning logging on")
        logging.basicConfig(level=logging.DEBUG)

    def log(self, t, K, label, **kwargs):
        if label=="npg":
            from lqr import pe
            # TODO: Get rid of pe
            (J_K, _) = pe.exact_policy_eval(K, self.env)
            num_samples = self.env.num_samples

            logging.debug(f"[{t}]: J(K)-J*={J_K-self.J_star}, J(K)={J_K}, num_samples={num_samples}")

            t = self.num_logs
            self.sample_arr[t] = num_samples
            self.Jgap_arr[t] = J_K-self.J_star
            self.J_arr[t] = J_K
            for key, value in kwargs.items():
                if key == "rho":
                    self.rho_arr[t] = value

            self.num_logs += 1
            m = len(self.sample_arr)
            if self.num_logs == m:
                self.sample_arr = np.append(self.sample_arr, np.zeros(m, dtype=int))
                self.Jgap_arr = np.append(self.Jgap_arr, np.zeros(m, dtype=float))
                self.J_arr = np.append(self.J_arr, np.zeros(m, dtype=float))
                self.rho_arr = np.append(self.rho_arr, np.zeros(m, dtype=float))

    def save(self):
        if self.fname is None:
            return

        with open(self.fname, "w") as fp:
            fp.write("iter,samples,Jgap,J_K,rho\n")
            for t in range(self.num_logs):
                fp.write(f"{t+1},{self.sample_arr[t]},{self.Jgap_arr[t]},{self.J_arr[t]},{self.rho_arr[t]}\n")
