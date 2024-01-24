import numpy.linalg as la

from run import setup_cartpole_env

(env, K_0) = setup_cartpole_env(seed=0)

(A, B, _, _, _, _) = env.get_system_parameters()
