# online-lqr
An actor-critic approach to solving LQR with only a single trajectory.

## How to run
The code requires minimal installation, e.g., NumPy, SciPy, and argparse. To quickstart:
```
python run.py 
```
which runs the simple (i.e., synthetic) environment once.

For a more specific experimental setup, you can pass in arguments:
```
python run.py --env simple --num_runs 32
```
Here, `simple` is the synthetic environment. 
Change it to `cartpole` to get the inverted pendulum. 
Input `32` is the number of trials to run, where all the performance is logged and saved into the provided `/logs` directory.

The algorithm already contains the tuned parameters. 
However, if you want to modify it, you can finetune with
```
python run.py --env simple --tune stepsize
```
which tunes the critic's stepsize parameters. 
You can also tune the critic's `iter` as well as the actor's `po_stepsize`.
This process is noisy and still rudimentary, so make sure to run multiple trials and make a judgement yourself.

## Testing (WIP)
We have some basic tests:
```
python -m unittest discover tests
```

## Acknowledgements
This code is freely available for everyone to use. 
If you would like to acknowledge the usage of this code, please cite the following paper:

Ju, Caleb, Georgios Kotsalis, and Guanghui Lan. "A model-free first-order method for linear quadratic regulator with $\tilde {O}(1/\varepsilon) $ sampling complexity." arXiv preprint arXiv:2212.00084 (2022).
