# online-lqr
An actor-critic approach to solving LQR with only a single trajectory.

## How to run
The code requires minimal installation, e.g., NumPy, SciPy, and argparse. To quickstart:
```
python run.py --dynamic
```
which runs the simple (i.e., synthetic) environment once with dynamic diameter (for best performance).

For a more specific experimental setup, you can pass in arguments:
```
python run.py --env simple --num_runs 32 --dynamic --seed 1000
```
Here, `simple` is the synthetic environment. 
Change it to `boeing` to run the control of the longitudinal dynamics of a Boeing 747 aircraft (remove the `dynamic` flag for optimal performance).
Input `32` is the number of trials to run, where all the performance is logged and saved into the `/logs` directory (make sure this exists before running). In our experiments, we start with seed 1000 (i.e., `--seed 1000`) since we used smaller seeds for tuning and wanted to avoid overfitting to the seed.

The algorithm already contains the tuned parameters. 
However, if you want to modify it, you can finetune with
```
python run.py --env simple --tune stepsize
```
which tunes the critic's stepsize parameters. 
You can also tune the critic's `iter` as well as the actor's `po_stepsize`.
This process is noisy and still rudimentary, so make sure to run multiple trials and make a judgement yourself.

## Code structure
For those who want to play with the optimization or models, head to the `/lqr` folder.

Codes pertaining to algorithms can be found in `po.py` (policy optimization) and `pe.py` (policy evaluation).

Examples of our model can be found in `example_envs.py`, which calls `lqr_envs.py` under the hood.

The remaining files are either for logging, linear algebra, or tuning.

## Testing (WIP)
We have some basic tests:
```
python -m unittest discover tests
```

## Acknowledgements
This code is freely available for everyone to use. 
If you would like to acknowledge the usage of this code, please cite the following paper:

Ju, Caleb, Georgios Kotsalis, and Guanghui Lan. "A model-free first-order method for linear quadratic regulator with $\tilde {O}(1/\varepsilon) $ sampling complexity." arXiv preprint arXiv:2212.00084 (2022).
