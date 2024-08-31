# online-lqr
An actor-critic approach to solving LQR with only a single trajectory.

## Requirements
You will need NumPy, SciPy, and argparse. You may also need matplotlib for plotting.

## How to run

## TLDR
If you want to recreate the all the results from the paper, *first* create a folder called `logs`, which is where the code saves the progress.

Then, you can run the following scripts:
```
python run.py --env simple --num_runs 32 --dynamic --seed 1000 --alg npg_rst
python run.py --env simple --num_runs 32 --dynamic --seed 1000 --alg npg
python run.py --env simple --num_runs 32 --dynamic --seed 1000 --alg tts_ac

python run.py --env large_simple --num_runs 32 --dynamic --seed 1000 --alg npg_rst
python run.py --env large_simple --num_runs 32 --dynamic --seed 1000 --alg npg
python run.py --env large_simple --num_runs 32 --dynamic --seed 1000 --alg tts_ac

python run.py --env boeing --num_runs 32 --seed 1000 --alg npg_rst
python run.py --env boeing --num_runs 32 --seed 1000 --alg npg
python run.py --env boeing --num_runs 32 --seed 1000 --alg tts_ac
```
We will provide more details about the arguments in the "More detail" section below.

The code will run 32 trials of the experiements starting with seed=1000 using the three algorithms:
- npg_rst: NPG with restart (i.e., multi-epoch).
- npg: NPG with only a single epoch
- tts_ac: Two-time scale actor-critic ([link](https://epubs.siam.org/doi/abs/10.1137/22M150277X))

Our code can also run each experiement in parallel. Just add a `--parallel` flag, i.e., 
```
python run.py --env simple --num_runs 32 --dynamic --seed 1000 --alg npg_rst --parallel
```

## How to parse
After running all nine scripts above, you can print the results by
```
plot_240829.py
```
Make sure to update the `root` variable inside the file.

The file will output a file `all_perfs.png`. 

## More details
You may notice we only use flag `--dynamic` for both simple environments (but not Boeing). 
When tuning the algorithm, we found the performance of policy evaluation for the two environments was sensitive to the diameter of the feasible region, `D_0`.
If it was too large, progress would be slow, while if it was too small, then the optimal solution may not be found.
To mitigate this, the `--dynamic` flag dynamically sets `D_0` to the norm of the solution from the previous call of policy evaluation.

We have code for tuning, but unfortunately it was made ad-hoc and not well-structured, so we will not provide extensive documentation for it. 
If you still want to play around, you can try:
```
python run.py --tune <tune_setting>
```
where `tune_setting` can either be:
- stepsize (tunes policy evaluation's stepsize) 
- iter (tunes number of iterations and epochs for policy evaluation)
- po_stepsize (tunes policy optimizaiton's stepsize)
- tts_ac (tunes stepsize for two-time scale actor-critic)

## Code structure
For those who want to play with the optimization or models, head to the `/lqr` folder.

The main file `run.py` is the place we input the tuning parameters.

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
