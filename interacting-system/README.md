## Interacting particles

The DSM training is given in `train-DSM-interact-2particles.py`. This file runs DSM training, saves trained models, losses plot, samples with trained NNs and makes density plots after training; it also runs the numerical solution for the specifies problem and saves density plots/statistics plots. To run it run from the terminal:
```
run_dsm.sh
```
Feel free to play with hyperparameters (for example, running training for -n_epochs=10 epochs to see how it works, -invar=0 to use regular NN architecture).

To run PINN:

```
python interacting_PINN.py
```


We use qmsolve lib as a numerical solver, but it requires to change some constants in the library: open constants file in your system (for example, vim `env_python/lib/python3.8/site-packages/qmsolve/util/constants.py`), and change them as they're provided here (see file `constants.py` in this repo). You'd need to restart notebook after this change.
