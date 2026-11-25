# Navier-Stokes Flow Predictor

  

A neural surrogate that predicts 2D airflow around airfoils from an image of the shape, trained on the [AirfRANS](https://airfrans.readthedocs.io) benchmark (RANS solutions of the Navier-Stokes equations).



Instead of solving the equations numerically for every new shape, the model learns the mapping from shape to flow field and predicts it in milliseconds. A CFD solver provides the ground truth to compare against.



## Planned outputs



1. **Flow field:** pressure and velocity around the airfoil

2. **Shape parameters:** NACA camber, camber position and thickness read from the image

3. **Equation:** the NACA formula built from the predicted parameters



## Status



- [x] Training and evaluation pipeline (tested on synthetic data)

- [ ] Real AirfRANS data loader

- [ ] U-Net baseline trained on AirfRANS

- [ ] Comparison with a Fourier Neural Operator

- [ ] Shape-parameter and equation output

- [ ] Speed-up vs. CFD solver

- [ ] Generalization to unseen shapes and flow conditions

- [ ] Demo app



## Setup



Install the requirements:



    pip install -r requirements.txt



Download and convert the data (needs the airfrans package):



    python src/download.py
    python src/convert.py



Train and evaluate:



    python src/train.py --epochs 50
    python src/evaluate.py



## Results



| Model | Pressure error | Velocity error | Time per sample |
|-------|----------------|----------------|-----------------|
| U-Net | to be added | to be added | to be added |
| FNO | to be added | to be added | to be added |



Real results will be added after training on AirfRANS.



The figure below comes from **synthetic placeholder data** and only shows that the pipeline runs. It is not a real result.



![placeholder](results/comparison.png)



## Independent AirfRANS test protocol

For an unbiased final result, keep the official AirfRANS training and test splits separate. Do not use the official test split for epoch-by-epoch validation, hyperparameter selection, or model selection.

1. Download the official data if it is not already present:

       python src/download.py

2. Convert the two official splits into separate files:

       python src/convert_airfrans.py --root data/Dataset --out-dir data --grid 64

   This creates \`data/airfrans_train_g64.npz\` (official training split) and
   \`data/airfrans_test_g64.npz\` (official test split). The converter no longer
   combines them unless \`--combined-out\` is explicitly supplied.

3. Train a fresh model using only the official training split. The training
   script reserves 20% of those 800 training cases for validation (640 train,
   160 validation), and only evaluates on the official test split once training
   has finished:

       python src/train.py --model unet --data data/airfrans_train_g64.npz --test data/airfrans_test_g64.npz --epochs 100 --batch 32 --lr 0.001 --sched cosine --base 16 --name unet_airfrans_g64_independent

4. Independently reproduce/report the saved checkpoint's test metrics:

       python src/evaluate_airfrans_test.py --data data/airfrans_test_g64.npz --checkpoint results/unet_airfrans_g64_independent.pt

The evaluator reports mean, median and 90th-percentile per-simulation relative
L2 errors for u/U, v/U and Cp, plus inference time, and writes
\`results/airfrans_independent_test.csv\`.

**Important:** the earlier \`7.14%\` result came from a workflow that used the
official 200-case test split as validation during training. It should be
labelled a validation result, not an independent test result. The fresh
checkpoint and untouched official test metrics from this protocol are the
numbers to use for final generalization claims.
