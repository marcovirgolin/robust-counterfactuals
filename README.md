# Robust Counterfactual Explanations
This repository comes with the paper 
"On the Robustness of Counterfactual Explanations to Adverse Perturbations".

## Installation
Run `pip install .` to install this repo. 
To install the search algorithms CoGS, LORE, and Growing Spheres, do:
```
cd methods; for m in *; do cd $m; pip install .; cd ../; done; cd ../;
```

## Running 
The file `run.py` can be used to search for counterfactual examples $\mathbf{z}$ in a dataset.
For example, 
```
python run.py --method cogs --dataset garments --check_plausibility 1 --optimize_C_robust 0 --optimize_K_robust 0 --n_jobs 16 --n_reps 5
```
will use CoGS to search for counterfactual examples in the data set Garments (called *Productivity* in the paper) while respecting the plausibility constraints that are defined for that data set.

## Reproducing the results of the paper
The notebook to reproduce all figures and tables in the paper is `notebooks/reproduce.ipynb`. Furthermore, some meta-info on the data sets and annotations is provided in `notebooks/dataset_info.ipynb`.
Note that the logs in the folder `results` require GIT-LFS to be obtained.

## Including your own counterfactual search algorithm
If you with to include your own counterfactual search algorithm, you must create a wrapper for it. 
This can be done by editing the file `robust_cfe/wrappers.py` similarly to how we implemented wrappers for CoGS, LORE, GrowingSpheres, and SciPy's Nelder-Mead.

## Creating your own data set
Upload the data of interest in the folder `datasets`.
Next, edit the file `robust_cfe/dataproc.py` to include the pre-processing for your data set, which should be similar to the one we provide for the existing data sets.
If you wish, you can define plausibility constraints and possible perturbations, like we did.