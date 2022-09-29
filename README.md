# Robust Counterfactual Explanations
Counterfactual explanations may be invalidated or require more intervention than originally expected, if adverse events take place (e.g., genetic predisposition to resist a treatment).
Can we search for counterfactual explanations that are *robust* to such unfortunate circumstances?
What is the extra cost that robust counterfactual explanations entail?
Is this cost justified?

Our research paper attempts to answer these questions:
```
@misc{virgolin2022robustness,
      title={On the Robustness of Sparse Counterfactual Explanations to Adverse Perturbations}, 
      author={Marco Virgolin and Saverio Fracaros},
      year={2022},
      eprint={2201.09051},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      howpublished={\url{https://arxiv.org/abs/2201.09051}},
}
```

## Examples of perturbations
Besides code to run experiments, we implemented possible adverse perturbations due to unfortunate events for five data sets that are commonly used in the literature.
You can see our choices in [robust_cfe/dataproc.py](robust_cfe/dataproc.py).

Here's a snippet as an example for the data set Credit (also with plausibility constraints):

```python
# can get older, say up to half-a-year of delay in the assessment
# cannot get younger
perturb['age'] = {'type':'absolute', 'increase':0.5, 'decrease':0}   
plausib['age'] = '>='

# how long one has been at the current residence
# cannot vary without our control (sure one could be evicted but let's assume that's extremely rare)
perturb['present_res_since'] = None 
# this can grow or drop (e.g., to 0) if one relocates; for simplicity let's just assume that one does not change residence (hence >=)
plausib['present_res_since'] = '>=' 

# duration of the credit
# assume perturbations can change it a bit relative to what was asked for
# (but less likely to decrease than increase)
perturb['duration_in_month'] = {'type':'relative', 'increase':0.25, 'decrease':0.05} 
plausib['duration_in_month'] = None 
```


## Installation
Run `pip install .` to install this repo. 
To install the search algorithms CoGS, LORE, and Growing Spheres, do:
```
cd methods; for m in *; do cd $m; pip install .; cd ../; done; cd ../;
```
(the latest version of CoGS can be found at [https://github.com/marcovirgolin/CoGS](https://github.com/marcovirgolin/CoGS))

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
If you wish to include your own counterfactual search algorithm, you must create a wrapper for it. 
This can be done by editing the file `robust_cfe/wrappers.py` similarly to how we implemented wrappers for CoGS, LORE, GrowingSpheres, and SciPy's Nelder-Mead.

## Creating your own data set
Upload the data of interest in the folder `datasets`.
Next, edit the file `robust_cfe/dataproc.py` to include the pre-processing for your data set, which should be similar to the one we provide for the existing data sets.
If you wish, you can define plausibility constraints and possible perturbations, like we did.
