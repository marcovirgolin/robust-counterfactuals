import pandas as pd, numpy as np
import os
from cogs.distance import gower_distance
from cogs import util
from robust_cfe.dataproc import gimme

"""
This script goes through all "result_fold_x.csv" files in the sub-folders of "result/" and:
- appends the Gower distance between x and z
- appends a sparsity score (% features that remained identical)
- appends the type of blackbox (if not already present)
"""

''' Load datasets (they include meta-info) '''
datasets = dict()
for dataset_name in ['credit','adult','boston','garments','compas']:
  datasets[dataset_name] = gimme(dataset_name, datasets_folder="datasets")
  print(dataset_name, datasets[dataset_name]['X'].shape)



# read all results
for f in os.listdir("results"):

  # get folder that applies
  if not f.startswith("dataset_"):
    continue
  
  # get all log files in that folder
  for r in os.listdir(os.path.join("results",f)):

    filepath = os.path.join("results",f,r)

    # if it is an old ".postproc.", delete it
    if ".postproc." in r:
      os.remove(filepath)
      continue
    
    df = pd.read_csv(filepath)
    
    # append blackbox info
    if "blackbox" not in df.columns:
      blackbox = f.split("blackbox_")[1].split("_")[0]
      df["blackbox"] = blackbox

    df["gower_dist"] = 0
    df["loss"] = 0

    # add gower dist & loss
    for i, row in df.iterrows():
      z = np.array(eval(row['z']))
      x = np.array(eval(row['x']))
      dataset_name = row.dataset
      feature_intervals = datasets[dataset_name]['feature_intervals']
      indices_categorical_features = datasets[dataset_name]['indices_categorical_features']
      num_feature_ranges = util.compute_ranges_numerical_features(feature_intervals, indices_categorical_features)
      gd = gower_distance(z, x, num_feature_ranges, indices_categorical_features)
      df.loc[i, 'gower_dist'] = gd
      l_0 = 1/len(x) * np.sum(z != x)
      is_not_cfe = 0 if row['pred_class_z'] == row['desired_class'] else 1
      loss = .5*gd + .5*l_0 + is_not_cfe
      df.loc[i, "sparsity"] = 1.0 - l_0
      df.loc[i, 'loss'] = loss

    # save df
    df.to_csv(filepath, index=False)