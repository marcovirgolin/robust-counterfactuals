import numpy as np

from cogs.distance import *
from cogs.util import *



def gower_fitness_function(genes, x, blackbox, desired_class, feature_intervals, 
  indices_categorical_features=None, plausibility_constraints=None, 
  apply_fixes=False):

  is_single_candidate = len(genes.shape) == 1

  if is_single_candidate:
    genes = genes.reshape((1,-1))

  # assert plausibility constraints are held
  '''
  if plausibility_constraints is not None:
    for i, plau_c in enumerate(plausibility_constraints):
      # the value cannot change (e.g., not actionable feature)
      if is_single_candidate:
        to_consider_i = genes[i]
      else:
        to_consider_i = genes[:,i]

      if plau_c == '=':
        assert((to_consider_i == x[i]).all())
      # the value can only increase (e.g, age)
      elif plau_c == '>=':
        #genes[:,i] = np.where(genes[:,i] < x[i], x[i], genes[:,i])
        assert((to_consider_i >= x[i]).all())
      # the value can only decrease
      elif plau_c == '<=':
        #genes[:,i] = np.where(genes[:,i] > x[i], x[i], genes[:,i])
        assert((to_consider_i <= x[i]).all())

  # assert bounds are held
  mask_categorical = get_mask_categorical_features(len(x), indices_categorical_features)
  for i, is_c in enumerate(mask_categorical):
    if is_c:
      continue

    if is_single_candidate:
      to_consider_i = genes[i]
    else:
      to_consider_i = genes[:,i]
    
    low = feature_intervals[i][0]
    high = feature_intervals[i][1]
    assert((to_consider_i >= low).all())
    assert((to_consider_i <= high).all())
  '''
  
  if apply_fixes:
    genes = fix_features(genes, x, feature_intervals, 
      indices_categorical_features, plausibility_constraints)

  # compute distance, must be normalized to cap at 1
  num_feature_ranges = compute_ranges_numerical_features(feature_intervals, indices_categorical_features)
  gower_dist = gower_distance(genes, x, num_feature_ranges, indices_categorical_features)
  l_0 = 1/genes.shape[1] * np.sum(genes != x, axis=1)
  dist = .5*gower_dist + .5*l_0

  # then add additional penalty of +1 if the class is not the one that we want
  #if is_single_candidate:
  #  preds = blackbox.predict([genes])
  #else:
  preds = blackbox.predict(genes)
  failed_preds = preds != desired_class
  dist += failed_preds.astype(float) # add 1 if the class is incorrect
  # we maximize the fitness
  fitness_values = -dist

  if is_single_candidate:
    fitness_values = fitness_values[0]

  return fitness_values
