import numpy as np
from cogs.fitness import gower_fitness_function
from cogs.util import *


def is_same_point(z_1, z_2, x, feature_intervals, indices_categorical_features=None, tol=5e-2):
  
  # first check that the features acted upon are the same
  acted_upon_1 = z_1 != x
  acted_upon_2 = z_2 != x
  if (acted_upon_1 != acted_upon_2).any():
    return False
  
  # Now check feature values (tol is used for numerical features)
  for i in range(len(x)):
    if i in indices_categorical_features:
      # category must match
      if z_1[i] != z_2[i]: 
        return False
    else:
      # numerical value must be close enough
      range_i = feature_intervals[i][1] - feature_intervals[i][0]
      abs_norm_diff_i = np.abs(z_1[i] - z_2[i]) / range_i
      if abs_norm_diff_i > tol:
        return False

  return True


def generate_K_neighborhood(Z, x, perturb, feature_intervals, indices_categorical_features, n_samples):

  is_single_candidate = len(Z.shape) < 2
  if is_single_candidate:
    Z = Z.reshape((1,-1))
    
  Z_neighbors = np.repeat(Z, repeats=n_samples, axis=0)

  # Mask based on features to keep idle
  feature_to_consider_mask = np.where(Z_neighbors == x, True, False)

  # Populate random matrix of displacements
  for i in range(Z.shape[1]):
    if perturb[i] is None:
      continue
    # categorical feature perturbations
    if i in indices_categorical_features:
      # only done for absolute categorical
      assert(perturb[i]['type']=='absolute')
      random_categories_i = np.random.choice(perturb[i]['categories'], size=len(Z_neighbors))
      Z_neighbors[:,i] = np.where(feature_to_consider_mask[:,i], random_categories_i, Z_neighbors[:,i])
    else:
      if perturb[i]['type']=='absolute':
        perturbed_i = Z_neighbors[:,i] + np.random.uniform(low=-perturb[i]['decrease'], high=perturb[i]['increase'], size=len(Z_neighbors))
      else:
        perturbed_i = Z_neighbors[:,i] + Z_neighbors[:,i]*np.random.uniform(low=-perturb[i]['decrease'], high=perturb[i]['increase'], size=len(Z_neighbors))
      # fix out of bounds
      perturbed_i = np.where(perturbed_i < feature_intervals[i][0], feature_intervals[i][0], perturbed_i)
      perturbed_i = np.where(perturbed_i > feature_intervals[i][1], feature_intervals[i][1], perturbed_i)
      Z_neighbors[:,i] = np.where(feature_to_consider_mask[:,i], perturbed_i, Z_neighbors[:,i])

  return Z_neighbors


def compute_K_robustness_score(Z, x, perturb, blackbox, feature_intervals, indices_categorical_features, n_samples=1000):
  is_single_candidate = len(Z.shape) < 2
  if is_single_candidate:
    Z = Z.reshape((1,-1))
    
  # generate random neighborhood
  Z_neighbors = generate_K_neighborhood(Z, x, perturb, feature_intervals, indices_categorical_features, n_samples)

  # compute predictions
  p_z = blackbox.predict(Z)
  p_z_neighbors = blackbox.predict(Z_neighbors)
  
  p_z_neighbors = p_z_neighbors.reshape((len(Z),n_samples)).T
  
  # compute scores for each point in Z
  score = np.sum(np.equal(p_z,p_z_neighbors),axis=0)/n_samples

  if is_single_candidate:
    score = score[0]
  
  return score

  

def compute_worst_C_setbacks(Z, x, perturb, indices_categorical_features=None):
  is_single_candidate = len(Z.shape) == 1

  if is_single_candidate:
    Z = Z.reshape((1,-1))

  setbacks = np.zeros(Z.shape)

  # compute signs as per definition of worst-case in the paper
  signs = np.sign(Z-x)
  for i in range(len(x)):
    
    # skip perturbations that cannot happen
    if perturb[i] is None:
      continue

    if indices_categorical_features is not None and i in indices_categorical_features:
      # for gower it does not matter for what the mismatch is
      # let's set a magnitude of setback of 1 (category mismatch)
      setbacks[:,i] = np.where(signs[:,i] != 0, 1, setbacks[:,i])
    else:
      # numerical feature

      # bound as per definition
      oppo_sign_diff = -(Z[:,i] - x[i])

      if perturb[i]['type'] == 'absolute':
        neg_perturb = -perturb[i]['decrease']
        pos_perturb = perturb[i]['increase']
      else: # relative
        neg_perturb = -perturb[i]['decrease'] * Z[:,i]
        pos_perturb = perturb[i]['increase'] * Z[:,i]

      min_neg_perturbation = np.where(neg_perturb > oppo_sign_diff, neg_perturb, oppo_sign_diff)
      max_pos_perturbation = np.where(pos_perturb < oppo_sign_diff, pos_perturb, oppo_sign_diff)
      
      # we must apply the minimal negative perturbation if sign(z-x)>0
      # and the maximal positive perturbation if sign(z-x) < 0
      setbacks[:,i] = np.where(signs[:,i] > 0, min_neg_perturbation, setbacks[:,i])
      setbacks[:,i] = np.where(signs[:,i] < 0, max_pos_perturbation, setbacks[:,i])
      # note: no changes are made for the case signs==0 (as the feature is not in C)

  if is_single_candidate:
    setbacks = setbacks[0]
  
  return setbacks



def compute_fitness_contribution_C_setbacks(Z, x, perturb, indices_categorical_features=None, norm='L1'):
  # now compute setbacks
  setbacks = compute_worst_C_setbacks(Z, x, perturb, indices_categorical_features)
   
  if norm == 'L1':
    if len(setbacks.shape) > 1:
      contribution = np.sum(np.abs(setbacks), axis=1)
    else:
      contribution = np.sum(np.abs(setbacks))
  else:
    raise ValueError("Unrecognized norm",norm,"to compute the contribution of worst-case C-setbacks")

  return contribution


def gower_fitness_function_with_worst_C_setbacks(Z, x, blackbox, desired_class, perturb, feature_intervals, 
  indices_categorical_features=None, plausibility_constraints=None, apply_fixes=False):

  # compute normal fitness values
  fitness_values = gower_fitness_function(Z, x, blackbox, desired_class, feature_intervals, 
    indices_categorical_features, plausibility_constraints, apply_fixes)

  contribution_C_setbacks = compute_fitness_contribution_C_setbacks(Z, x, perturb, indices_categorical_features)

  # account under the way gower distance is used in coputing the gower_fitness_function
  result = fitness_values - .5*contribution_C_setbacks/len(x)

  return result


def compute_fitness_contribution_K_scores(Z, x, blackbox, perturb, feature_intervals, indices_categorical_features=None, n_samples_b_robust=100):
  k_robust_scores = compute_K_robustness_score(Z, x, perturb, blackbox, feature_intervals, indices_categorical_features, n_samples=n_samples_b_robust)
  k_nonrobust_scores = 1-k_robust_scores
  return k_nonrobust_scores


def gower_fitness_function_with_K_robustness_scores(Z, x, blackbox, desired_class, perturb, feature_intervals, 
  indices_categorical_features=None, plausibility_constraints=None, n_samples_k_robust=100):

  # compute normal fitness values
  fitness_values = gower_fitness_function(Z, x, blackbox, desired_class, feature_intervals, 
    indices_categorical_features, plausibility_constraints)

  contrib_K_nonrobust_scores = compute_fitness_contribution_K_scores(Z, x, blackbox, perturb, feature_intervals, indices_categorical_features, n_samples_k_robust)

  # account for this, conmensurate to gower_fitness_function
  result = fitness_values - .5*contrib_K_nonrobust_scores

  return result


def gower_fitness_function_with_CK_robustness(Z, x, blackbox, desired_class, perturb, feature_intervals, 
  indices_categorical_features=None, plausibility_constraints=None, n_samples_k_robust=100, apply_fixes=False):  

  fitness_values = gower_fitness_function(Z, x, blackbox, desired_class, feature_intervals, 
    indices_categorical_features, plausibility_constraints, apply_fixes)

  contribution_C_setbacks = compute_fitness_contribution_C_setbacks(Z, x, perturb, indices_categorical_features)
  contribution_K_nonrobust_scores = compute_fitness_contribution_K_scores(Z, x, blackbox, perturb, feature_intervals, indices_categorical_features, n_samples_k_robust)

  # like above
  result = fitness_values - .5*contribution_C_setbacks/len(x) - .5* contribution_K_nonrobust_scores

  return result