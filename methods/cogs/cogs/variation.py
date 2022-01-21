import numpy as np
from cogs.util import *

def crossover(genes):

  offspring = genes.copy()
  parents_1 = np.vstack((genes[:len(genes)//2], genes[:len(genes)//2]))
  parents_2 = np.vstack((genes[len(genes)//2:], genes[len(genes)//2:]))
  mask_cross = np.random.choice([True, False], size=genes.shape)
  offspring = np.where(mask_cross, parents_1, parents_2)

  return offspring


# linear crossover does not work well under L0!
def linear_crossover(genes, indices_categorical_features=None):
  offspring = np.zeros(shape=genes.shape)

  parents_1 = np.vstack((genes[:len(genes)//2], genes[:len(genes)//2]))
  parents_2 = np.vstack((genes[len(genes)//2:], genes[len(genes)//2:]))
  
  # create matrix for linear combos for numerical features
  alpha_num_cross = np.random.uniform(size=genes.shape)
  # create matrix to pick categories for categorical features
  mask_cat_cross = np.random.choice([True, False], size=genes.shape)

  # create mask to tell which features are categorical and which are not
  mask_categorical = np.zeros(genes.shape[1], bool)
  if (indices_categorical_features):
    mask_categorical[indices_categorical_features] = True

  for i, is_c in enumerate(mask_categorical):
    if is_c:
      # categorical
      offspring[:,i] = np.where(mask_cat_cross[:,i], parents_1[:,i], parents_2[:,i])
    else:
      # numerical
      offspring[:,i] = alpha_num_cross[:,i] * parents_1[:,i] + (1 - alpha_num_cross[:,i]) * parents_2[:,i]
  
  return offspring

# Mutation
def mutate(genes, feature_intervals, indices_categorical_features, 
  x, plausibility_constraints=None,
  mutation_probability=0.1, num_features_mutation_strength=0.05):

  # which genes will mutate
  mask_mut = np.random.choice([True, False], size=genes.shape, p=[mutation_probability, 1-mutation_probability])

  # instantiate mutations
  mutations = generate_plausible_mutations(genes, feature_intervals, indices_categorical_features, x, plausibility_constraints, num_features_mutation_strength)
  
  # apply mutations where decided by mask_mut
  offspring = np.where(mask_mut, mutations, genes).astype(float)

  return offspring



def generate_plausible_mutations(genes, feature_intervals, indices_categorical_features, x, plausibility_constraints=None, num_features_mutation_strength=0.25):
  # create mask to tell which features are categorical and which are not
  mask_categorical = get_mask_categorical_features(genes.shape[1], indices_categorical_features)

  # instantiate mutations
  mutations = np.zeros(shape=genes.shape)
  # fill in this matrix using feature ranges
  for i, is_c in enumerate(mask_categorical):
    if is_c:
      candidates = feature_intervals[i]
      if plausibility_constraints is not None and plausibility_constraints[i] is not None:
        if plausibility_constraints[i] == '=':
          candidates = [x[i]]
        else:
          raise ValueError("Unrecognized plausibility constraint",plausibility_constraints[i],"for feature",i)
        mutations[:,i] = np.random.choice(candidates, size=mutations.shape[0])
    else:
      if plausibility_constraints is not None and plausibility_constraints[i] is not None:
        if plausibility_constraints[i] == '=':
          low = high = range_num = 0
        elif plausibility_constraints[i] == '>=':
          range_num = feature_intervals[i][1] - x[i]
          low = 0
          high = num_features_mutation_strength
        elif plausibility_constraints[i] == '<=':
          range_num = x[i] - feature_intervals[i][0]
          low = -num_features_mutation_strength
          high = 0
        else:
          raise ValueError("Unrecognized plausibility constraint",plausibility_constraints[i],"for feature",i)
      else:
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2
        
      mutations[:,i] = range_num * np.random.uniform(low=low, high=high, size=mutations.shape[0])
      mutations[:,i] += genes[:,i]

      # fix out-of-range
      mutations[:,i] = np.where(mutations[:,i] > feature_intervals[i][1], feature_intervals[i][1], mutations[:,i])
      mutations[:,i] = np.where(mutations[:,i] < feature_intervals[i][0], feature_intervals[i][0], mutations[:,i])

  return mutations