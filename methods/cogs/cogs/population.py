import numpy as np

from cogs.variation import generate_plausible_mutations
from cogs.util import *

class Population:

  def __init__(self, population_size, genotype_length):
    self.genes = np.empty(shape=(population_size, genotype_length))
    self.fitnesses = np.zeros(shape=(population_size,))
   
  def initialize(self, x, feature_intervals, indices_categorical_features, plausibility_constraints=None, num_features_max_mutation_strength=0.05):
    n = self.genes.shape[0]
    l = self.genes.shape[1]

    mask_categorical = get_mask_categorical_features(l, indices_categorical_features)

    # fill in this matrix using feature ranges according to plausib
    for i, is_c in enumerate(mask_categorical):
      init_feat_i = None
      if is_c:
        if plausibility_constraints is not None and plausibility_constraints[i] is not None:
          if plausibility_constraints[i] == '=':
            init_feat_i = x[i]
          else:
            raise ValueError("Unrecognized plausibility constraint",plausibility_constraints[i],"for feature",i)
        else:
          init_feat_i = np.random.choice(feature_intervals[i], size=n)
      else:
        if plausibility_constraints is not None and plausibility_constraints[i] is not None:
          if plausibility_constraints[i] == '=':
            init_feat_i = x[i]
          elif plausibility_constraints[i] == '>=':
            init_feat_i = np.random.uniform(low=x[i], high=feature_intervals[i][1], size=n)
          elif plausibility_constraints[i] == '<=':
            init_feat_i = np.random.uniform(low=feature_intervals[i][0], high=x[i], size=n)
          else:
            raise ValueError("Unrecognized plausibility constraint",plausibility_constraints[i],"for feature",i)
        else:
          init_feat_i = np.random.uniform(low=feature_intervals[i][0], high=feature_intervals[i][1], size=n)
      # make sure we did something
      assert(init_feat_i is not None)
      self.genes[:,i] = init_feat_i

    # copy values directly from x
    which_from_x = np.random.choice((True,False), size=self.genes.shape, p=[2/l, 1-2/l])
    self.genes = np.where(which_from_x, x, self.genes)

    

  def stack(self, other):
    self.genes = np.vstack((self.genes, other.genes))
    self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

  def shuffle(self):
    random_order = np.random.permutation(self.genes.shape[0])
    self.genes = self.genes[random_order,:]
    self.fitnesses = self.fitnesses[random_order]

  def is_converged(self):
    return len(np.unique(self.genes, axis=0)) < 2

  def delete(self, indices):
    self.genes = np.delete(self.genes, indices, axis=0)
    self.fitnesses = np.delete(self.fitnesses, indices)