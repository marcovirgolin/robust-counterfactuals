import numpy as np
import warnings

from cogs.fitness import gower_fitness_function
from cogs.util import fix_features
from robust_cfe.dataproc import gimme as gimme_dataset
from robust_cfe.robustness import *

# wrapper-specific imports
from cogs.evolution import Evolution
import scipy
from lore import lore
from lore.neighbor_generator import genetic_neighborhood
from growingspheres.counterfactuals import CounterfactualExplanation as GrowingSpheresCFs
import cma
import pandas as pd
import dice_ml
import fatf.transparency.predictions.counterfactuals as fatf_cf



class Wrapper:

  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function='gower', check_plausibility=False, 
    optimize_C_robust=False, optimize_K_robust=0,
    method_kwargs=None):

    # assert problem makes sense
    assert(blackbox.predict([x])[0] != desired_class)
    
    # store some info
    self.x = x
    self.dataset = dataset
    self.blackbox = blackbox
    self.desired_class = desired_class
    self.check_plausibility = check_plausibility
    self.optimize_C_robust = optimize_C_robust
    self.optimize_K_robust = optimize_K_robust

    # set up similarity function
    if type(similarity_function) is str:
      if similarity_function == 'gower':
        self.similarity_function = gower_fitness_function
        if optimize_C_robust:
          self.similarity_function = gower_fitness_function_with_worst_C_setbacks
      else:
        raise ValueError("Unknown similarity function name:", similarity_function)

    # kwargs of the specific method
    self.method_kwargs = method_kwargs

  def find_cfe(self):
    raise NotImplementedError("Not implemented")



class ScipyOptWrapper(Wrapper):

  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function='gower', check_plausibility=False,
    optimize_C_robust=False, optimize_K_robust=0,
    method_kwargs=None): 

    # inherit init from base class
    Wrapper.__init__(self, x, dataset, blackbox, desired_class, 
        similarity_function, check_plausibility, 
        optimize_C_robust, optimize_K_robust, method_kwargs)

    if check_plausibility:
      raise ValueError("Check plausibility is not supported")
    if optimize_C_robust:
      raise ValueError("Optimize for C-robustness is not supported")
    if optimize_K_robust > 0:
      raise ValueError("Optimize for K-robustness is not supported")

    # scipy optimizer minimizes a loss function
    def loss_function(z, x, blackbox, desired_class, feature_intervals, 
      indices_categorical_features=None, plausibility_constraints=None, apply_fixes=False):
      loss = -self.similarity_function(z, x, blackbox, desired_class, feature_intervals, 
          indices_categorical_features, plausibility_constraints, apply_fixes)
      return loss
    self.loss_function = loss_function
    self.loss_args = (self.x, self.blackbox, self.desired_class, 
      self.dataset['feature_intervals'], self.dataset['indices_categorical_features'], 
      self.dataset['plausibility_constraints'], False)
    

  def find_cfe(self):

    # set defaults for kwargs if None
    if self.method_kwargs is None:
      self.method_kwargs = { 
        'optimizer_name' : 'Nelder-Mead',
        'optimizer_options': {'maxiter':100}
      }

    result = scipy.optimize.minimize(self.loss_function, self.x, 
      args=self.loss_args, bounds=self.dataset['feature_intervals'], 
      method=self.method_kwargs['optimizer_name'], options=self.method_kwargs['optimizer_options'])
    
    #z = fix_features(result.x, self.x, self.dataset['feature_intervals'], self.dataset['indices_categorical_features'], self.dataset['plausibility_constraints'])
    z = fix_categorical_features(result.x, self.dataset['feature_intervals'], self.dataset['indices_categorical_features'])

    return z


class CogsWrapper(Wrapper):

  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function='gower', check_plausibility=False,
    optimize_C_robust=False,
    optimize_K_robust=0,
    method_kwargs=None):
    # inherit init from base class
    Wrapper.__init__(self, x, dataset, blackbox, desired_class, 
        similarity_function, check_plausibility, 
        optimize_C_robust, optimize_K_robust, method_kwargs)

  def find_cfe(self):

    if self.method_kwargs is None:

      fitness_function = gower_fitness_function
      fitness_function_kwargs = {'blackbox': self.blackbox, 'desired_class': self.desired_class}
      if self.optimize_C_robust and self.optimize_K_robust == 0:
        fitness_function = gower_fitness_function_with_worst_C_setbacks
      elif not self.optimize_C_robust and self.optimize_K_robust > 0:
        fitness_function = gower_fitness_function_with_K_robustness_scores
      elif self.optimize_C_robust and self.optimize_K_robust > 0:
        fitness_function = gower_fitness_function_with_CK_robustness
      
      if self.optimize_C_robust:
        fitness_function_kwargs['perturb'] = self.dataset['perturbations']
      if self.optimize_K_robust > 0:
        fitness_function_kwargs['perturb'] = self.dataset['perturbations']
        fitness_function_kwargs['n_samples_k_robust'] = self.optimize_K_robust

      evo = Evolution(
          x=self.x,
          fitness_function=fitness_function,
          fitness_function_kwargs=fitness_function_kwargs,
          feature_intervals=self.dataset['feature_intervals'],
          indices_categorical_features=self.dataset['indices_categorical_features'],
          plausibility_constraints=self.dataset['plausibility_constraints'],
          mutation_probability='inv_mutable_genotype_length', 
          noisy_evaluations=self.optimize_K_robust > 0,
          verbose=False,
          )
    else:
      evo = Evolution(**self.method_kwargs)

    evo.run()
    z = evo.elite
    return z


class DiCEWrapper(Wrapper):
  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function=None, check_plausibility=False,
    optimize_C_robust=False, optimize_K_robust=0,
    method_kwargs=None):

    # inherit init from base class
    Wrapper.__init__(self, x, dataset, blackbox, desired_class, 
        similarity_function, check_plausibility, 
        optimize_C_robust, optimize_K_robust, method_kwargs)

    if check_plausibility:
      raise ValueError("Check plausibility is not supported")
    if optimize_C_robust:
      raise ValueError("Optimize for C-robustness is not supported")
    if optimize_K_robust > 0:
      raise ValueError("Optimize for K-robustness is not supported")
    if similarity_function is not None:
      raise ValueError("Custom similarity functions not supported for DiCE")

    continuous_feature_names = [n for n in dataset["feature_names"] if n not in dataset["categorical_feature_names"]]
    self.dice_x = pd.DataFrame(x.reshape((1,-1)), columns=dataset["feature_names"])
    # dice breaks if categorical features are expressed as floats (e.g., "0.0" instead of "0")
    for feat in dataset["categorical_feature_names"]:
      self.dice_x[feat] = self.dice_x[feat].astype("int")
    self.dice_data = dice_ml.Data(dataframe=dataset["df"], continuous_features=continuous_feature_names, outcome_name='LABEL')
    self.dice_model = dice_ml.Model(model=blackbox, backend="sklearn")

    if self.method_kwargs and "method" in self.method_kwargs:
      self.method = self.method_kwargs["method"]
    else:
      self.method = "genetic"

  def find_cfe(self):
    exp = dice_ml.Dice(self.dice_data, self.dice_model, method=self.method)
    try:
      result = exp.generate_counterfactuals(self.dice_x, total_CFs=1, desired_class=self.desired_class)
      z = result.cf_examples_list[0].final_cfs_df.drop("LABEL",axis=1).to_numpy().reshape((-1,)).astype(float)
      return z
    except dice_ml.utils.exception.UserConfigValidationException:
      # failed to find a cfe
      return self.x

class FatFWrapper(Wrapper):
  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function=None, check_plausibility=False,
    optimize_C_robust=False, optimize_K_robust=0,
    method_kwargs=None):

    # inherit init from base class
    Wrapper.__init__(self, x, dataset, blackbox, desired_class, 
        similarity_function, check_plausibility, 
        optimize_C_robust, optimize_K_robust, method_kwargs)

    if check_plausibility:
      raise ValueError("Check plausibility is not supported")
    if optimize_C_robust:
      raise ValueError("Optimize for C-robustness is not supported")
    if optimize_K_robust > 0:
      raise ValueError("Optimize for K-robustness is not supported")
    if similarity_function is not None:
      raise ValueError("Custom similarity functions not supported for DiCE")

    # feature ranges in fatf format
    self.feature_ranges = dict()
    for i, _ in enumerate(self.dataset["feature_names"]):
      if i in self.dataset["indices_categorical_features"]:
        self.feature_ranges[i] = list(self.dataset["feature_intervals"][i])
      else:
        self.feature_ranges[i] = tuple(self.dataset["feature_intervals"][i])

    # distance functions: essentially gower
    self.distance_functions = dict()
    for i, _ in enumerate(self.dataset["feature_names"]):
      if i in self.dataset["indices_categorical_features"]:
        def categorical_distance(a,b):
          return 0 if a==b else 1.0
        self.distance_functions[i] = categorical_distance
      else:
        curr_range = self.feature_ranges[i][1] - self.feature_ranges[i][0]
        def numerical_distance_this_feature(a,b):
          dist = np.abs(a-b) / curr_range
          return dist
        self.distance_functions[i] = numerical_distance_this_feature

  def find_cfe(self):
    # set up experiment
    exp = fatf_cf.CounterfactualExplainer(
      model=self.blackbox, 
      dataset=self.dataset["X"], 
      feature_ranges = self.feature_ranges,
      distance_functions = self.distance_functions,
      max_counterfactual_length = 0,
      categorical_indices=self.dataset["indices_categorical_features"],
      default_numerical_step_size=0.1)

    # get counterfactuals
    cfs = exp.explain_instance(self.x)
    
    # first closet of the desired class
    for i, pred in enumerate(cfs[2]): # cfs[2] contains the predictions
      if pred == self.desired_class:
        break

    if i == len(cfs[2]):
      # fail
      z = self.x.copy()
    else:
      z = cfs[0][i]     
    return z

class GrowingSpheresWrapper(Wrapper):
  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function=None, check_plausibility=False,
    optimize_C_robust=False, optimize_K_robust=0,
    method_kwargs=None):

    # inherit init from base class
    Wrapper.__init__(self, x, dataset, blackbox, desired_class, 
        similarity_function, check_plausibility, 
        optimize_C_robust, optimize_K_robust, method_kwargs)

    if check_plausibility:
      raise ValueError("Check plausibility is not supported")
    if optimize_C_robust:
      raise ValueError("Optimize for C-robustness is not supported")
    if optimize_K_robust > 0:
      raise ValueError("Optimize for K-robustness is not supported")
    if similarity_function is not None:
      raise ValueError("Custom similarity functions not supported for growing spheres, please set 'metric' in gs_kwargs")

    # set up gower distance in case we want to use it
    indices_numerical_features = [i for i in range(len(x)) if i not in self.dataset['indices_categorical_features']] 
    ranges_numerical_features = [intv[1]-intv[0] for i,intv in enumerate(self.dataset['feature_intervals']) if i in indices_numerical_features]
    def gower(x, z):
        x_c, z_c = x[self.dataset['indices_categorical_features']], z[self.dataset['indices_categorical_features']]
        x_n, z_n = x[indices_numerical_features], z[indices_numerical_features]
        d_c = np.sum(x_c != z_c)
        d_n = np.sum(np.divide(np.abs(x_n - z_n), ranges_numerical_features))
        d = 1/len(x) * (d_c + d_n)
        return d
    self.gower = gower
    # if metric='gower' in method_kwargs, update it to the callable function
    if self.method_kwargs is not None and 'metric' in self.method_kwargs and self.method_kwargs['metric'] == 'gower':
        self.method_kwargs['metric'] = self.gower

  
  def find_cfe(self):

    gs = GrowingSpheresCFs(self.x, self.blackbox.predict, method='GS', target_class = self.desired_class)
    if self.method_kwargs is not None:
      gs.fit(**self.method_kwargs)
    else:
      gs.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, metric=self.gower, sparse=True, verbose=False) 

    z = fix_categorical_features(gs.enemy, self.dataset['feature_intervals'], self.dataset['indices_categorical_features'])
    return z


class LOREWrapper(Wrapper):
  
  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function=None, check_plausibility=False,
    optimize_C_robust=False, optimize_K_robust=0,
    method_kwargs=None):

    # inherit init from base class
    Wrapper.__init__(self, x, dataset, blackbox, desired_class, 
        similarity_function, check_plausibility, 
        optimize_C_robust, optimize_K_robust, 
        method_kwargs)

    # check plausibility not available for LORE
    if check_plausibility:
      raise ValueError("Check plausibility not available for LORE")

    if optimize_C_robust:
      raise ValueError("Optimize under worst-case C setbacks not available for LORE")

    # no option to change similarity function
    if similarity_function:
      raise ValueError("LORE (or better, this implementation) does not support custom similarity functions")

    # check that the dataset is in LORE format
    if not hasattr(dataset, 'class_name'):
      warnings.warn("Using LOREWrapper but data set appears *not* to be in LORE format, attempting re-load")
      # temp store X and y in case they are set to the test set or something else than the full original data, for some reason
      X, y = dataset['X'], dataset['y'] 
      self.dataset = gimme_dataset(dataset['name'], return_lore_version=True)
      self.dataset['X'], self.dataset['y'] = X, y 



  def find_cfe(self):
    
    # lore requires index of record to explain, let's find it out
    idx_x = np.squeeze(np.where(np.all(self.x==self.dataset['X'], axis=1)))
    explanation, _ = lore.explain(idx_x, self.dataset['X'], self.dataset, self.blackbox,
                                      ng_function=genetic_neighborhood,
                                      discrete_use_probabilities=False,
                                      continuous_function_estimation=False,
                                      returns_infos=False,
                                      path="./methods/lore/lore/", sep=';', log=False)
    z = lore.explanation_to_cfe(explanation, self.x, self.dataset['feature_names'], self.blackbox, self.desired_class)

    return z


class CMAWrapper(Wrapper):

  def __init__(self, x, dataset, blackbox, desired_class,
    similarity_function='gower', check_plausibility=False, 
    optimize_C_robust=False, optimize_K_robust=0, 
    method_kwargs=None):

    Wrapper.__init__(self, x, dataset, blackbox, desired_class, 
        similarity_function, check_plausibility, 
        optimize_C_robust, optimize_K_robust, method_kwargs)
    
    # to work between 0 and 1
    self.precision = 6
    self.scaled_x = self.scale(x)
    if self.precision is not None:
      self.scaled_x = np.round(self.scaled_x, self.precision) # up to some level of precision

    # Set bounds w.r.t. min-max normalization
    self.bounds = dataset['feature_intervals'].copy()
    for i in range(len(x)):
      if i in dataset['indices_categorical_features']:
        intv = self.bounds[i][1] - self.bounds[i][0]
        st_min = 2*np.min(dataset['feature_intervals'][i]) - intv
        st_max = 2*np.max(dataset['feature_intervals'][i]) - intv
        self.bounds[i] = [st_min, st_max]
      else:
        self.bounds[i] = [0,1]
    
  # We use min-max normalization during the search
  def scale(self, x):
    st_x = x.copy().astype(float)
    for i in range(len(x)):
      intv = self.dataset['feature_intervals'][i][1] - self.dataset['feature_intervals'][i][0]
      if i in self.dataset['indices_categorical_features']:
        # center around 0
        st_x[i] = 2*x[i] - intv
      else:
        st_x[i] = (x[i] - self.dataset['feature_intervals'][i][0]) / intv
    return st_x

  def unscale(self, st_x):
    x = st_x.copy()
    is_single_point = len(x.shape) < 2
    if is_single_point:
      x = x.reshape((1,-1))
      st_x = st_x.reshape((1,-1))
    n_dim = x.shape[1]
    for i in range(n_dim):
      intv = self.dataset['feature_intervals'][i][1] - self.dataset['feature_intervals'][i][0]
      if i in self.dataset['indices_categorical_features']:
        # center around 0
        x[:,i] = (st_x[:,i] + intv) / 2
      else:
        x[:,i] = st_x[:,i] * intv + self.dataset['feature_intervals'][i][0]
    if is_single_point:
      x = x.reshape((-1))
    return x


  def evaluate(self, Z):
    Z = np.array(Z)
    Z_unsc = self.unscale(Z)
    if self.precision is not None:
      Z_unsc = np.round(self.unscale(Z), self.precision)
    # Wrapping this function inside cma.s.ft.IntegerMixedFunction seems not to work for "parallel_objective" 
    # So we fix categories ourselves here
    Z_unsc = fix_categorical_features(Z_unsc, self.dataset['feature_intervals'], self.dataset['indices_categorical_features'])
    objs = -gower_fitness_function(Z_unsc, self.x, self.blackbox, self.desired_class, 
      self.bounds, self.dataset['indices_categorical_features'], self.dataset['plausibility_constraints'], apply_fixes=False)
    # CMA likes lists
    objs = objs.tolist()
    return objs


  def find_cfe(self):

    cma_bounds = [[self.bounds[i][0] for i in range(len(self.bounds))], [self.bounds[i][1] for i in range(len(self.bounds))]]

    opts = cma.CMAOptions()
    opts['bounds'] = cma_bounds
    opts['integer_variables'] = self.dataset['indices_categorical_features']
    opts['maxfevals'] = 100*1000
    opts['maxiter'] = np.inf
    opts['popsize'] = 100 #4 + 3 *np.log(len(self.x)) # default
    opts['verbose'] = -9

    sigma = 0.25 # advise is to set it to 1/4th of space, this is handled automatically for integer variables
    res = cma.fmin(None, self.scaled_x, sigma, args=(), options=opts, 
      eval_initial_x=True,
      parallel_objective=self.evaluate)
    z = self.unscale(res[0])
    if self.precision is not None:
      z = np.round(z, self.precision)
    z = fix_categorical_features(z, self.dataset['feature_intervals'], self.dataset['indices_categorical_features'])
    return z