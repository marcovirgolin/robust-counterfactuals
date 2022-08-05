import inspect
import numpy as np
from copy import deepcopy
from cogs import selection, variation
from cogs.population import Population


class Evolution:
  def __init__(self,
    x,
    fitness_function,
    fitness_function_kwargs,
    feature_intervals,
    indices_categorical_features = None,
    plausibility_constraints=None,
    evolution_type='classic',
    population_size=1000,
    n_generations=100,
    mutation_probability='inv_mutable_genotype_length',
    num_features_mutation_strength=.25,
    num_features_mutation_strength_decay=None,
    num_features_mutation_strength_decay_generations=None,
    selection_name='tournament_2',
    noisy_evaluations=False,
    verbose=False,
    ):

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop('self')
    for arg, val in values.items():
      setattr(self, arg, val)

    # check that tournament size is compatible
    if 'tournament' in selection_name:
      self.tournament_size = int(selection_name.split('_')[-1])
      if self.population_size % self.tournament_size != 0:
        raise ValueError('The population size must be a multiple of the tournament size')

    # set up population and elite
    self.genotype_length = len(feature_intervals)
    self.population = Population(self.population_size, self.genotype_length)
    self.elite = None
    self.elite_fitness = -np.inf

    # set up mutation probability if set to default "inv_mutable_genotype_length"
    if mutation_probability == 'inv_genotype_length':
      mutation_probability = 1 / self.genotype_length
    elif mutation_probability == "inv_mutable_genotype_length":
      num_unmutable_features = len([p for p in plausibility_constraints if p == '='])
      self.mutation_probability = 1 / ( self.genotype_length - num_unmutable_features ) 
    
  def __update_elite(self, population):
    best_fitness_idx = np.argmax(population.fitnesses)
    best_fitness = population.fitnesses[best_fitness_idx]
    if self.noisy_evaluations or best_fitness > self.elite_fitness:
      self.elite = population.genes[best_fitness_idx, :].copy()
      self.elite_fitness = best_fitness

  def __classic_generation(self, merge_parent_offspring=False):
    # create offspring population
    offspring = Population(self.population_size, self.genotype_length)
    offspring.genes[:] = self.population.genes[:]
    offspring.shuffle()
    # variation
    offspring.genes = variation.crossover(offspring.genes)
    offspring.genes = variation.mutate(offspring.genes, self.feature_intervals, self.indices_categorical_features, 
      self.x, self.plausibility_constraints,
      mutation_probability=self.mutation_probability, num_features_mutation_strength = self.num_features_mutation_strength)
    # evaluate offspring
    offspring.fitnesses = self.fitness_function(Z=offspring.genes, x=self.x, 
      feature_intervals=self.feature_intervals, indices_categorical_features=self.indices_categorical_features,
      plausibility_constraints=self.plausibility_constraints,
      **self.fitness_function_kwargs)
    
    self.__update_elite(offspring)

    # selection
    if merge_parent_offspring:
      # p+o mode
      self.population.stack(offspring)
    else:
      # just replace the entire thing
      self.population = offspring

    self.population = selection.select(self.population, self.population_size, selection_name=self.selection_name)


  def __regularized_aging_generation(self, ablate_age_reg=False):
    # outer loop for uniform count of generations uniformly
    num_sub_generations = self.population_size // 2

    for _ in range(num_sub_generations):
      # select 2 parents
      idx_first_parent = selection.one_tournament_round(self.population, self.tournament_size, return_winner_index=True)
      idx_second_parent = selection.one_tournament_round(self.population, self.tournament_size, return_winner_index=True)

      # create offspring
      offspring = Population(2, self.genotype_length)
      offspring.genes[:] = self.population.genes[[idx_first_parent, idx_second_parent],:]
      # variation
      offspring.genes = variation.crossover(offspring.genes)
      offspring.genes = variation.mutate(offspring.genes, self.feature_intervals, self.indices_categorical_features, 
        self.x, self.plausibility_constraints,
        mutation_probability=self.mutation_probability, num_features_mutation_strength = self.num_features_mutation_strength)
      # evaluate offspring
      offspring.fitnesses = self.fitness_function(Z=offspring.genes, x=self.x, 
        feature_intervals=self.feature_intervals, indices_categorical_features=self.indices_categorical_features,
        plausibility_constraints=self.plausibility_constraints,
        **self.fitness_function_kwargs)

      # update elite
      self.__update_elite(offspring)

      # add offspring to the bottom
      self.population.stack(offspring)

      if not ablate_age_reg:
        # remove oldest (top)
        self.population.delete([0,1])
      else:
        # delete weakest
        indices_weakest = np.argsort(self.population.fitnesses)[:2]
        self.population.delete(indices_weakest)



  def run(self):
    #self.population.initialize(self.feature_intervals, self.indices_categorical_features)
    # initialize the population as mutated copies of x
    #self.population.genes[:] = self.x
    #self.population.genes = variation.mutate(self.population.genes, self.feature_intervals, self.indices_categorical_features, 
    #  mutation_probability=self.mutation_probability, num_features_mutation_strength = self.num_features_mutation_strength)
    self.population.initialize(self.x, self.feature_intervals, self.indices_categorical_features,
      self.plausibility_constraints, self.num_features_mutation_strength)

    self.population.fitnesses = self.fitness_function(Z=self.population.genes, x=self.x, 
      feature_intervals=self.feature_intervals, indices_categorical_features=self.indices_categorical_features,
      plausibility_constraints=self.plausibility_constraints,
      **self.fitness_function_kwargs)
    best_fitness_idx = np.argmax(self.population.fitnesses)
    best_fitness = self.population.fitnesses[best_fitness_idx]
    if best_fitness > self.elite_fitness:
      self.elite = self.population.genes[best_fitness_idx, :].copy()
      self.elite_fitness = best_fitness

    # run n_generations
    for i_gen in range(self.n_generations):

      if self.num_features_mutation_strength_decay_generations is not None:
        if i_gen in self.num_features_mutation_strength_decay_generations:
          self.num_features_mutation_strength *= self.num_features_mutation_strength_decay

      if self.evolution_type == 'classic':
        self.__classic_generation(merge_parent_offspring=False)
      elif self.evolution_type == 'p+o':
        self.__classic_generation(merge_parent_offspring=True)
      elif self.evolution_type == 'age_reg':
        self.__regularized_aging_generation()
      elif self.evolution_type == 'abl_age_reg':
        self.__regularized_aging_generation(ablate_age_reg=True)
      else:
        raise ValueError('unknown evolution type:',self.evolution_type)
     
      # generation terminated
      i_gen += 1
      if self.verbose:
        print('generation:',i_gen,'best fitness:',self.elite_fitness, 'avg. fitness:',np.mean(self.population.fitnesses))
        #print('generation:',i_gen,'best fitness:',self.elite_fitness, 'elite', self.elite, 'avg. fitness:',np.mean(self.population.fitnesses))

      # check if evolution should terminate because optimum reached or population converged
      if self.population.is_converged():
        break
