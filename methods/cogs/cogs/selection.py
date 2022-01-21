import numpy as np
from cogs.population import Population

def select(population, selection_size, selection_name = 'tournament_4'):
  if 'tournament' in selection_name:
    tournament_size = int(selection_name.split('_')[-1])
    if 'roulette' in selection_name:
      return roulette_tournament_select(population, selection_size, tournament_size)
    else:
      return tournament_select(population, selection_size, tournament_size)
  elif selection_name == 'truncation':
    return truncation_select(population, selection_size)
  else:
    raise ValueError('Invalid selection name:',selection_name)

def multiobj_select(population, selection_size, selection_name = 'tournament_2'):
  if selection_name.startswith('tournament'):
    tournament_size = int(selection_name.split('_')[1])
    return mo_tournament_select(population, selection_size, tournament_size)
  else:
    raise ValueError('Invalid selection name:',selection_name)

def truncation_select(population, selection_size):
  genotype_length = population.genes.shape[1]
  selected = Population(selection_size, genotype_length)
  # shuffle
  population.shuffle()
  # get sort order of fitnesses
  sort_order = np.argsort(population.fitnesses*-1)[:selection_size]
  selected.genes = population.genes[sort_order,:]
  selected.fitnesses = population.fitnesses[sort_order]

  return selected

def one_tournament_round(population, tournament_size, return_winner_index=False):
  rand_perm = np.random.permutation(len(population.fitnesses))
  competing_fitnesses = population.fitnesses[rand_perm[:tournament_size]]
  winning_index = rand_perm[np.argmax(competing_fitnesses)]
  if return_winner_index:
    return winning_index
  else:
    return {
      'genotype':population.genes[winning_index,:], 
      'fitness':population.fitnesses[winning_index],
      }

def tournament_select(population, selection_size, tournament_size=4):
  genotype_length = population.genes.shape[1]
  selected = Population(selection_size, genotype_length)

  n = len(population.fitnesses)
  num_selected_per_iteration = n // tournament_size
  num_parses = selection_size // num_selected_per_iteration

  for i in range(num_parses):
    # shuffle
    population.shuffle()

    winning_indices = np.argmax(population.fitnesses.squeeze().reshape((-1, tournament_size)), axis=1)
    winning_indices += np.arange(0, n, tournament_size)

    selected.genes[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration,:] = population.genes[winning_indices,:]
    selected.fitnesses[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration] = population.fitnesses[winning_indices]

    '''
    # to debug
    for j in range(num_selected_per_iteration):
      start_idx = j*tournament_size
      end_idx = (j+1)*tournament_size

      best_idx = start_idx
      for k in range(start_idx+1, end_idx):
        if population.fitnesses[k,0] > population.fitnesses[best_idx,0]:
          best_idx = k
      selected.genes[i*num_selected_per_iteration + j, :] = population.genes[best_idx,:]
      selected.fitnesses[i*num_selected_per_iteration + j, :] = population.fitnesses[best_idx,:]      
    '''
    
  return selected

def roulette_tournament_select(population, selection_size, tournament_size=4):
  genotype_length = population.genes.shape[1]
  selected = Population(selection_size, genotype_length, 
    n_objectives=population.fitnesses.shape[1])

  n = len(population.fitnesses)
  num_selected_per_iteration = n // tournament_size
  num_parses = selection_size // num_selected_per_iteration

  for i in range(num_parses):
    # shuffle
    population.shuffle()

    '''
    winning_indices = np.argmax(population.fitnesses.squeeze().reshape((-1, tournament_size)), axis=1)
    winning_indices += np.arange(0, n, tournament_size)

    selected.genes[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration,:] = population.genes[winning_indices,:]
    selected.fitnesses[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration,:] = population.fitnesses[winning_indices,:]

    '''
    # to debug
    for j in range(num_selected_per_iteration):
      start_idx = j*tournament_size
      end_idx = (j+1)*tournament_size

      tournament_fitnesses = population.fitnesses[start_idx:end_idx,0]
      
      logits = tournament_fitnesses  #/ np.sum(tournament_fitnesses)
      temperature = 0.1
      logits = logits / temperature
      softmax = (np.exp(logits)) / np.sum(np.exp(logits))
      softmax /= np.sum(softmax)

      winning_idx = start_idx + np.random.choice(range(tournament_size), p=softmax)
      #print(tournament_fitnesses, softmax, winning_idx%tournament_size)
      #input()
      
      selected.genes[i*num_selected_per_iteration + j, :] = population.genes[winning_idx,:]
      selected.fitnesses[i*num_selected_per_iteration + j, :] = population.fitnesses[winning_idx,:]      
    
  return selected




def mo_tournament_select(population, selection_size, tournament_size=4):
  genotype_length = population.genes.shape[1]
  selected = Population(selection_size, genotype_length, 
    n_objectives=population.fitnesses.shape[1])

  n = len(population.ranks)
  num_selected_per_iteration = n // tournament_size
  num_parses = selection_size // num_selected_per_iteration

  for i in range(num_parses):
    # shuffle
    population.shuffle()

    '''
    ranks_grouped_in_tournaments = population.ranks.reshape((-1, tournament_size))
    best_ranks_per_tournament = np.argmin(ranks_grouped_in_tournaments, axis=1)
    crowddists_grouped_in_tournaments = population.crowding_distances.reshape((-1, tournament_size))

    #equally_good_rank_indices = ranks_grouped_in_tournaments[ranks_grouped_in_tournaments == best_ranks_per_tournament,:]
    crowddists_filtered = crowddists_grouped_in_tournaments[ranks_grouped_in_tournaments == best_ranks_per_tournament,:]
    winning_indices = np.argmax(crowddists_filtered, axis=1)
    winning_indices += np.arange(0, n, tournament_size)

    selected.genes[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration,:] = population.genes[winning_indices,:]
    selected.fitnesses[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration,:] = population.fitnesses[winning_indices,:]
    selected.ranks[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration,:] = population.ranks[winning_indices,:]
    selected.crowding_distances[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration,:] = population.crowding_distances[winning_indices,:]

    '''
    # to debug
    for j in range(num_selected_per_iteration):
      start_idx = j*tournament_size
      end_idx = (j+1)*tournament_size

      best_idx = start_idx
      for k in range(start_idx+1, end_idx):
        if population.ranks[k] < population.ranks[best_idx] or (population.ranks[k] == population.ranks[best_idx] and population.crowding_distances[k] > population.crowding_distances[best_idx]):
          best_idx = k
      selected.genes[i*num_selected_per_iteration + j, :] = population.genes[best_idx,:]
      selected.fitnesses[i*num_selected_per_iteration + j, :] = population.fitnesses[best_idx,:]  
      selected.ranks[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration] = population.ranks[best_idx]
      selected.crowding_distances[i*num_selected_per_iteration:(i+1)*num_selected_per_iteration] = population.crowding_distances[best_idx]
    
  return selected