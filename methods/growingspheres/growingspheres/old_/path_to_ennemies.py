
#entrées = 
#1. dataset (X)
#2. clf function
#3. observation to interprete
import random
from sklearn.metrics import pairwise_distances

def first_ennemy(X, observation, prediction_function, n_ennemies=1):
    D = pairwise_distances(X, observation.reshape(1, -1), metric='euclidean', n_jobs=-1)
    idxes = sorted(enumerate(D), key=lambda x:x[1])
    enn = []
    k = 0
    while len(enn) < n_ennemies:
        i = idxes[k]
        if (prediction_function(X[i[0]]) >= 0.5) != (prediction_function(observation) >=0.5):
            enn.append(X[i[0]])
        k += 1
    return enn

def distance(obs1, obs2):
    distance = pairwise_distances(obs1.reshape(1, -1), obs2.reshape(1, -1))
    return distance

def path_to_ennemy(X, prediction_function, obs_to_interprete, n_ennemies=3, n_layer=10000, which_enn=False):
    PRED_OBS = int(prediction_function(obs_to_interprete)>=0.5)
    ennemies = first_ennemy(X, obs_to_interprete, prediction_function, n_ennemies)
    min_d = 9999999999
    closest = []
    for k, enn in enumerate(ennemies):
        for i in range(n_layer):
            alpha = random.random()
            new = alpha * obs_to_interprete + (1 - alpha) * enn
            new_d = distance(new, obs_to_interprete)
            if (int(prediction_function(new)>=0.5) != PRED_OBS) & (new_d < min_d):
                closest = new
                min_d = new_d
                enn_used = k
    if which_enn:
        print('Ennemy used for closest generated:',enn_used)
    return closest
    
def main(X, prediction_function, obs_to_interprete, **kwargs):
    return path_to_ennemy(X, prediction_function, obs_to_interprete, **kwargs)
