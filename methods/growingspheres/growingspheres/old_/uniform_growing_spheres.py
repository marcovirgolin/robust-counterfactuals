

#entrées = 
#1. dataset (X)
#2. clf function
#3. observation to interprete

import math
import numpy as np
from numpy import random as nprand
import random
from sklearn.metrics.pairwise import pairwise_distances


### COST MEASURES TO TEST
def l1_norm(obs_to_interprete, observation):
    l1 = sum(map(abs, obs_to_interprete - observation))
    return l1

def weighted_l1(obs_to_interprete, observation):
    exp = sum(map(lambda x: math.exp(x**2), obs_to_interprete - observation))
    return exp

def penalized_l1(obs_to_interprete, observation):
    #nul
    GAMMA_ = 1.0
    l1 = l1_norm(observation)
    nonzeros = sum((obs_to_interprete - observation) != 0)
    return GAMMA_ * nonzeros + l1

def l2(obs1, obs2):
    return pairwise_distances(obs1.reshape(1, -1), obs2.reshape(1, -1))[0][0]

###OTHER
def distance_first_ennemy(X, observation, prediction_function):
    D = pairwise_distances(X, observation.reshape(1, -1), metric='euclidean', n_jobs=-1)
    idxes = sorted(enumerate(D), key=lambda x:x[1])
    for i in idxes:
        if (prediction_function(X[i[0]]) >= 0.5) != (prediction_function(observation) >=0.5):
            return X[i[0]], pairwise_distances(X[i[0]].reshape(1, -1), observation.reshape(1, -1))[0]

###GENERATION SPHERE AUTOUR. TROUVER ENNEMIS PUIS PRENDRE LE PLUS PROCHE
def generate_inside_ball(center, d, segment=(0,1), n=1):
    def norm(v):
        out = []
        for o in v:
            out.append(sum(map(lambda x: x**2, o))**(0.5))
        return np.array(out)
    z = nprand.normal(0, 1, (n, d))
    z = np.array([a * b / c for a, b, c in zip(z, nprand.uniform(*segment, n)**(1/float(d)),  norm(z))])
    z += center
    return z

def generate_layer_with_pred(prediction_function, center, d, n, segment):
    out = []
    a_ = generate_inside_ball(center, d, segment, n) #n observations
    pred_ = np.array([int(x>= 0.5) for x in prediction_function(a_)]) #predictions of n observations
    a_ = np.concatenate((a_, pred_.reshape(n, 1)), axis=1)
    return a_

def seek_ennemies(X, prediction_function, obs_to_interprete, n_layer=200, step=1/10000000.0, enough_ennemies=1):
    PRED_CLASS = int(prediction_function(obs_to_interprete)>=0.5)
    ennemies = []
    layer_ = []
    '''LAISSER CA MAIS UNIQUEMENT PARCE QUE ON LAISSE ENOUGH ENNEMY A 1'''
    fe, dfe = distance_first_ennemy(X, obs_to_interprete, prediction_function)
    step = dfe * step
    a0 = 0
    i = 0
    a1 = step
    while len(ennemies) < enough_ennemies and a1 <= dfe:
        layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
        layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS]
        ennemies.extend(layer_enn)
        while (i == 0) and (len(ennemies) > 0):
            layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
            layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS]
            ennemies.extend(layer_enn)
        #print(i, a1, len(ennemies))
        
        i += 1
        a0 += step
        a1 += step
    ennemies.append(np.array(list(fe) + [1 - PRED_CLASS]))
    #print('final nb of iterations ', i)
    #print('final number of ennemies generated ', len(ennemies)) 
    return layer_, ennemies

def seek_ennemies2(X, prediction_function, obs_to_interprete, n_layer, step, enough_ennemies):
    PRED_CLASS = int(prediction_function(obs_to_interprete)>=0.5)
    ennemies = []
    fe, dfe = distance_first_ennemy(X, obs_to_interprete, prediction_function)
    dfe = dfe[0]
    step = dfe * step
    a0, a1 = 0, step
    i = 0
    layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
    layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS]
    while len(layer_enn) > 0:
        step = step / 100.0
        a1 = step
        layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
        layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS]
        print('zoom')  
    else:
        while len(ennemies) < 1:
            layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
            layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS]
            ennemies.extend(layer_enn)
            i += 1
            a0 += step
            a1 += step
    print('Final nb of iterations ', i, 'Final radius', (a0,a1))
    return ennemies


def growing_sphere_explanation(X, prediction_function, obs_to_interprete, n_layer=10000, step=1/100000000.0, enough_ennemies=1, moving_cost=l2):
    ennemies = seek_ennemies2(X, prediction_function, obs_to_interprete, n_layer, step, enough_ennemies)
    nearest_ennemy = sorted(ennemies, key=lambda x: moving_cost(obs_to_interprete, x[:-1]))[0][:-1]
    return nearest_ennemy

def main(X, prediction_function, obs_to_interprete, **kwargs):
    return growing_sphere_explanation(X, prediction_function, obs_to_interprete, **kwargs)
