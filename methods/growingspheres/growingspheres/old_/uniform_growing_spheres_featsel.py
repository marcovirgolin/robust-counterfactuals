

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

def seek_ennemies2(X, prediction_function, obs_to_interprete, n_layer, step, enough_ennemies):
    PRED_CLASS = int(prediction_function(obs_to_interprete)>=0.5)
    ennemies = []
    fe, dfe = distance_first_ennemy(X, obs_to_interprete, prediction_function)
    dfe = dfe[0]
    step = dfe * step / float(X.shape[1])**(0.5) * 10
    a0, a1 = 0, step
    i = 0
    print('premier rayon', a1 * (X.shape[1]**(0.5)))
    print('distance first ennemy', dfe)
    layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
    layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS] 
    while len(layer_enn) > 0:
        raise ValueError
        step = step / 10000000000.0
        last_a1 = a1
        a1 = step
        if a1 == 0:
            ennemies = layer_enn
            break
        layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
        layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS]
        print('zoom', 100 * len(layer_enn)/float(len(layer_)), '%', a1)
    else:
        try:
            step = (float(last_a1) - float(a1))/100.0
        except:
            pass
        while len(ennemies) < 1:
            #print('step ', step)
            layer_ = generate_layer_with_pred(prediction_function, obs_to_interprete, X.shape[1], n=n_layer, segment=(a0, a1))
            layer_enn = [x for x in layer_ if x[-1] == 1-PRED_CLASS]
            ennemies.extend(layer_enn)
            i += 1
            a0 = a1
            a1 += step
            #try:
            #    print('Current to last dicreasing a1', float(a1), float(last_a1))
            #except:
            #    pass
            #print('Current distance to first ennemy', 100 * a1 * X.shape[1]**(0.5)/dfe, '%')
            #if i%100 == 0:
            #    print('iteration', i) 
            if a1 * X.shape[1]**(0.5) > dfe:
                print('Filling in with first ennemy')
                fe = np.concatenate((fe.reshape(-1,), np.array(1 - PRED_CLASS).reshape(1,)), axis=0)
                ennemies.append(fe)
    print('Final nb of iterations ', i, 'Final radius', (a0,a1))
    return ennemies


def growing_sphere_explanation(X, prediction_function, obs_to_interprete, n_layer=1, step=1/100.0, enough_ennemies=1, moving_cost=l2):
    ennemies = seek_ennemies2(X, prediction_function, obs_to_interprete, n_layer, step, enough_ennemies)
    nearest_ennemy = sorted(ennemies, key=lambda x: moving_cost(obs_to_interprete, x[:-1]))[0][:-1]
    return nearest_ennemy

def featred_random(prediction_function, obs_to_interprete, ennemy):
    PRED_OBS = int(prediction_function(obs_to_interprete)>=0.5) 
    moves = map(abs, obs_to_interprete - ennemy)
    moves = sorted(enumerate(moves), key=lambda x: x[1])
    out = ennemy.copy()
    for d in moves:
        new = out.copy()
        if d[1] > 0.0:
            new[d[0]] = obs_to_interprete[d[0]]
            class_new = int(prediction_function(new)>= 0.5)
            if class_new != PRED_OBS: #si cest toujours un ennemi
                out = new
    return out

def main(X, prediction_function, obs_to_interprete, **kwargs):
    enn = growing_sphere_explanation(X, prediction_function, obs_to_interprete, **kwargs)
    explanation = featred_random(prediction_function, obs_to_interprete, enn)
    return explanation
