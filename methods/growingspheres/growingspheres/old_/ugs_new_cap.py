

###
# inputs: obs_to_interprete, pred function
# out: ennemy
###

import math
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances

def norm(v):
        return np.linalg.norm(v, ord=2, axis=1) #array of l2 norms of vectors in v

def generate_inside_ball(center, segment=(0,1), n=1): #verifier algo comprendre racine 1/d et rapport entre segment et radius
    d = center.shape[0]
    z = np.random.normal(0, 1, (n, d))
    z = np.array([a * b / c for a, b, c in zip(z, np.random.uniform(*segment, n),  norm(z))])
    z = z + center
    return z # les z sont a distance de center comprise dans le segment 

def generate_layer_with_pred(prediction_function, center, segment, n):
    out = []
    a_ = generate_inside_ball(center, segment, n) #n observations
    capfunc = lambda x: min(max(x, -1.5), 2.5) ###ATTENTION CA CEST IMPORTANT: BORNES DES VARIABLES
    a_ = np.vectorize(capfunc)(a_)
    pred_ = prediction_function(a_)
    a_ = np.concatenate((a_, pred_.reshape(n, 1)), axis=1)
    return a_

def ennemies_in_layer(layer, pred_obs, target_class):
    return layer[layer[:,-1]==target_class]

def find_ennemies(prediction_function, obs_to_interprete, target_class):
    ### parameters
    N_LAYER = 10000
    FIRST_RADIUS = 0.1
    DICREASE_RADIUS = 10
    last_radius_with_ennemies = float(obs_to_interprete.shape[0]**(0.5) * 2)
    PRED_OBS = prediction_function(obs_to_interprete)
    if target_class == None:
        target_class = 1  - PRED_OBS

    ### generate inside first ball
    layer_with_pred = generate_layer_with_pred(prediction_function, obs_to_interprete, (0, FIRST_RADIUS), N_LAYER)
    layer_ennemies = ennemies_in_layer(layer_with_pred, PRED_OBS, target_class)
    radius = FIRST_RADIUS
    while layer_ennemies.size > 0: # zoom phase
        print("%d ennemies found. Zooming in..." %layer_ennemies.size)
        radius = radius / DICREASE_RADIUS 
        layer_with_pred = generate_layer_with_pred(prediction_function, obs_to_interprete, (0, radius), N_LAYER)
        layer_ennemies = ennemies_in_layer(layer_with_pred, PRED_OBS, target_class)
        if layer_ennemies.size > 0:
            last_radius_with_ennemies = radius
    else:
        it = 0
        ### exploration
        print("Exploring...")
        step = (last_radius_with_ennemies - radius)/10
        a0 = radius
        a1 = radius + step
        while layer_ennemies.shape[0] <= 0:
            layer_with_pred = generate_layer_with_pred(prediction_function, obs_to_interprete, (a0, a1), N_LAYER)
            layer_ennemies = ennemies_in_layer(layer_with_pred, PRED_OBS, target_class)
            a0 = a1
            a1 += step
            it += 1
        print('Final nb of iterations', it)
    print('Final radius', (a0, a1))
    print(layer_ennemies.shape)
    return layer_ennemies #array de potentiellement plusieurs ennemis, en derniere colonne la classe


def growing_sphere(prediction_function, obs_to_interprete, target_class):
    ennemies_found = find_ennemies(prediction_function, obs_to_interprete, target_class)
    def l2(obs1, obs2):
        return pairwise_distances(obs1.reshape(1, -1), obs2.reshape(1, -1))[0][0]
    #def cosine(obs1, obs2):
    #    return pairwise_distances(obs1.reshape(1, -1), obs2.reshape(1, -1), metric='cosine')[0][0]
    #def minkow(obs1, obs2, p=0.0001):
    #    m = sum(map(lambda x: abs(x)**float(p), obs1 - obs2))**(1/(p))
    #    return m
    ennemies_found = np.array([x[:-1] for x in ennemies_found])
    #closest_ennemy = ennemies_found
    closest_ennemy = sorted(ennemies_found, key= lambda x: l2(obs_to_interprete, x))[0] 
    return closest_ennemy

def feature_selection(prediction_function, obs_to_interprete, ennemy, i=0):
    CLASS_ENNEMY = prediction_function(ennemy)
    order = sorted(enumerate(abs(ennemy - obs_to_interprete)), key=lambda x: x[1])
    order = [x[0] for x in order if x[1] > 0.0]
    out = ennemy.copy()
    for k in order:
        new_enn = out.copy()
        new_enn[k] = obs_to_interprete[k]
        if prediction_function(new_enn) == CLASS_ENNEMY:
            out[k] = new_enn[k]
    #print('i', i)
    #if i < 3:
    #    print('redoing featsel', len(order))
    #    i += 1
    #    out = feature_selection(prediction_function, obs_to_interprete, out, i=i)
    return out
                   
def main(prediction_function, obs_to_interprete, target_class=None):
    closest_ennemy = growing_sphere(prediction_function, obs_to_interprete, target_class)
    #spars = np.array([feature_selection(prediction_function, obs_to_interprete, x) for x in closest_ennemy])
    #explanation = sorted(spars, key= lambda x: sum(obs_to_interprete!=x))[0]
    explanation = feature_selection(prediction_function, obs_to_interprete, closest_ennemy) 
    return explanation

if __name__ == '__main__':
    raise NotImplementedError
