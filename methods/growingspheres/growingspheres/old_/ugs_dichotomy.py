

#entrées = 
#1. dataset (X)
#2. clf function
#3. observation to interprete

import math
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances




def norm(v):
        return np.linalg.norm(v, ord=2, axis=1) #array of l2 norms of vectors in v

def generate_inside_ball(center, segment=(0,1), n=1): #verifier algo comprendre racine 1/d et rapport entre segment et radius
    d = center.shape[0]
    z = np.random.normal(0, 1, (n, d))
    #z = np.array([a * b / c for a, b, c in zip(z, np.random.uniform(*segment, n)**(1/float(d)),  norm(z))])
    # apriori ma version pour scale etait completement fausse. Remplacer la ligne ci-dessus par:
    z = np.array([a * b / c for a, b, c in zip(z, np.random.uniform(*segment, n),  norm(z))]) #(pas besoin de puissance, non dep de la dimension...)
    z = z + center
    # les z sont a distance de center comprise dans le segment 
    return z

def generate_layer_with_pred(prediction_function, center, segment, n):
    out = []
    a_ = generate_inside_ball(center, segment, n) #n observations
    pred_ = prediction_function(a_) #predictions of n observations 
    a_ = np.concatenate((a_, pred_.reshape(n, 1)), axis=1)
    return a_

def ennemies_in_layer(layer, pred_obs, target_class):
    return layer[layer[:,-1]==target_class]

def find_ennemies(prediction_function, obs_to_interprete, target_class):
    ### Notes
    # Premiere sphere de rayon 0.1 (arbitraire)
    # si ennemis zoom en divisant rayon par 10
    # si plus d'ennemis augmenter rayon par 1/10e de l'ecart entre les rayons avec et sans ennemis, ou 1/10e de l'ecart avec la distance max si non

    ### parameters
    N_LAYER = 10000
    FIRST_RADIUS = 0.1
    DICREASE_RADIUS = 10
    last_radius = float(obs_to_interprete.shape[0]**(0.5) * 2)
    PRED_OBS = prediction_function(obs_to_interprete)
    if target_class == None:
        target_class = 1  - PRED_OBS

    # generate inside first ball
    layer_with_pred = generate_layer_with_pred(prediction_function, obs_to_interprete, (0, FIRST_RADIUS), N_LAYER)
    layer_ennemies = ennemies_in_layer(layer_with_pred, PRED_OBS, target_class)
    radius = FIRST_RADIUS

    #tant qu'il y en a, diviser par 2. Quand il y en a, augmenter de la moitié de l'écart. Quand il y en a pas, diminuer de la moitié de l'écart
    #phase de zoom identique
    # quand elle s'arrete, c'est qu'on avait des ennemis mais on en a plus.
    # pour iteration in itmax
    #	step = moitié de l'écart entre les deux derniers.
    #   si pas d ennemis, augmenter.
    #   si ennemis, diminuer

    while layer_ennemies.size > 0: # zoom phase
        print("%d ennemies found. Zooming in..." %layer_ennemies.size)
        radius = radius / DICREASE_RADIUS 
        layer_with_pred = generate_layer_with_pred(prediction_function, obs_to_interprete, (0, radius), N_LAYER)
        layer_ennemies = ennemies_in_layer(layer_with_pred, PRED_OBS, target_class)
        if layer_ennemies.size > 0:
            last_radius = radius
    else:
        # exploration
        print("Exploring...")        
        sens = 1
        imax = 10
        for it in range(imax):
            print(it)
            step = abs((last_radius - radius)/2.0)
            new_radius = radius + sens * step
            a0, a1 = 0, new_radius #min(radius, new_radius), max(radius, new_radius)
            layer_with_pred = generate_layer_with_pred(prediction_function, obs_to_interprete, (a0, a1), N_LAYER)
            layer_ennemies = ennemies_in_layer(layer_with_pred, PRED_OBS, target_class)
            if layer_ennemies.size > 0: #si ennemis on zoom, si non on explore 
                sens = -1
                #print('Yes')
                last_layer_with_ennemies = layer_ennemies
            else:
                sens = 1
                #print('No')
            last_radius = radius
            radius = new_radius

    return last_layer_with_ennemies #array de potentiellement plusieurs ennemis, en derniere colonne la classe


def growing_sphere(prediction_function, obs_to_interprete, target_class):
    ennemies_found = find_ennemies(prediction_function, obs_to_interprete, target_class)
    def l2(obs1, obs2):
        return pairwise_distances(obs1.reshape(1, -1), obs2.reshape(1, -1))[0][0]
    ennemies_found = np.array([x[:-1] for x in ennemies_found])
    closest_ennemy = sorted(ennemies_found, key= lambda x: l2(obs_to_interprete, x))[0]
    return closest_ennemy

def feature_selection(prediction_function, obs_to_interprete, ennemy):
    CLASS_ENNEMY = prediction_function(ennemy)
    new_enn = ennemy.copy()
    order = sorted(enumerate(abs(ennemy - obs_to_interprete)), key=lambda x: x[1])
    order = [x[0] for x in order if x[1] > 0.0]
    out = ennemy.copy()
    for k in order:
        new_enn[k] = obs_to_interprete[k]
        if prediction_function(new_enn) == CLASS_ENNEMY:
            out[k] = new_enn[k]
    return out
                   
def main(prediction_function, obs_to_interprete, target_class=None):
    closest_ennemy = growing_sphere(prediction_function, obs_to_interprete, target_class)
    explanation = feature_selection(prediction_function, obs_to_interprete, closest_ennemy) 
    return explanation

if __name__ == '__main__':
    raise NotImplementedError
