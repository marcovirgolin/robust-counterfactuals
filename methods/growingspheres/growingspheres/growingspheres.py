#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .utils.gs_utils import generate_inside_ball, get_distances
from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state



class GrowingSpheres:
    """
    class to fit the Original Growing Spheres algorithm
    
    Inputs: 
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    caps: min max values of the explored area. Right now: if not None, the minimum and maximum values of the 
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=None,
                caps=None,
                n_in_layer=2000,
                first_radius=0.1,
                dicrease_radius=10,
                metric='l2',
                sparse=True,
                verbose=False):
        """
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        
        if target_class == None: #To change: works only for binary classification...
            target_class = 1 - self.y_obs
        
        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius 
        self.metric = metric
        self.sparse = sparse
        
        self.verbose = verbose
        
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")

        
    def find_counterfactual(self):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        ennemies_ = self.exploration()
        closest_ennemy_ = sorted(ennemies_, 
                                 key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1), metric=self.metric))[0] 
        self.e_star = closest_ennemy_
        if self.sparse == True:
            out = self.feature_selection(closest_ennemy_)
        else:
            out = closest_ennemy_
        return out
    
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_ennemies_ = 999
        radius_ = self.first_radius
        
        while n_ennemies_ > 0:
            first_layer_ = self.ennemies_in_layer_((0, radius_), self.caps, self.n_in_layer)
            n_ennemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.dicrease_radius
            if self.verbose == True:
                print(n_ennemies_," enemies found in initial sphere. Zooming in... (curr radius:",radius_,")")
            
        else:
            if self.verbose == True:
                print("Exploring...")
            iteration = 0
            step_ = (self.dicrease_radius - 1) * radius_/5.0
            
            while n_ennemies_ <= 0:
                layer = self.ennemies_in_layer_((radius_, radius_ + step_), self.caps, self.n_in_layer)
                n_ennemies_ = layer.shape[0]
                radius_ = radius_ + step_
                iteration += 1
            if self.verbose == True:
                print("Final number of iterations: ", iteration)
        if self.verbose == True:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
        return layer
    
    
    def ennemies_in_layer_(self, segment, caps=None, n=1000):
        """
        Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """
        layer = generate_inside_ball(self.obs_to_interprete, segment, n)
        #cap here: not optimal
        if caps != None:
            cap_fn_ = lambda x: min(max(x, caps[0]), caps[1])
            layer = np.vectorize(cap_fn_)(layer)
            
        preds_ = self.prediction_fn(layer)
        return layer[np.where(preds_ == self.target_class)]        
    
    
    def feature_selection(self, counterfactual):
        """
        Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class
        
        Inputs:
        counterfactual: e*
        """
        if self.verbose == True:
            print("Feature selection...")
        move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete)), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        out = counterfactual.copy()
        reduced = 0
        
        for k in move_sorted:
            new_enn = out.copy()
            new_enn[k] = self.obs_to_interprete[k]
            if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                out[k] = new_enn[k]
                reduced += 1
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out

    
    def feature_selection_all(self, counterfactual):
        """
        Try all possible combinations of projections to make the explanation as sparse as possible. 
        Warning: really long!
        """
        if self.verbose == True:
            print("Grid search for projections...")
        for k in range(self.obs_to_interprete.size):
            print('==========', k, '==========')
            for combo in combinations(range(self.obs_to_interprete.size), k):
                out = counterfactual.copy()
                new_enn = out.copy()
                for v in combo:
                    new_enn[v] = self.obs_to_interprete[v]
                if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                    print('bim')
                    out = new_enn.copy()
                    reduced = k
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out
