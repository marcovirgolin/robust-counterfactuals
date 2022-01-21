# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import check_random_state

from .utils.gs_utils import get_distances
from . import growingspheres


class CounterfactualExplanation:
    """
    Class for defining a Counterfactual Explanation: this class will help point to specific counterfactual approaches
    """
    def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None):
        """
        Init function
        method: algorithm to use
        random_state
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.method = method
        self.target_class = target_class
        self.random_state = check_random_state(random_state)
        
        self.methods_ = {'GS': growingspheres.GrowingSpheres,
                         #'HCLS': lash.HCLS,
                         #'directed_gs': growingspheres.DirectedGrowingSpheres
                        }
        self.fitted = 0
        
    def fit(self, caps=None, n_in_layer=2000, first_radius=0.1, dicrease_radius=10, metric='l2', sparse=True, verbose=False):
        """
        find the counterfactual with the specified method
        """
        cf = self.methods_[self.method](self.obs_to_interprete,
                self.prediction_fn,
                self.target_class,
                caps,
                n_in_layer,
                first_radius,
                dicrease_radius,
                metric,
                sparse,
                verbose)
        self.enemy = cf.find_counterfactual()
        self.e_star = cf.e_star
        self.move = self.enemy - self.obs_to_interprete
        self.fitted = 1
    
    def distances(self, metric='l2'):
        """
        scores de distances entre l'obs et le counterfactual
        """
        if self.fitted < 1:
            raise AttributeError('CounterfactualExplanation has to be fitted first!')
        return get_distances(self.obs_to_interprete, self.enemy, metric=metric)
    