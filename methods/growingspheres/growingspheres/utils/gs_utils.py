#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances

    
def get_distances(x1, x2, metric='l2'):
    x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
    euclidean = pairwise_distances(x1, x2, metric='euclidean')[0][0]
    dist_metric = pairwise_distances(x1, x2, metric=metric)[0][0]
    same_coordinates = sum((x1 == x2)[0])
    
    #pearson = pearsonr(x1, x2)[0]
    kendall = kendalltau(x1, x2)
    out_dict = {
                'euclidean': euclidean,
                'sparsity': x1.shape[1] - same_coordinates,
                'kendall': kendall,
                'custom_metric': dist_metric,
               }
    return out_dict        


def generate_inside_ball(center, segment, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    
    d = center.shape[1] if len(center.shape) > 1 else center.shape[0]
    z = np.random.normal(0, 1, (n, d))
    u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    r = u**(1/float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r, norm(z))])
    z = z + center
    z = np.unique(z, axis=0).reshape((-1,d))
    return z
