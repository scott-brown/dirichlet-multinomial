# -*- coding: utf-8 -*-
import numpy as np
from itertools import product
from random import sample
from scipy.stats import dirichlet
from numpy.random import multinomial

def flatten(ls):
    return [item for sublist in ls for item in sublist]

def random_index(n,p,x,minp=1):
    """
    n,p,x are all integer arguments
    n & p specify the row,column dimensions of a 2d array
    x is the desired number of index tuples to be returned
    minp is the minimum number of non-missing values per row
    """
    all_idx = list(product(xrange(n),xrange(p)))
    exclude_idx = flatten(sample(all_idx[(p*i):(p*i+p)],minp) for i in xrange(n))
    return sample(set(all_idx) - set(exclude_idx),x)
    
def nansample(alpha,total_count):
    k = alpha.shape[0]
    mask = ~np.isnan(alpha)
    sample = np.empty((k,))
    sample[:] = np.nan
    sample[mask] = multinomial(total_count, pvals = dirichlet.rvs(alpha[mask],size=1).reshape(mask.sum()))
    return sample
    
def generate_fake_counts(alpha, total_count_range, n, n_nan):
    total_counts_per_row = np.random.randint(*total_count_range,size=n)
    p = alpha.shape[0]
    nan_idx = random_index(n,p,n_nan)
    alpha_tile = np.tile(alpha,(n,1))
    for (i,j) in nan_idx:
        alpha_tile[i,j] = np.nan 
    return np.array(map(nansample, alpha_tile, total_counts_per_row))