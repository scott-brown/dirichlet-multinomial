# -*- coding: utf-8 -*-
import numpy as np
from itertools import product
from random import sample
from scipy.stats import dirichlet, geom
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
    
def logp(pvals, prior, counts):
    """
    pvals is a 1-dim array of class probabilities
    prior is a 1-dim array of concentration values
    counts is a 2-dim array of class count data
    that may or may not contain NaN values:
        - each row is a trial
        - each column corresponds to a unique class
    """   
    n,p = counts.shape
    pmat = np.tile(pvals,(n,1))
    pmat[np.isnan(counts)] = np.nan
    return np.sum(prior*np.log(pvals)) + np.nansum(counts*np.log(pmat)) - np.sum(np.nansum(counts,1)*np.log(np.nansum(pmat,1)))
    
def dirichlet_multinomial_MH(counts, alpha, repl = 10000, burn = 50, beta = 100.):   
    N,K = counts.shape
    if N == 0:
        trace = dirichlet.rvs(alpha, size = repl)
        accept = float(repl)
    else:
        accept = 0.
        x_t = np.array([1./K]*K)
        trace = np.empty((repl,K))
        for i in xrange(repl+burn):
            prop = dirichlet.rvs(x_t*beta).reshape(K)
            r = np.exp(logp(prop,alpha,counts) - logp(x_t,alpha,counts) + dirichlet.logpdf(x_t,prop*beta) - dirichlet.logpdf(prop,x_t*beta))
            if r > 1. or np.random.rand() < r:
                x_t = prop
                if i >= burn:
                    accept += 1.                    
            if i >= burn:
                trace[i-burn] = x_t
    return {'trace':trace,'acceptance_rate':accept/repl}
    
def dirichlet_multinomial_gibbs(counts, alpha, repl = 10000, burn = 50):
    N,K = counts.shape
    col_counts = np.nansum(counts,0)
    row_counts = np.nansum(counts,1)
    row_idx_missing = np.arange(N)[np.isnan(counts).any(1)]
    col_idx_missing = [np.where(row)[0] for row in np.isnan(counts)[row_idx_missing]]
    p = np.array([1./K]*K)
    trace = np.empty((repl,K))
    for ii in xrange(repl+burn):
        indv_augmented_counts = np.zeros(K)
        for (i,j) in zip(row_idx_missing,col_idx_missing):
            total_augmented_count = geom.rvs(1-p[j].sum(), loc = -1, size = row_counts[i]).sum()
            indv_augmented_counts[j] += multinomial(n = total_augmented_count, pvals = (p[j] / p[j].sum()))
        p = dirichlet.rvs(alpha = alpha + col_counts + indv_augmented_counts).reshape(K)
        if ii >= burn:
            trace[ii-burn] = p
    return trace