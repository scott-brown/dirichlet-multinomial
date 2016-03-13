# -*- coding: utf-8 -*-
cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sampler(np.ndarray[np.float64_t, ndim=2] counts,
            np.ndarray[np.float64_t, ndim=1] prior_alpha,
            int repl = 10000,
            int burn = 50):
    """
    Executes a Markov Chain Monte Carlo simulation to estimate the distribution
    of class probabilities by using an augmented variable Gibbs sampling scheme.
    Censored count data is sampled in a two step process. First, the total augmented count
    for each data sample is drawn from a negative binomial distribution. Next, individual
    augmented counts are drawn from a multinomial distribution, conditioned on both the total
    augmented count and the current mcmc sample of probabilities.
        - counts is a 2-dim array of count values, including NaNs
        - prior_alpha is a 1-dim vector of concentration parameters from the dirichlet prior
        - repl is an integer of the desired number of mcmc samples
        - burn is an integer representing the number of samples in the burn-in phase
    """                                
    cdef:
        Py_ssize_t i,j,ii,jj,iii,cursor
        int N = counts.shape[0]
        int K = counts.shape[1]
        int N_missing
        double sum_p
        unsigned int total_augmented_count
        np.ndarray[int, ndim=1] row_idx_missing
        np.ndarray[double, ndim=1] row_counts_missing, col_counts
        np.ndarray[double, ndim=2] col_idx_missing
        np.ndarray[unsigned int, ndim=1] individual_augmented_counts = np.zeros(K, dtype=np.uint32)
        np.ndarray[double, ndim=1] p = np.array([1./K]*K, dtype=np.float64)
        np.ndarray[double, ndim=1] p_ = np.zeros(K, dtype=np.float64)
        np.ndarray[double, ndim=1] posterior_alpha = np.empty(K, dtype=np.float64)
        np.ndarray[double, ndim=2] trace = np.empty((repl, K), dtype=np.float64)
    
    row_idx_missing = np.arange(N)[np.isnan(counts).any(1)].astype(np.int32)
    N_missing = row_idx_missing.shape[0]
    
    row_counts_missing = np.nansum(counts,1)[row_idx_missing]
    col_counts = np.nansum(counts,0)
    
    # initialize col_idx_missing with NaN and then iterate over rows
    # filling each row from left to right with doubles that correspond
    # to the column indices of missing count data - doubles must be 
    # used in order to store NaN, the indices will be later converted
    # to integers (or Py_ssize_t) to perform actual indexing
    col_idx_missing = np.empty((N_missing,K),dtype=np.float64)
    col_idx_missing[:] = np.nan
        
    ii = 0                 
    for i in row_idx_missing:
        jj = 0
        for j in xrange(K):
            if gsl_isnan(counts[i,j]):
                col_idx_missing[ii,jj] = j
                jj += 1
        ii += 1
        
    ##### begin mcmc sampling #####
    # the following iterator conventions are used:
    # i,j are indices with respect to the original count data
    # ii,jj are indices with respect to the subset of count data concerning
    # censored/missing data - in other words, ii & jj are indices of indices
    # iii represents the index of mcmc samples
    # cursor indexes into p_ and is used as a flexible descriptor of the
    # length of the set of missing column indices for a given row
    for iii in xrange(repl+burn):
        posterior_alpha = col_counts + prior_alpha
        for ii in xrange(N_missing):
            sum_p = 1.0
            cursor = 0
            for jj in xrange(K):
                if gsl_isnan(col_idx_missing[ii,jj]):
                    break
                else:
                    j = <Py_ssize_t>col_idx_missing[ii,jj]
                    sum_p -= p[j]
                    p_[cursor] = p[j]
                    cursor += 1
             
            # sample augmented variables to "fill-in" censored data
            total_augmented_count = gsl_ran_negative_binomial(r,sum_p,row_counts_missing[ii])
            gsl_ran_multinomial(r,cursor,total_augmented_count,&p_[0],&individual_augmented_counts[0])
            
            # increment posterior alpha with the augmented counts
            for jj in xrange(K):
                if gsl_isnan(col_idx_missing[ii,jj]):
                    break
                else:
                    j = <Py_ssize_t>col_idx_missing[ii,jj]
                    posterior_alpha[j] += individual_augmented_counts[jj]
         
        # sample a new probability vector
        gsl_ran_dirichlet(r,K,&posterior_alpha[0],&p[0])
        if iii >= burn:
            trace[iii-burn,:] = p
            
    return trace