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
    Executes a Markov Chain Monte Carlo simulation to estimate the distributions
    of class probabilities by using an augmented variable Gibbs sampling scheme.
    Censored count data is sampled in a two step process. First, the total augmented count
    for each data sample is drawn from a sum of geomteric distributions. Next, individual
    augmented counts are drawn from a multinomial distribution, conditioned on both the total
    augmented count and the current mcmc sample of probabilities.
        - counts is a 2-dim array of count values, including NaNs
        - prior_alpha is a 1-dim vector of concentration parameters from the dirichlet prior
        - repl is an integer of the desired number of mcmc samples
        - burn is an integer representing the number of samples in the burn-in phase
    """                                
    cdef:
        Py_ssize_t i,j,x,y,ii
        int N = counts.shape[0]
        int K = counts.shape[1]
        np.ndarray[int, ndim=1] I_missing = np.arange(N)[np.isnan(counts).any(1)].astype(np.int32)
        int N_nan = I_missing.shape[0]
        np.ndarray[double, ndim=2] J_missing = np.empty((N_nan,K),dtype=np.float64)
        np.ndarray[int, ndim=1] row_counts_missing = np.nansum(counts,1)[I_missing].astype(np.int32)
        np.ndarray[unsigned int, ndim=1] individual_augmented_counts = np.zeros(K, dtype=np.uint32)
        np.ndarray[double, ndim=1] p = np.array([1./K]*K, dtype=np.float64)
        np.ndarray[double, ndim=1] p_ = np.zeros(K, dtype=np.float64)
        np.ndarray[double, ndim=1] posterior_alpha = np.empty(K, dtype=np.float64)
        np.ndarray[double, ndim=2] trace = np.empty((repl, K), dtype=np.float64)
        np.ndarray[double, ndim=2] counts_cpy = counts.copy()
        int total_augmented_count = 0        
        double sum_p
        double k
        
    J_missing[:] = np.nan
    
    x = 0
    for i in I_missing:
        y = 0
        for j in xrange(K):
            if gsl_isnan(counts_cpy[i,j]):
                J_missing[x,y] = j
                y += 1
        x += 1
        
    for ii in xrange(repl+burn):
        for i in xrange(N_nan):
            sum_p = 1.0
            x = 0
            for j in xrange(K):
                k = J_missing[i,j]
                if gsl_isnan(k):
                    break
                else:
                    sum_p -= p[<Py_ssize_t>k]
                    p_[x] = p[<Py_ssize_t>k]
                    x += 1
                    
            total_augmented_count = -row_counts_missing[i]
            for k in xrange(row_counts_missing[i]):
                total_augmented_count += gsl_ran_geometric(r,sum_p)

            gsl_ran_multinomial(r,x,<unsigned int>total_augmented_count,&p_[0],&individual_augmented_counts[0])
            for j in xrange(K):
                k = J_missing[i,j]
                if gsl_isnan(k):
                    break
                else:
                    counts_cpy[<Py_ssize_t>I_missing[i],<Py_ssize_t>k] = individual_augmented_counts[j]
                    
        posterior_alpha = counts_cpy.sum(0) + prior_alpha
        gsl_ran_dirichlet(r,K,&posterior_alpha[0],&p[0])
        if ii >= burn:
            trace[ii-burn,:] = p
        
    return trace