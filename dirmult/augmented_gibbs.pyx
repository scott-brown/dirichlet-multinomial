# -*- coding: utf-8 -*-
cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

def sampler(np.ndarray[np.float64_t, ndim=2] counts,
            np.ndarray[np.float64_t, ndim=1] prior_alpha,
            int repl = 10000,
            int burn = 50):
    cdef:
        Py_ssize_t i,j,x,y,ii
        int N = counts.shape[0]
        int K = counts.shape[1]
        np.ndarray[int, ndim=1] I_missing = np.arange(N)[np.isnan(counts).any(1)].astype(np.int32)
        np.ndarray[double, ndim=2] J_missing = np.empty((I_missing.shape[0],K),dtype=np.float64)
        np.ndarray[int, ndim=1] row_counts_missing = np.nansum(counts,1)[I_missing].astype(np.int32)
        np.ndarray[unsigned int, ndim=1] n = np.zeros(K, dtype=np.uint32)
        np.ndarray[double, ndim=1] p = np.array([1./K]*K, dtype=np.float64)
        np.ndarray[double, ndim=1] p_
        np.ndarray[double, ndim=1] posterior_alpha = np.empty(K, dtype=np.float64)
        np.ndarray[double, ndim=2] trace = np.empty((repl, K), dtype=np.float64)
        int augmented_count = 0
        int N_nan = I_missing.shape[0]
        double sum_p = 0.0
        double k
        
    J_missing[:] = np.nan
    
    x = 0
    for i in I_missing:
        y = 0
        for j in xrange(K):
            if gsl_isnan(counts[i,j]):
                J_missing[x,y] = j
                y += 1
        x += 1
        
    for ii in xrange(repl+burn):
        for i in xrange(N_nan):
            sum_p = 1.0
            p_ = np.zeros(K)
            x = 0
            for j in xrange(K):
                k = J_missing[i,j]
                if gsl_isnan(k):
                    break
                else:
                    sum_p -= p[<Py_ssize_t>k]
                    p_[x] = p[<Py_ssize_t>k]
                    x += 1
            augmented_count = 0
            for k in xrange(row_counts_missing[i]):
                augmented_count += gsl_ran_geometric(r,sum_p)
            augmented_count -= row_counts_missing[i]
            gsl_ran_multinomial(r,x,<unsigned int>augmented_count,&p_[0],&n[0])
            for j in xrange(K):
                k = J_missing[i,j]
                if gsl_isnan(k):
                    break
                else:
                    counts[<Py_ssize_t>I_missing[i],<Py_ssize_t>k] = <double>n[j]
        posterior_alpha = counts.sum(0) + prior_alpha
        gsl_ran_dirichlet(r,K,&posterior_alpha[0],&p[0])
        trace[ii-burn,:] = p
    
    return trace