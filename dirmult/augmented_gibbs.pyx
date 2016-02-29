# -*- coding: utf-8 -*-
cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)   
def sampler(np.ndarray[np.float64_t, ndim=2] counts,
            np.ndarray[np.float64_t, ndim=1] prior_alpha,
            int repl = 10000,
            int burn = 50):
    cdef:
        Py_ssize_t i
        Py_ssize_t j,k
        Py_ssize_t ii,jj
        int n = counts.shape[0]
        int n_nan = np.isnan(counts).sum()
        size_t K = counts.shape[1]
        np.ndarray[double, ndim=2] trace = np.empty((repl, K), dtype=np.float64)
        np.ndarray[double, ndim=1] p = np.array([1./K]*K, dtype=np.float64)
        np.ndarray[double, ndim=1] col_counts = np.nansum(counts,0)
        np.ndarray[double, ndim=1] row_counts = np.nansum(counts,1)
        np.ndarray[int, ndim=1] col_idx = np.empty(n_nan, dtype=np.int32)
        np.ndarray[int, ndim=1] row_idx = np.empty(n_nan, dtype=np.int32)
        int aug
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        
    it = np.nditer(counts,flags=['multi_index'])
    j = 0
    while not it.finished:
        if np.isnan(it[0]):
            row_idx[j] = it.multi_index[0]
            col_idx[j] = it.multi_index[1]
            j += 1
        it.iternext()
        
    for i in range(repl+burn):
        
        for j in range(n_nan):
            ii = row_idx[j]
            jj = col_idx[j]
            for k in range(row_counts[ii]):
                aug += gsl_ran_geometric(r,p[jj])
            
        if r >= burn:
            trace[r-burn,:] = 
            
        
def dirichlet_multinomial_gibbs(X, alpha = None, form = 'probability', repl = 10000, burn = 25):
    import numpy as np
    from scipy.stats import dirichlet, geom
    assert form in {'probability','concentration'}
    n,k = X.shape
    if alpha is None:
        alpha = np.array([1.]*k)
    col_counts = np.nansum(X,0)
    row_counts = np.nansum(X,1)
    I_missing = np.arange(n)[np.isnan(X).any(1)]
    J_missing = [np.where(row)[0] for row in np.isnan(X)[I_missing]]
    p = np.array([1./k]*k)
    augment = np.zeros(k)
    trace = np.empty((repl,k))
    for r in xrange(repl+burn):
        augment = np.zeros(k)
        for (i,j) in zip(I_missing,J_missing):
            missing_count = geom.rvs(1-p[j].sum(), loc = -1, size = row_counts[i]).sum()
            augment[j] += np.random.multinomial(n = missing_count, pvals = (p[j] / p[j].sum()))
        if r >= burn:
            if form == 'probability':
                trace[r-burn] = dirichlet.rvs(alpha = alpha + col_counts + augment).reshape(k)
            if form == 'concentration':
                trace[r-burn] = alpha + col_counts + augment
    return trace