# -*- coding: utf-8 -*-
cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np

from libc.math cimport log, exp

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
cdef gsl_rng *u = gsl_rng_alloc(gsl_rng_mt19937)
     
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef log_posterior(np.ndarray[np.float64_t, ndim=1] probability,
                    np.ndarray[np.float64_t, ndim=1] alpha,
                    np.ndarray[np.float64_t, ndim=2] counts):
    """
    Calculates a value that's proportional to the
    posterior density and returns its logarithm. 
    In this case, the posterior model follows from a
    censored multinomial likelihood and dirichlet prior.
        - probability is a 1-dim vector of probabilities
        - alpha is a 1-dim vector of concentration parameters from the dirichlet prior
        - counts is a 2-dim array of count values, including NaNs
    """
    cdef:
        int N,K
        Py_ssize_t i,j
        double result = 0.0
        double row_probability_sum, row_count_sum
        
    N = counts.shape[0]
    K = counts.shape[1]
    
    """
    accumulate log density contribution of the prior
    """    
    for j in range(K):
        result += alpha[j]*log(probability[j])
        
    """
    accumulate log density contribution of the likelihood
    """
    for i in range(N):
        row_count_sum = 0
        row_probability_sum = 0.0
        for j in range(K):
            if gsl_isnan(counts[i,j]):
                next
            else:
                row_probability_sum += probability[j]
                row_count_sum += counts[i,j]
                result += counts[i,j]*log(probability[j])
        result -= row_count_sum*log(row_probability_sum)
    
    return result
    
cpdef log_p_ratio(np.ndarray[np.float64_t, ndim=1] prop,
                  np.ndarray[np.float64_t, ndim=1] x_t,
                  np.ndarray[np.float64_t, ndim=1] alpha,
                  np.ndarray[np.float64_t, ndim=2] counts):
    """
    Calculates a value that is proportional to the ratio
    of posterior densities between the proposed sample
    and the previous sample and returns its logarithm.
    This value is the first of two terms needed to compute 
    the acceptance ratio for the Metropolis-Hastings sampler.
        - prop is a 1-dim vector of probabilities for the proposed sample
        - x_t is a 1-dim vector of probabilities for the previous sample
        - alpha is a 1-dim vector of concentration parameters from the dirichlet prior
        - counts is a 2-dim array of count values, including NaNs
    """
    return log_posterior(prop,alpha,counts) - log_posterior(x_t,alpha,counts)
    
cpdef log_q_ratio(np.ndarray[np.float64_t, ndim=1] prop,
                  np.ndarray[np.float64_t, ndim=1] x_t,
                  double beta):
    """
    Calculates a value that is proportional to the ratio 
    of proposal distribution densities between the proposed 
    sample and the previous sample and returns its logarithm.
    This value is the second of two terms needed to compute 
    the acceptance ratio for the Metropolis-Hastings sampler.
        - prop is a 1-dim vector of probabilities for the proposed sample
        - x_t is a 1-dim vector of probabilities for the previous sample
        - beta is a double that scales the concentration of the proposal distribution
    """
    cdef:
        int K = prop.shape[0]
        np.ndarray[double, ndim=1] prop_alpha = x_t * beta
        np.ndarray[double, ndim=1] x_t_alpha = prop * beta
        
    return gsl_ran_dirichlet_lnpdf(K,&x_t_alpha[0],&x_t[0]) - gsl_ran_dirichlet_lnpdf(K,&prop_alpha[0],&prop[0])
    
cpdef accept_ratio(np.ndarray[np.float64_t, ndim=1] prop,
                   np.ndarray[np.float64_t, ndim=1] x_t,
                   np.ndarray[np.float64_t, ndim=1] alpha,
                   np.ndarray[np.float64_t, ndim=2] counts,
                   double beta):
    """
    Calculates the acceptance ratio for the Metropolis-Hastings sampler.
    """
    return exp(log_p_ratio(prop,x_t,alpha,counts) + log_q_ratio(prop,x_t,beta))
    
@cython.boundscheck(False)
@cython.wraparound(False)   
def sampler(np.ndarray[np.float64_t, ndim=2] counts,
            np.ndarray[np.float64_t, ndim=1] prior_alpha,
            int repl = 10000,
            int burn = 50,
            double beta = 100.):
    """
    Executes a Markov Chain Monte Carlo simulation to estimate the distribution
    of class probabilities by using the Metropolis Hastings algorithm. New samples
    are proposed from a dirichlet distribution.
        - counts is a 2-dim array of count values, including NaNs
        - prior_alpha is a 1-dim vector of concentration parameters from the dirichlet prior
        - repl is an integer of the desired number of mcmc samples
        - burn is an integer representing the number of samples in the burn-in phase
        - beta is a float parameter that controls the dispersion of the proposal distribution:
            larger beta values correspond to greater dispersion of proposals
    """
    cdef:
        Py_ssize_t i
        int N = counts.shape[0]
        int K = counts.shape[1]
        np.ndarray[double, ndim=2] trace = np.empty((repl, K), dtype=np.float64)
        np.ndarray[double, ndim=1] prop = np.empty(K, dtype=np.float64)
        np.ndarray[double, ndim=1] proposal_alpha = np.empty(K, dtype=np.float64)
        np.ndarray[double, ndim=1] x_t = np.array([1./K]*K, dtype=np.float64)
        double accept = 0.0
        double ratio
        
    if N == 0:
        for i in range(repl):
            gsl_ran_dirichlet(r,K,&prior_alpha[0],&prop[0])
            trace[i,:] = prop
        accept += repl
    else:
        for i in range(repl+burn):
            proposal_alpha = x_t * beta
            gsl_ran_dirichlet(r,K,&proposal_alpha[0],&prop[0])
            ratio = accept_ratio(prop,x_t,prior_alpha,counts,beta)
            if ratio > 1.0 or ratio > gsl_ran_flat(u,0.0,1.0):
                x_t[:] = prop
                if i >= burn:
                    accept += 1.0
            if i >= burn:
                trace[i-burn,:] = x_t
    return {'trace':trace,'acceptance_rate':accept/repl}