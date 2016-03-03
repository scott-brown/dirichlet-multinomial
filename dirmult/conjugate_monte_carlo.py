# -*- coding: utf-8 -*-
import numpy as np

def sampler(counts, prior_alpha, repl = 10000):
    """
    Executes a Monte Carlo simulation to estimate the distributions
    of class probabilities based on the dirichlet-multinomial conjugate
    model with non-missing count data.
        - counts is a 2-dim array of count values, including NaNs
        - prior_alpha is a 1-dim vector of concentration parameters from the dirichlet prior
        - repl is an integer of the desired number of mcmc samples
    """
    if np.isnan(counts).any():
        raise ValueError("Conjugate Monte Carlo sampler may only model fully non-missing count data.")
    else:
        return np.random.dirichlet(counts.sum(0) + prior_alpha,repl)