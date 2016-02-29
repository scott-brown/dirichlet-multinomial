# -*- coding: utf-8 -*-
import numpy as np

def sampler(counts, prior_alpha, repl = 10000):
    if np.isnan(counts).any():
        raise ValueError("Conjugate Monte Carlo sampler may only model fully non-missing count data.")
    else:
        return np.random.dirichlet(counts.sum(0) + prior_alpha,repl)