# -*- coding: utf-8 -*-
import numpy as np

def sampler(counts, prior_alpha, repl = 10000):
    return np.random.dirichlet(counts.sum(0) + prior_alpha,repl)