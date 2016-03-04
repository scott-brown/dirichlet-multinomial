# -*- coding: utf-8 -*-
import numpy as np
from dirmult.tests.utils import *
from dirmult.tests.implementation import *

if __name__ == "__main__":
    ### SET PARAMETERS
    n = 82
    k = 6
    n_nan = 60
    true_alpha = np.random.uniform(3,20,k) 
    prior = np.array([10.]*k)
    
    ### GENERATE FAKE DATA
    counts = generate_fake_counts(true_alpha, (15,25), n, n_nan)
    
    a = MetropolisHastingsTest(counts, prior, 300000)
    a.plots()
    
    b = AugmentedGibbsTest(counts, prior, 10000)
    b.plots()
    
    c = MethodCrosscheck(counts, prior, 350000, 50, 1100.)
    c.mh_chain['acceptance_rate']
    c.plots()