dirmult
---------------------
Dirichlet-multinomial posterior models for truncated count data. For all observations, the likelihood term is defined on the full parameter vector of class probabilities but conditioned on a subset of class counts being zero a priori.
```python
import numpy as np
import dirmult.metropolis_hastings as mh
import dirmult.augmented_gibbs as ag
from numpy import nan

data = np.array([[nan,7,9,nan],
                 [3,2,nan,1],
                 [4,4,12,nan],
                 [nan,nan,9,3],
                 [10,nan,13,2]])
prior = np.array([5.,4.,7.,2.])
mcmc_samples = 100000
```
We can simulate the posterior using a Metropolis-Hastings MCMC sampler, but we must select a parameter beta that controls the dispersion of our proposals.
```python
mixing_beta = 35.
chain1 = mh.sampler(data,prior,mcmc_samples,beta=mixing_beta)
```
Or we can use a Gibbs sampling approach with augmented variables. We avoid having to choose hyperparameters and get great mixing by default. 
```python
chain2 = ag.sampler(data,prior,mcmc_samples)
```
Both approaches are implemented in Cython and cross-checked with pure Python implementations.
