# -*- coding: utf-8 -*-
import numpy as np
import dirmult.metropolis_hastings as mh
import dirmult.augmented_gibbs as ag
import matplotlib.pyplot as plt
from scipy.stats import dirichlet, geom
from numpy.random import multinomial

class MetropolisHastingsTest(object):  
    def __init__(self, counts, alpha, repl = 10000, burn = 50, beta = 500.):
        """
        counts is a 2-dim array of class count data
        that may or may not contain NaN values:
            - each row is a trial
            - each column corresponds to a unique class
        alpha is a 1-dim array of concentration values               
        """ 
        self.counts = counts
        self.alpha = alpha
        self.repl = repl
        self.burn = burn
        self.beta = beta
        self.N = self.counts.shape[0]
        self.K = self.counts.shape[1]
        self.python_chain = self.python_sampler()
        self.cython_chain = self.cython_sampler()
        
    def logp(self, pvals):
        """
        pvals is a 1-dim array of class probabilities
        """
        pmat = np.tile(pvals,(self.N,1))
        pmat[np.isnan(self.counts)] = np.nan
        return np.sum(self.alpha*np.log(pvals)) + np.nansum(self.counts*np.log(pmat)) - np.sum(np.nansum(self.counts,1)*np.log(np.nansum(pmat,1)))
   
    def python_sampler(self):   
        if self.N == 0:
            trace = dirichlet.rvs(self.alpha, size = self.repl)
            accept = float(self.repl)
        else:
            accept = 0.
            x_t = np.array([1./self.K]*self.K)
            trace = np.empty((self.repl,self.K))
            for i in xrange(self.repl+self.burn):
                prop = dirichlet.rvs(x_t*self.beta).reshape(self.K)
                r = np.exp(self.logp(prop) - self.logp(x_t) + dirichlet.logpdf(x_t,prop*self.beta) - dirichlet.logpdf(prop,x_t*self.beta))
                if r > 1. or np.random.rand() < r:
                    x_t = prop
                    if i >= self.burn:
                        accept += 1.                    
                if i >= self.burn:
                    trace[i-self.burn] = x_t
        return {'trace':trace,'acceptance_rate':accept/self.repl}
        
    def cython_sampler(self):
        return mh.sampler(self.counts, self.alpha, self.repl, self.burn, self.beta)

    def plots(self, numbins = 40):
        for i in xrange(self.K):
            bins = np.linspace(min(self.python_chain['trace'][:,i].min(),
                                   self.cython_chain['trace'][:,i].min()),
                               max(self.python_chain['trace'][:,i].max(),
                                   self.cython_chain['trace'][:,i].max()),numbins)
            plt.figure(figsize=(16,8))
            plt.hist(self.python_chain['trace'][:,i],bins=bins,alpha=0.5,label='Python',normed=True)
            plt.hist(self.cython_chain['trace'][:,i],bins=bins,alpha=0.5,label='Cython',normed=True)
            plt.legend(loc='upper right',fontsize=20)
            plt.show()
            
class AugmentedGibbsTest(object):  
    def __init__(self, counts, alpha, repl = 10000, burn = 50):
        """
        counts is a 2-dim array of class count data
        that may or may not contain NaN values:
            - each row is a trial
            - each column corresponds to a unique class
        alpha is a 1-dim array of concentration values               
        """ 
        self.counts = counts
        self.alpha = alpha
        self.repl = repl
        self.burn = burn
        self.N = self.counts.shape[0]
        self.K = self.counts.shape[1]
        self.python_chain = self.python_sampler()
        self.cython_chain = self.cython_sampler()
        
    def python_sampler(self):
        col_counts = np.nansum(self.counts,0)
        row_counts = np.nansum(self.counts,1)
        row_idx_missing = np.arange(self.N)[np.isnan(self.counts).any(1)]
        col_idx_missing = [np.where(row)[0] for row in np.isnan(self.counts)[row_idx_missing]]
        p = np.array([1./self.K]*self.K)
        trace = np.empty((self.repl,self.K))
        for ii in xrange(self.repl+self.burn):
            indv_augmented_counts = np.zeros(self.K)
            for (i,j) in zip(row_idx_missing,col_idx_missing):
                total_augmented_count = geom.rvs(1-p[j].sum(), loc = -1, size = row_counts[i]).sum()
                indv_augmented_counts[j] += multinomial(n = total_augmented_count, pvals = (p[j] / p[j].sum()))
            p = dirichlet.rvs(alpha = self.alpha + col_counts + indv_augmented_counts).reshape(self.K)
            if ii >= self.burn:
                trace[ii-self.burn] = p
        return trace        
        
    def cython_sampler(self):
        return ag.sampler(self.counts, self.alpha, self.repl, self.burn)

    def plots(self, numbins = 40):
        for i in xrange(self.K):
            bins = np.linspace(min(self.python_chain[:,i].min(),
                                   self.cython_chain[:,i].min()),
                               max(self.python_chain[:,i].max(),
                                   self.cython_chain[:,i].max()),numbins)
            plt.figure(figsize=(16,8))
            plt.hist(self.python_chain[:,i],bins=bins,alpha=0.5,label='Python',normed=True)
            plt.hist(self.cython_chain[:,i],bins=bins,alpha=0.5,label='Cython',normed=True)
            plt.legend(loc='upper right',fontsize=20)
            plt.show()
            
class MethodCrosscheck(object):  
    def __init__(self, counts, alpha, repl = 10000, burn = 50, beta = 500.):
        """
        counts is a 2-dim array of class count data
        that may or may not contain NaN values:
            - each row is a trial
            - each column corresponds to a unique class
        alpha is a 1-dim array of concentration values               
        """ 
        self.counts = counts
        self.alpha = alpha
        self.repl = repl
        self.burn = burn
        self.beta = beta
        self.N = self.counts.shape[0]
        self.K = self.counts.shape[1]
        self.mh_chain = self.mh_sampler()
        self.ag_chain = self.ag_sampler()
        
    def mh_sampler(self):
        return mh.sampler(self.counts, self.alpha, self.repl, self.burn, self.beta)        
        
    def ag_sampler(self):
        return ag.sampler(self.counts, self.alpha, self.repl, self.burn)

    def plots(self, numbins = 40):
        for i in xrange(self.K):
            bins = np.linspace(min(self.mh_chain['trace'][:,i].min(),
                                   self.ag_chain[:,i].min()),
                               max(self.mh_chain['trace'][:,i].max(),
                                   self.ag_chain[:,i].max()),numbins)
            plt.figure(figsize=(16,8))
            plt.hist(self.mh_chain['trace'][:,i],bins=bins,alpha=0.5,label='Metropolis-Hastings',normed=True)
            plt.hist(self.ag_chain[:,i],bins=bins,alpha=0.5,label='Augmented Gibbs',normed=True)
            plt.legend(loc='upper right',fontsize=20)
            plt.show()