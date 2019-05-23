#utility functions are mostly adapted from Ali Taylan Cemgil's notes and codes
import numpy as np

#random number generator for given probabilities with inverse transform sampling
def randgen(prob, val_list=[]):
    N = len(prob)
    values = []
    if len(val_list) == 0:
        values = range(N)
    else:
        values = val_list
    r = np.random.rand()
    cdf = np.cumsum(prob)
    for n in range(N):
        if r <= cdf[n]:
            return values[n]

#log-sum-exp trick to overcome the vanishing problem
def log_sum_exp(M,log_v):
    mx = np.max(log_v)
    v = np.exp(log_v-mx)
    log_Mv = np.log(np.dot(M,v)) + mx
    return log_Mv

#Normalization for (state*times) matrix with log probabilities
def normalize_exp(log_x):
    return np.exp(log_x - np.max(log_x,axis=0))/np.exp(log_x - np.max(log_x,axis=0)).sum(axis=0).reshape(1,log_x.shape[1])

