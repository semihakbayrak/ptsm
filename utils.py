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

#Normalization for inference phase
def normalize_exp(log_P, axis=None):
    a = np.max(log_P, keepdims=True, axis=axis)
    P = normalize(np.exp(log_P - a), axis=axis)
    return P

def normalize(A, axis=None):
    Z = np.sum(A, axis=axis,keepdims=True)
    idx = np.where(Z == 0)
    Z[idx] = 1
    return A/Z 
