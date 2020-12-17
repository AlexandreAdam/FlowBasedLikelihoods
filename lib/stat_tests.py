import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis as kurt

def KL(X, Y, d=37):
    # see eq 23   
    n = X.shape[0]
    m = Y.shape[0]
    
    nu_k = np.zeros(n)
    rho_k = np.zeros(n)
    
    for i in range(n):
        rho_k_dist = np.sum((X - X[i])**2,axis=1)**0.5
        rho_k_dist[rho_k_dist==0] = 999 # lazy cheat
        #bad_idx = rho_k_dist==999
        rho_k[i] = np.min(rho_k_dist)
        
        nu_k_dist = np.sum((Y - X[i])**2,axis=1)**0.5
        nu_k[i] = np.min(nu_k_dist)

    return d/n * np.sum(np.log(nu_k/rho_k)) + np.log(m/(n-1))


def standard_error_skew(n):
    return np.sqrt(6*n*(n-1)/((n-2)*(n+1)*(n+3)))

def standard_error_kurt(n):
    return np.sqrt(24*n*(n-1)**2/((n-3)*(n-2)*(n+3)*(n+5)))


def t_stats(X):

    SE_skew = standard_error_skew(X.shape[0])
    SE_kurt = standard_error_kurt(X.shape[0])

    skew_ = skew(X,axis=0)
    kurt_ = kurt(X,axis=0)
    
    t_skew = skew_/SE_skew
    t_kurt = kurt_/SE_kurt

    return t_skew,t_kurt
