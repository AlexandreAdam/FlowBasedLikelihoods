"""
=============================================
Title: Transcovariance

Author(s): Alexandre Adam, Olivier Vincent

Last modified: December 16, 2020

Description: Compute the transcovariance matrix to get the 
    mean integrated square error.
=============================================
"""
import numpy as np
from scipy.stats import gaussian_kde, norm
from scipy.special import binom

ell = np.logspace(np.log10(500), np.log10(5000), 36) 

def zca_whiten(X):
    """
    Center the data and rotate the columns to diagonalize and 
    unitize the covariance matrix with Singular Value Decomposition
    of the covariance matrix
    """
    cov = np.cov(X.T)
    U, Sigma, V = np.linalg.svd(cov)
    D = np.diag(np.sqrt(1/Sigma)) # square root inverse of singular value matrix
    W = U @ D @ V # rotation matrix
    centered = X - X.mean(axis=0)
    X_white = np.einsum("ij, ...j -> ...i", W, centered)
    return X_white

def pca_whiten(X):
    """
    Center the data and rotate the columns to diagonalize and 
    unitize the covariance matrix with eigenvalue decomposition 
    of the covariance matrix
    """
    cov = np.cov(X.T)
    Sigma, U = np.linalv.eig(cov)
    D = np.diag(np.sqrt(1/Sigma)) # square root inverse of singular value matrix
    W = D @ U.T # rotation matrix
    centered = X - X.mean(axis=0)
    X_white = np.einsum("ij, ...j -> ...i", W, centered)
    return X_white

def gaussian_pdf(x, mean, var):
    return 1/np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (x - mean)**2/var)

def pairwise_bins(X, bins=10):
    """
    Pair bins as described in III.B for tabular data (power spectra)

    bins: The number of poit as which to evaluate the KDE and N(0, 2). Large number 
        of bins makes this function slow.
    """
    d = X.shape[1]
    x = np.linspace(-4, 4, bins) # points where the KDE and N(0, 2) are evaluated

    total_pairs = int(binom(d, 2))

    # Get paired bins (eq 18)
    H_b = [] # binned probbaility
    l = 0 
    for i in range(d):
        for j in range(i+1, d):
            pairs = zca_whiten(X[:, (i, j)])

            # fit kde on the sum
            K = gaussian_kde(pairs[:, 0] + pairs[:, 1])
            H_b.append(K.pdf(x))

            print(f"\r transcovariance pair done: {l}/{total_pairs-1:d}", end="", flush=True)
            l += 1
    H_b = np.array(H_b).T
    return H_b, x

def integrated_square_error(X, bins=10):
    d = X.shape[1]
    H_b, x = pairwise_bins(X, bins=bins)

    support = np.tile(x, (H_b.shape[1], 1)).T

    # mean integrated square error
    mise = ((H_b - gaussian_pdf(support, mean=0, var=2))**2).mean(axis=0)

    # standard deviation integrated square error
    stdise = ((H_b - gaussian_pdf(support, mean=0, var=2))**2).std(axis=0)

    return mise, stdise

def transcovariance_matrix(X, bins=10):
    d = X.shape[1]
    mise, stdise = integrated_square_error(X, bins=bins)

    # fill upper and lower triangle of the power spectrum bin matrix
    _transcovariance_matrix = np.zeros((d, d))
    upper_tri_indices = np.triu_indices(d, k=1) # k=1: ommit diagonal

    _transcovariance_matrix[upper_tri_indices] = mise # upper triangle

    _transcovariance_matrix.T[upper_tri_indices] = mise # lower triangle

    epsilon_plus = _transcovariance_matrix.sum(axis=1)

    _transcovariance_error = np.zeros((d, d))
    _transcovariance_error[upper_tri_indices] = stdise
    _transcovariance_error.T[upper_tri_indices] = stdise

    epsilon_var = np.sqrt((_transcovariance_error**2).sum(axis=1)) # propagation of error rule

    return epsilon_plus, epsilon_var


def transcovariance_plot(X, bins=10, ax=None, **kwargs):
    eps_plus, eps_var = transcovariance_matrix(X, bins=bins)

    # null hypothesis
    null_hypothesis = np.random.normal(size=(1000, X.shape[1]))
    eps_null, eps_var_null = transcovariance_matrix(null_hypothesis, bins=bins)

    if ax is None:
        figsize = kwargs.pop("figsize", (4, 4))
        f, ax = plt.subplots(figsize=figsize)
    color = kwargs.pop("color", "b")
    ax.plot(ell, eps_plus, ls="-", color=color, label="Data")
    ax.plot(ell, eps_null, ls="-", color="gray", label="Gaussian")

    ax.fill_between(ell, eps_plus + eps_var, eps_plus - eps_var, color=color, alpha=0.5)
    ax.fill_between(ell, eps_null + eps_var_null, eps_null - eps_var_null, color="gray", alpha=0.5)



if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("../powerspec.csv")
    ell_bins = [f"ell{i}" for i in range(37)]
    X = data[ell_bins].to_numpy()
    transcovariance_matrix(X[:100])


