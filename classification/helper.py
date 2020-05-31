''' helper functions for the model
'''
import numpy as np


def moment_matching_beta(xbar, s2):
    """ Computes the parameters (for moment matching)
    for a beta distribution from a given mean and variance
    INPUTS:
    ------------------------------------------------------------
    xbar                    : mean
    s2                      : desired variance
    """
    a = xbar * (xbar * (1. - xbar) / s2 - 1.0)
    b = a * (1. - xbar) / xbar
    return([a,b])
  
def MAP_beta(X1,X0, alpha_0, beta_0):
    """ the Maximum A Posteriori estimate for the beta
    distribution
    INPUTS:
    ------------------------------------------------------------
    X1                       : scores for the "1"s observed values
    X0                       : scores for the "0"s observed values
    alpha_0, beta_0          : hyper parameters for the beta distribution
    """
    mu_map = (np.sum(X1) + alpha_0 - 1) / (np.sum(X0) + beta_0 - 2)
    return mu_map