#!/usr/bin/env python

# Rosso, Craig Moscato -- Shakespearer and other english renaissance authors as
# characterized by information theory complexity quantifiers, 2009, Physica A
# 388, Elsevier

import numpy as np
from numpy import log as ln


def uniform_like(P):
    """Returns a uniform distribution like P"""
    N = len(P)
    P_e = np.zeros_like(P)
    P_e.fill(1/N)
    return P_e


def shannon_entropy(P) -> np.float64:
    """Functional to calculate shannon entropy over P
       P is a probability distribution (all values >= 0)
       Returns a real value"""

    S_terms = (P * ln(P))
    S = np.sum(S_terms)
    return -S


def normalized_shannon_entropy(P) -> np.float64:
    """ Functional for entropic measure H_S(P)
        P is a probability distribution (all values >= 0, sum = 1)
        Returns a normalized real value [0, 1]"""
    S = shannon_entropy

    N = len(P)
    S_max = ln(N)
    H_s = S(P) / S_max
    return H_s


def jessen_divergence(P1, P2) -> np.float64:
    """ Calculates Jessen divergence between two probability distributions
        P1, P2 are probability distributions
        Returns a real value
    """
    S = shannon_entropy
    J_s =  S((P1 + P2) / 2) - S(P1)/2 - S(P2)/2
    return J_s

def Q_desequilibrium(P1, P2) -> np.float64:
    """ Calculates the Q desequilibrium between two distributions
    Returns a real value
    """
    # Definitions
    J_s = jessen_divergence
    N = len(P1)

    # Normalization constant
    Q_0 = -2 / ( (((N + 1) / N) * ln(N + 1)) - (2 * ln(2 * N)) + ln(N))
    Q_j = Q_0 * J_s(P1, P2)

    return Q_j

def mpr_complexity(P) -> np.float64:
    """ Functional that calculates Martin-Plastino-Rosso statistical complexity
        where:

        P is a probability distribution
        Pe is a uniform distribution over P

       Returns a real value 
    """
    # Possible states 
    Q_j = Q_desequilibrium
    H_s = normalized_shannon_entropy
    Pe = uniform_like(P)
    
    C_MPR_JS = Q_j(P, Pe) * H_s(P)
    
    return C_MPR_JS


def test():
    from scipy.stats import norm
    lower_bound = norm.ppf(0.01)
    upper_bound = norm.ppf(0.99)
    N = 100
    x, step = np.linspace(lower_bound, upper_bound, N, retstep=True)
    P = norm.pdf(x) 
    Ps = P * step
    print(mpr_complexity(Ps))

