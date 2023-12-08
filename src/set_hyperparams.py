import numpy as np
from scipy.stats import norm


def cdf(x):
    return norm.cdf(x)


def cauchy_prob_curve(bin_param_r, d):
    x = d / bin_param_r
    return 2 * np.arctan(1 / x) / np.pi - x * np.log(1 + 1 / x ^ 2) / np.pi


def gauss_prob_curve(bin_param_r, d):
    x = d / bin_param_r
    return (
        1
        - 2 * cdf(-1 / x)
        - 2 * x * (1 - np.exp(-1 / (2 * x ^ 2))) / (np.sqrt(2 * np.pi))
    )


def compute_params(n, p_norm: int = 1, bin_param_r=1.0, r1=1.0, c=2):
    r2 = r1 * c
    if p_norm == 1:
        p1 = cauchy_prob_curve(bin_param_r, r1)
        p2 = cauchy_prob_curve(bin_param_r, r2)
    elif p_norm == 2:
        p1 = gauss_prob_curve(bin_param_r, r1)
        p2 = gauss_prob_curve(bin_param_r, r2)

    gamma = np.log(1 / p1) / np.log(1 / p2)
    l = 2 * n ^ gamma
    k = np.log(n) / np.log(1 / p2)
    return p1, p2, l, k, gamma
