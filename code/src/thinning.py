import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn

from stein_thinning.thinning import thin_gf


def laplace_approximation(logpdf, x0):
    res = minimize(lambda x: -logpdf(x), x0, method='BFGS', options={'gtol': 1e-3})
    assert res.success
    return res.x, res.hess_inv


def gaussian_thin(sample, log_p, mean, cov, thinned_size, range_cap=200):
    log_q = mvn.logpdf(sample, mean=mean, cov=cov)
    gradient_q = -np.einsum('ij,kj->ki', np.linalg.inv(cov), sample - mean)
    return thin_gf(sample, log_p, log_q, gradient_q, thinned_size, range_cap=range_cap)
