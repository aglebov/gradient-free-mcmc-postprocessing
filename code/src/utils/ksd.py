from typing import Callable

import numpy as np

from stein_thinning.stein import ksd
from stein_thinning.thinning import _make_stein_integrand


def reindex_integrand(
        integrand: Callable[[np.ndarray, np.ndarray], np.ndarray],
        indices: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create integrand with index transformation"""
    def res(ind1, ind2):
        return integrand(indices[ind1], indices[ind2])
    return res


def calculate_ksd(
        sample: np.ndarray,
        gradient: np.ndarray,
        idx: np.ndarray
) -> np.ndarray:
    """Calculate cumulative KSD for the provided indices"""
    integrand = _make_stein_integrand(sample, gradient)
    integrand_restricted = reindex_integrand(integrand, idx)
    return ksd(integrand_restricted, idx.shape[0])
