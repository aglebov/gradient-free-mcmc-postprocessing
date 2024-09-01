import numpy as np

from stein_thinning.stein import kmat

from utils.ksd import reindex_integrand


def test_reindex_integrand():
    mat = np.array([
        [1, 2, 3,  4,  5],
        [2, 6, 7,  8,  9],
        [3, 7, 10, 11, 12],
        [4, 8, 11, 13, 14],
        [5, 9, 12, 14, 15],
    ])
    def integrand(ind1, ind2):
        return mat[ind1, ind2]
    indices = np.array([2, 0, 4, 3, 1])
    new_integrand = reindex_integrand(integrand, indices)
    res = kmat(new_integrand, mat.shape[0])
    expected = np.array([
        [10, 3, 12, 11, 7],
        [3,  1, 5,  4,  2],
        [12, 5, 15, 14, 9],
        [11, 4, 14, 13, 8],
        [7,  2, 9,  8,  6],
    ])
    np.testing.assert_array_equal(res, expected)
