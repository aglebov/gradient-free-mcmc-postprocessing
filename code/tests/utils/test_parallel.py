import numpy as np
from numpy.testing import assert_almost_equal

from utils.parallel import map_parallel, apply_along_axis_parallel


class SequentialClient:
    def submit(self, func, item):
        return func(item)

    def gather(self, futures):
        return futures


def test_map_parallel():
    client = SequentialClient()
    arr = np.arange(10)
    func = lambda x: x * 2
    assert np.all(
        np.array(map_parallel(func, arr, client)) == np.fromiter(map(func, arr), int)
    )


def test_apply_along_axis_1d_result():
    client = SequentialClient()
    rng = np.random.default_rng(12345)
    arr = rng.random((100, 100))

    assert_almost_equal(
        apply_along_axis_parallel(np.sum, 0, arr, 10, client),
        np.apply_along_axis(np.sum, 0, arr)
    )
    assert_almost_equal(
        apply_along_axis_parallel(np.sum, 1, arr, 10, client),
        np.apply_along_axis(np.sum, 1, arr)
    )


def test_apply_along_axis_2d_result():
    client = SequentialClient()
    rng = np.random.default_rng(12345)
    arr = rng.random((100, 100))

    def first_two(x):
        return x[:2]

    assert_almost_equal(
        apply_along_axis_parallel(first_two, 0, arr, 10, client),
        np.apply_along_axis(first_two, 0, arr)
    )
    assert_almost_equal(
        apply_along_axis_parallel(first_two, 1, arr, 10, client),
        np.apply_along_axis(first_two, 1, arr)
    )
