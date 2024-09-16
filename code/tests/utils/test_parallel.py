import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

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


def _map_sequential(func, items):
    return list(map(func, items))

def test_apply_along_axis_1d_result():
    rng = np.random.default_rng(12345)
    arr = rng.random((100, 100))

    assert_almost_equal(
        apply_along_axis_parallel(np.sum, 0, arr, 10, _map_sequential),
        np.apply_along_axis(np.sum, 0, arr)
    )
    assert_almost_equal(
        apply_along_axis_parallel(np.sum, 1, arr, 10, _map_sequential),
        np.apply_along_axis(np.sum, 1, arr)
    )


def test_apply_along_axis_2d_result():
    rng = np.random.default_rng(12345)
    arr = rng.random((100, 100))

    def first_two(x):
        return x[:2]

    assert_almost_equal(
        apply_along_axis_parallel(first_two, 0, arr, 10, _map_sequential),
        np.apply_along_axis(first_two, 0, arr)
    )
    assert_almost_equal(
        apply_along_axis_parallel(first_two, 1, arr, 10, _map_sequential),
        np.apply_along_axis(first_two, 1, arr)
    )


def test_apply_along_axis_aggregate():
    arr = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])

    func = lambda x: x ** 2
    aggregate = lambda x: np.sum(x, axis=0, keepdims=True)

    assert_array_equal(
        apply_along_axis_parallel(func, 1, arr, 2, _map_sequential, aggregate=aggregate),
        np.sum(arr ** 2, axis=0, keepdims=True),
    )
