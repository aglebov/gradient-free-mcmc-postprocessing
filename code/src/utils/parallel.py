from typing import Any, Callable, Iterable, Protocol, Sequence, Tuple, TypeVar

from joblib import Parallel, delayed

import numpy as np


T = TypeVar('T')
U = TypeVar('U')
Future = TypeVar('Future')


class Client(Protocol):
    def submit(self, func: Callable[[T], U], item: T) -> Future: ...
    def gather(self, futures: Sequence[Future]) -> Sequence[U]: ...


def map_parallel(
        func: Callable[[T], U],
        iterable: Iterable[T],
        client: Client,
) -> Iterable[U]:
    """Apply the function to each element of the iterable in parallel

    Parameters
    ----------
    func: Callable[[T], U]
        the function to be applied to each element of the iterable
    iterable: Iterable[T]
        an iterable whose elements need to be processed
    client: Client
        a client object that supports submitting tasks for execution and gathering results

    Returns
    -------
    Iterable[U]
        the results of applying the function to each element of the input iterable in order"""
    futures = [client.submit(func, item) for item in iterable]
    return client.gather(futures)


def get_map_parallel(client):
    def map_parallel_client(func, iterable):
        return map_parallel(func, iterable, client)
    return map_parallel_client


def get_map_parallel_joblib(n_jobs):
    parallel = Parallel(n_jobs=n_jobs)
    def map_parallel(func, it):
        return parallel(delayed(func)(i) for i in it)
    return map_parallel


def apply_along_axis_parallel(
        func1d: Callable[[np.ndarray], np.ndarray],
        axis: int,
        arr: np.ndarray,
        chunk_size: int,
        client: Client,
        args: Tuple[Any] = tuple(),
) -> np.ndarray:
    """Apply function along the given axis and parallelise in chunks

    The function is equivalent to ``numpy.apply_along_axis``, but performs the processing
    in parallel.

    Parameters
    ----------
    func1d: Callable[[np.ndarray], np.ndarray]
        this function is applied to 1-D slices of ``arr`` along the specified axis
    axis: int
        axis along which ``arr`` is sliced
    arr: np.ndarray
        input array
    chunk_size: int
        size of the chunk to submit for parallel processing
    client: Client
        a client object that supports submitting tasks for execution and gathering results
    args: Tuple[Any]
        additional arguments to ``func1d``

    Returns
    -------
    np.ndarray
        the output array
    """
    assert axis in (0, 1)

    # define a generator to obtain chunks of the matrix
    n_chunks = (arr.shape[1 - axis] - 1) // chunk_size + 1
    def chunker():
        for i in range(n_chunks):
            index_slice = slice(i * chunk_size, (i + 1) * chunk_size)
            if axis == 1:
                chunk = arr[index_slice, :]
            else:
                chunk = arr[:, index_slice]
            yield chunk

    # define the function to apply to chunks
    def func(chunk):
        return np.apply_along_axis(func1d, axis, chunk, *args)

    # apply function to chunks in parallel
    results = map_parallel(func, chunker(), client)

    # concatenate the results
    max_ndim = max((result.ndim for result in results))
    if max_ndim > 1:
        return np.concatenate(results, axis=1 - axis)
    else:
        return np.concatenate(results)


def parallelise_for_unique(func, sample, client, row_chunk_size=200):
    """Calculate gradients for samples"""
    # we can save time by calculating gradients for unique samples only
    unique_samples, inverse_index = np.unique(sample, axis=0, return_inverse=True)
    res = apply_along_axis_parallel(func, 1, unique_samples, row_chunk_size, client)
    return res[inverse_index]
