from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np


def calculate_cached(
    calculation: Callable[[], np.ndarray],
    filepath: Path,
    recalculate: bool = False,
    save: bool = False,
) -> np.ndarray:
    """Perform expensive calculation or retrieve cached result

    Parameters
    ----------
    calculation: Callable[[], np.ndarray]
        calculation to perform
    filepath: Path
        path to cache the result of the calculation
    recalculate: bool
        recalculate the results from scratch if True, use the cached result if False
    save: bool
        save the result of the calculation (only has effect if `recalculate` is True)

    Returns
    -------
    np.ndarray
        result of the calculation
    """
    if recalculate:
        res = calculation()
        if save:
            np.savetxt(filepath, res, delimiter=',')
    else:
        res = np.genfromtxt(filepath, delimiter=',')
    return res


def calculate_iterable_cached(
    calculation: Callable[[], np.ndarray],
    filepath_gen: Callable[[int], Path],
    n_items: Optional[int] = None,
    recalculate: bool = False,
    save: bool = False,
) -> np.ndarray:
    """Perform expensive calculation or retrieve cached result

    The calculated items are stored in separate files.

    Parameters
    ----------
    calculation: Callable[[], np.ndarray]
        calculation to perform
    filepath_gen: Callable[[int], Path]
        function returning the path for cache file
    n_items: Optional[int]
        number of cached items to retrieve
    recalculate: bool
        recalculate the results from scratch if True, use the cached result if False
    save: bool
        save the result of the calculation (only has effect if `recalculate` is True)

    Returns
    -------
    np.ndarray
        result of the calculation
    """
    if recalculate:
        items = list(calculation())
        if save:
            for i, item in enumerate(items):
                np.savetxt(filepath_gen(i), item, delimiter=',')
    else:
        items = [np.genfromtxt(filepath_gen(i), delimiter=',') for i in range(n_items)]
    return items


def map_cached(
    func: Callable[[Any], np.ndarray],
    items: Iterable[Any],
    filepath_gen: Callable[[Any], Path],
    recalculate: bool = False,
    save: bool = False,
    mapper: Callable[[Callable[[Any], np.ndarray], Iterable[Any]], Iterable[np.ndarray]] = map,
):
    """Perform expensive calculation for provided items or read stored results

    Parameters
    ----------
    func: Callable[[Any], np.ndarray]
        function to apply to every item
    items: Iterable[Any]
        items to map
    filepath_gen: Callable[[Any], Path]
        function returning the path for cache file
    recalculate: bool
        recalculate the results from scratch if True, use the cached result if False
    save: bool
        save the result of the calculation (only has effect if `recalculate` is True)
    mapper: Callable[[Callable[[Any], np.ndarray], Iterable[Any]], Iterable[np.ndarray]]
        mapper function to use (defaults to the standard `map` function)
    """
    def calculate(item):
        return calculate_cached(lambda: func(item), filepath_gen(item), recalculate, save)
    return list(mapper(calculate, items))
