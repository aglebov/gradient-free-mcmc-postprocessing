from functools import wraps
import logging
from pathlib import Path
import pickle
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)
cache_dir = Path('.')


def adjust_path(filepath, expected_type):
    if expected_type is np.ndarray:
        return filepath.parent / f'{filepath.stem}.npy'
    else:
        return filepath


def save_obj(filepath, obj):
    filepath = adjust_path(filepath, type(obj))
    if isinstance(obj, np.ndarray):
        np.save(filepath, obj, allow_pickle=False)
    elif isinstance(obj, pd.DataFrame):
        obj.to_csv(filepath)
    elif isinstance(obj, jax.Array):
        jnp.save(filepath, obj, allow_pickle=False)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)


def read_obj(filepath, expected_type: type):
    filepath = adjust_path(filepath, expected_type)
    if expected_type is np.ndarray:
        return np.load(filepath, allow_pickle=False)
    elif expected_type is pd.DataFrame:
        return pd.read_csv(filepath, index_col=0)
    elif expected_type is jax.Array:
        return jnp.load(filepath, allow_pickle=False)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def calculate_cached(
    calculation: Callable[[], np.ndarray],
    filepath: Path,
    recalculate: bool = False,
    save: bool = False,
    expected_type: type = np.ndarray,
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
        save the result of the calculation
    expected_type: type
        expected type of the result when retrieving from file

    Returns
    -------
    np.ndarray
        result of the calculation
    """
    filepath = adjust_path(filepath, expected_type)
    if recalculate or not filepath.exists():
        logger.debug('Recalculating')
        res = calculation()
        if save:
            logger.debug('Persisting calculation result')
            save_obj(filepath, res)
    else:
        logger.debug('Reading from disk cache')
        res = read_obj(filepath, expected_type)
    return res


def calculate_iterable_cached(
    calculation: Callable[[], np.ndarray],
    filepath_gen: Callable[[int], Path],
    n_items: Optional[int] = None,
    recalculate: bool = False,
    save: bool = False,
    expected_type: type = np.ndarray,
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
        save the result of the calculation
    expected_type: type
        expected type of the result when retrieving from file

    Returns
    -------
    np.ndarray
        result of the calculation
    """
    cache_available = all(
        adjust_path(filepath_gen(i), expected_type).exists() for i in range(n_items)
    )
    if recalculate or not cache_available:
        items = list(calculation())
        if save:
            for i, item in enumerate(items):
                save_obj(filepath_gen(i), item)
    else:
        items = [read_obj(filepath_gen(i), expected_type) for i in range(n_items)]
    return items


def map_cached(
    func: Callable[[Any], np.ndarray],
    items: Iterable[Any],
    filepath_gen: Callable[[Any], Path],
    recalculate: bool = False,
    save: bool = False,
    mapper: Callable[[Callable[[Any], np.ndarray], Iterable[Any]], Iterable[np.ndarray]] = map,
    expected_type: type = np.ndarray,
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
    expected_type: type
        expected type of the result when retrieving from file

    Returns
    -------
    list[Any]
        result of calculation for each item
    """
    def calculate(item):
        return calculate_cached(lambda: func(item), filepath_gen(item), recalculate, save)
    return list(mapper(calculate, items))


def cached(recalculate=False, persist=False, filename_gen=None):
    """Decorator adding caching on disk to functions"""
    if filename_gen is None:
        def filename_gen(func_name, *args, **kwargs):
            assert len(kwargs) == 0, 'kwargs not supported'
            return func_name + '_' + '_'.join([str(arg) for arg in args])
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            filepath = cache_dir / filename_gen(func.__name__, *args, **kwargs)
            logger.debug('Cache file path: %s', filepath)
            expected_type = func.__annotations__['return']
            calculation = lambda: func(*args, **kwargs)
            return calculate_cached(calculation, filepath, recalculate, persist, expected_type)
        return wrapper
    return decorator


def subscriptable(func):
    """Decorator to use indexing on a function"""
    class SubscriptableFuncion:
        def __init__(self, func):
            self.func = func
        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
        def __getitem__(self, key):
            return self.func(key)
        def __setitem__(self, key, value):
            raise NotImplementedError
        def __delitem__(self, key):
            raise NotImplementedError      
    return wraps(func)(SubscriptableFuncion(func))
