from functools import lru_cache, wraps
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)
cache_dir = Path('.')


def get_path(entry_name: str, expected_type: type) -> Path:
    """Construct path for persisting an entry of a given type

    Parameters
    ----------
    entry_name: str
        name of the entry
    expected_type: type
        type of the entry

    Returns
    -------
    Path
        path to the file on disk
    """
    if expected_type is np.ndarray:
        return cache_dir / f'{entry_name}.npy'
    else:
        return cache_dir / entry_name


def save_obj(entry_name: str, obj: Any):
    """Persist entry on disk

    Parameters
    ----------
    entry_name: str
        name of the entry
    obj: Any
        entry object to persist
    """
    filepath = get_path(entry_name, type(obj))
    logger.debug('Writing %s', filepath)
    if isinstance(obj, np.ndarray):
        np.save(filepath, obj, allow_pickle=False)
    elif isinstance(obj, pd.DataFrame):
        obj.to_csv(filepath)
    elif isinstance(obj, jax.Array):
        jnp.save(filepath, obj, allow_pickle=False)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)


def read_obj(entry_name: str, expected_type: type) -> Any:
    """Read persisted entry from disk

    Parameters
    ----------
    entry_name: str
        name of the entry
    expected_type: type
        type of the entry

    Returns
    -------
    Any
        entry object read from disk
    """
    filepath = get_path(entry_name, expected_type)
    logger.debug('Reading %s', filepath)
    if expected_type is np.ndarray:
        return np.load(filepath, allow_pickle=False)
    elif expected_type is pd.DataFrame:
        return pd.read_csv(filepath, index_col=0)
    elif expected_type is jax.Array:
        return jnp.load(filepath, allow_pickle=False)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def calculate_cached[T](
    calculation: Callable[[], T],
    entry_name: str,
    recalculate: bool = False,
    save: bool = False,
    expected_type: type = np.ndarray,
) -> T:
    """Perform expensive calculation or retrieve cached result

    Parameters
    ----------
    calculation: Callable[[], T]
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
    T
        result of the calculation
    """
    filepath = get_path(entry_name, expected_type)
    if recalculate or not filepath.exists():
        logger.info('Recalculating: %s', entry_name)
        start_time = time.time()
        res = calculation()
        end_time = time.time()
        logger.info('Calculation time for %s: %f s', entry_name, end_time - start_time)
        if save:
            logger.debug(f'Persisting calculation result: {entry_name}')
            save_obj(entry_name, res)
    else:
        logger.debug('Reading from disk cache: %s', entry_name)
        res = read_obj(entry_name, expected_type)
    return res


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


def cached[T](
        recalculate: bool = False,
        persist: bool = True,
        filename_gen: Optional[Callable[[str, int], str]] = None,
        memory_cache_maxsize: Optional[int] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator adding caching on disk to functions

    Parameters
    ----------
    recalculate: bool
        if True, perform the calculation from scratch ignoring any cached results
    persist: bool
        if True, persist recalculated result on disk (only has effect if `recalculate` is `True`)
    filename_gen: Optional[Callable[[str, int], str]]
        function returning file name to use when persisting results on disk
    memory_cache_maxsize: Optional[int]
        maximum number of results to retain in memory cache

    Returns
    -------
    Callable[[Callable[..., T]], Callable[..., T]]
        decorator function
    """
    if filename_gen is None:
        def filename_gen(func_name, *args, **kwargs):
            assert len(kwargs) == 0, 'kwargs not supported'
            if len(args) > 0:
                return func_name + '_' + '_'.join([str(arg) for arg in args])
            else:
                return func_name
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            entry_name = filename_gen(func.__name__, *args, **kwargs)
            expected_type = func.__annotations__['return']
            calculation = lambda: func(*args, **kwargs)
            return calculate_cached(calculation, entry_name, recalculate, persist, expected_type)
        return lru_cache(memory_cache_maxsize)(wrapper)
    return decorator


def cached_batch[T](
        *,
        item_type: type,
        recalculate: bool = False,
        persist: bool = True,
        filename_gen: Optional[Callable[[str, int], str]] = None,
        memory_cache_maxsize: Optional[int] = None,
) -> Callable[[Callable[[], Iterable[T]]], Callable[[int], T]]:
    """Decorator adding caching on disk to functions that batch calculation for multiple items

    Parameters
    ----------
    item_type: type
        type of items in the batch
    recalculate: bool
        if True, perform the calculation from scratch ignoring any cached results
    persist: bool
        if True, persist recalculated result on disk (only has effect if `recalculate` is `True`)
    filename_gen: Optional[Callable[[str, int], str]]
        function returning file name to use when persisting results on disk
    memory_cache_maxsize: Optional[int]
        maximum number of results to retain in memory cache

    Returns
    -------
    Callable[[Callable[[], Iterable[T]]], Callable[[int], T]]
        decorator function
    """
    if filename_gen is None:
        def filename_gen(func_name, i):
            return f'{func_name}_{i}'
    def decorator(func):
        def do_recalculate(persist: bool = True):
            logger.info('Recalculating batch: %s', func.__name__)
            start_time = time.time()
            batch = func()
            end_time = time.time()
            logger.info('Calculation time for %s: %f s', func.__name__, end_time - start_time)
            if persist:
                for j, item in enumerate(batch):
                    entry_name = filename_gen(func.__name__, j)
                    logger.debug(f'Persisting calculation result: {entry_name}')
                    save_obj(entry_name, item)
            return batch

        @wraps(func)
        def wrapper(i):
            entry_name = filename_gen(func.__name__, i)
            filepath = get_path(entry_name, item_type)
            if recalculate or not filepath.exists():
                res = do_recalculate(persist)[i]
            else:
                logger.debug('Reading from disk cache: %s', entry_name)
                res = read_obj(entry_name, item_type)
            return res

        wrapper.recalculate = do_recalculate

        return lru_cache(memory_cache_maxsize)(wrapper)
    return decorator


def subscriptable(
        func: Optional[Callable] = None,
        *,
        n: Optional[int] = None,
):
    """Decorator to use indexing on a function

    For a function taking one argument:
    ```
    @subscriptable
    def func(i):
        ...
    ```
    the decorator returns an object that can indexed:
    ```
    func[5]
    ```
    which translates into an invocation
    ```
    func(5)
    ```

    This decorator can be used either with the `n` argument or without arguments.
    """
    if func is None:
        # decorator is called with the size parameter
        assert n is not None, 'n must be provided'
        def decorator(func):
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
                def __len__(self):
                    return n
                def __iter__(self):
                    for i in range(n):
                        yield self(i)
            return wraps(func)(SubscriptableFuncion(func))
        return decorator
    else:
        # decorator is called without the size parameter
        assert n is None
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
            def __len__(self):
                return n
        return wraps(func)(SubscriptableFuncion(func))
