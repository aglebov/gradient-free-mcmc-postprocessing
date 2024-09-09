from functools import lru_cache, wraps
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

import cachetools

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


cache = cachetools.LRUCache(maxsize=32)


class CacheFunc:

    def __init__(
            self,
            func: Callable[[], Iterable],
            item_type: type,
            recalculate: bool,
            persist: bool,
            read_only: bool,
            batch: bool,
            batch_size: Optional[int] = None,
            filename_gen: Optional[Callable[[str, int], str]] = None,
    ):
        assert not (recalculate and read_only), 'Cannot use recalculate and read_only together'
        assert not batch or item_type is not None, 'Item type must be provided in batch mode'
        self._func = func
        self._item_type = item_type or func.__annotations__['return']
        self._recalculate = recalculate
        self._persist = persist
        self._read_only = read_only
        self._batch = batch
        self._batch_size = batch_size
        self._filename_gen = filename_gen
        

    def recalculate(self, *args, persist: bool = True):
        if self._batch:
            if len(args) != 1:
                raise ValueError('index argument expected in the batch mode')
            i = args[0]
            if self._batch_size is not None and (i < 0 or i >= self._batch_size):
                raise IndexError('item index out of range')
            logger.info('Recalculating batch: %s', self._func.__name__)
            start_time = time.time()
            batch = self._func()
            end_time = time.time()
            logger.info('Calculation time for %s: %f s', self._func.__name__, end_time - start_time)
            if persist:
                for j, item in enumerate(batch):
                    entry_name = self._filename_gen(self._func.__name__, j)
                    logger.debug(f'Persisting calculation result: {entry_name}')
                    save_obj(entry_name, item)
            return batch[i]
        else:
            entry_name = self._filename_gen(self._func.__name__, *args)
            logger.info('Recalculating: %s', entry_name)
            start_time = time.time()
            res = self._func(*args)
            end_time = time.time()
            logger.info('Calculation time for %s: %f s', entry_name, end_time - start_time)
            if persist:
                logger.debug(f'Persisting calculation result: {entry_name}')
                save_obj(entry_name, res)
            return res

    @cachetools.cached(cache=cache)
    def __call__(self, *args):
        entry_name = self._filename_gen(self._func.__name__, *args)
        filepath = get_path(entry_name, self._item_type)
        if self._read_only or (filepath.exists() and not self._recalculate):
            logger.debug('Reading from disk cache: %s', entry_name)
            res = read_obj(entry_name, self._item_type)
        else:
            res = self.recalculate(*args, persist=self._persist)

        return res

    def __getitem__(self, i: int):
        return self.__call__(i)

    def __setitem__(self, key, val):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        return self._batch_size

    def __iter__(self):
        for i in range(self._batch_size):
            yield self.__call__(i)


def cached[T](
        *,
        item_type: Optional[type] = None,
        recalculate: bool = False,
        persist: bool = True,
        read_only: bool = False,
        filename_gen: Optional[Callable[[str, int], str]] = None,
        batch: bool = False,
        batch_size: Optional[int] = None,
) -> Callable[[Callable[[], Iterable[T]]], Callable[[int], T]]:
    """Decorator adding caching on disk to functions that batch calculation for multiple items

    Parameters
    ----------
    item_type: type
        type of items in the batch
    recalculate: bool
        if True, perform the calculation from scratch ignoring any cached results. Default: False
    persist: bool
        if True, persist recalculated result on disk (only has effect if `recalculate` is `True`). Default: True
    read_only: bool
        if True, read the cached batch from disk and never attempt to recalculate. Default: False
    filename_gen: Optional[Callable[[str, int], str]]
        function returning file name to use when persisting results on disk
    batch: bool
        if True, the underlying calculation returns a sequence of items as opposed to invidivual items. Default: False
    batch_size: Optional[int]
        size of the batch to use in range checks (only has effect if `batch` is `True`)

    Returns
    -------
    Callable[[Callable[[], Iterable[T]]], Callable[[int], T]]
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
        cache_func = CacheFunc(
            func=func,
            item_type=item_type,
            recalculate=recalculate,
            persist=persist,
            read_only=read_only,
            batch=batch,
            batch_size=batch_size,
            filename_gen=filename_gen,
        )
        return wraps(func)(cache_func)
    return decorator
