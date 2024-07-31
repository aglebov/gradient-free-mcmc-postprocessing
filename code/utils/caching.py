import numpy as np


def cached_calculate(items, func, filepath_gen, recalculate=False, save=False, mapper=None):
    """Perform expensive calculation for provided items or read stored results"""
    if recalculate:
        if mapper is None:
            results = map(func, items)
        else:
            results = mapper(func, items)
        results = list(results)
        if save:
            for i, result in enumerate(results):
                filepath = filepath_gen(i)
                np.savetxt(filepath, result, delimiter=',')
    else:
        results = []
        for i in range(len(items)):
            filepath = filepath_gen(i)
            results.append(np.genfromtxt(filepath, delimiter=','))
    return results
