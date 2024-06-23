from typing import Iterable

import arviz as az
import numpy as np


def to_arviz(chains: Iterable[np.ndarray], var_names: Iterable[str]) -> az.InferenceData:
    """Convert output to arviz format

    Parameters
    ----------
    chains: Iterable[np.ndarray]
        samples from chains of MCMC, each element is a 2D array with observation in rows
        and variables in columns
    var_names: Iterable[str]
        names of variables in the MCMC samples

    Returns
    -------
    az.InferenceData
        an ``InferenceData`` object suitable for further analysis with ``arviz``
    """
    assert len(chains) > 0
    return az.from_dict({
        var_name: np.stack([chain[:, i] for chain in chains]) for i, var_name in enumerate(var_names)
    })
