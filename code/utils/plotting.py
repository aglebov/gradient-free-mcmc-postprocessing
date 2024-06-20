from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def plot_trace(samples: np.ndarray, axs: Sequence[Axes] = None) -> Figure:
    """Plot trace of an MCMC chain
    
    Parameters
    ----------
    samples: np.ndarray
        array of samples (rows are observations, columns are variables)
    axs: Sequence[Axes]
        axes to plot on. If not provided, a new figure will be created
    
    Returns
    -------
    Figure
        figure that was used for plotting
    """
    n_cols = samples.shape[1]
    if axs is None:
        _, axs = plt.subplots(1, samples.shape[1], figsize=(3 * n_cols, 3), constrained_layout=True)
    for i in range(n_cols):
        axs[i].plot(samples[:, i])
        axs[i].set_xscale('log')
        axs[i].set_xlabel('n')
        axs[i].set_ylabel(f'Parameter {i + 1}')
    
    return axs[0].figure


def plot_traces(traces: Sequence[np.ndarray]) -> Figure:
    """Plot traces of multiple MCMC chains
    
    Parameters
    ----------
    traces: Sequence[np.ndarray]
        samples for each chain

    Returns
    -------
    Figure
        figure that was used for plotting
    """
    assert len(traces) > 0
    n_rows = len(traces)
    n_cols = traces[0].shape[1]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), constrained_layout=True, sharex=True)
    for i, trace in enumerate(traces):
        plot_trace(trace, axs[i] if n_rows > 1 else axs)

    return fig


def plot_paths(
        traces: Sequence[np.ndarray],
        theta_inits: np.ndarray,
        idx1: int = 0,
        idx2: int = 1,
        ax: Axes = None
) -> Figure:
    """Plot paths of MCMC chains

    Parameters
    ----------
    traces: Sequence[np.ndarray]
        samples for each chain
    theta_inits: np.ndarray
        starting values for each chain
    idx1: int
        index of variable to plot on the x axis
    idx2: int
        index of variable to plot on the y axis
    ax: Axes
        axes to plot on

    Returns
    -------
    Figure
        figure used for plotting        
    """
    if ax is None:
        _, ax = plt.subplots()
    for i, trace in enumerate(traces):
        p = ax.plot(trace[:, idx1], trace[:, idx2], label=f'Chain {i + 1}')
        ax.scatter(theta_inits[i][idx1], theta_inits[i][idx2], color=p[0].get_color(), marker='x')
        ax.set_xlabel(f'$x_{idx1 + 1}$')
        ax.set_ylabel(f'$x_{idx2 + 1}$')
    ax.legend()

    return ax.figure
