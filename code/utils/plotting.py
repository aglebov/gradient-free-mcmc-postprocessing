from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def plot_trace(
    samples: np.ndarray,
    axs: Optional[Sequence[Axes]] = None,
    var_labels: Sequence[str] = None,
) -> Figure:
    """Plot trace of an MCMC chain
    
    Parameters
    ----------
    samples: np.ndarray
        array of samples (rows are observations, columns are variables)
    axs: Optional[Sequence[Axes]]
        axes to plot on. If not provided, a new figure will be created
    var_labels: Optional[Sequence[str]]
        labels for variables to use on y axes
    
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
        if var_labels is not None:
            axs[i].set_ylabel(var_labels[i])
        else:
            axs[i].set_ylabel(f'Parameter {i + 1}')
    
    return axs[0].figure


def plot_traces(
    traces: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    var_labels: Optional[Sequence[str]] = None,
) -> Figure:
    """Plot traces of multiple MCMC chains
    
    Parameters
    ----------
    traces: Sequence[np.ndarray]
        samples for each chain
    titles: Optional[Sequence[str]]
        titles for rows of plots
    var_labels: Optional[Sequence[str]]
        labels for variables to use on y axes

    Returns
    -------
    Figure
        figure that was used for plotting
    """
    assert len(traces) > 0
    n_rows = len(traces)
    n_cols = traces[0].shape[1]
    fig = plt.figure(constrained_layout=True, figsize=(3 * n_cols, 3 * n_rows))
    subfigs = fig.subfigures(nrows=n_rows, ncols=1)
    for i, trace in enumerate(traces):
        if titles is not None:
            subfigs[i].suptitle(titles[i]);
        axs = subfigs[i].subplots(nrows=1, ncols=n_cols, sharex=True)
        plot_trace(trace, axs, var_labels)

    return fig


def plot_paths(
        traces: Sequence[np.ndarray],
        theta_inits: np.ndarray,
        idx1: int = 0,
        idx2: int = 1,
        ax: Optional[Axes] = None,
        label1: Optional[str] = None,
        label2: Optional[str] = None,
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
    ax: Optional[Axes]
        axes to plot on
    label1: Optional[str]
        variable label corresponding to `ind1`
    label2: Optional[str]
        variable label corresponding to `ind2`

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
        if label1 is not None:
            ax.set_xlabel(label1)
        else:
            ax.set_xlabel(f'$x_{idx1 + 1}$')
        if label2 is not None:
            ax.set_ylabel(label2)
        else:
            ax.set_ylabel(f'$x_{idx2 + 1}$')
    ax.legend()

    return ax.figure


def plot_thinned_coords(
    samples: np.ndarray,
    thinned_idx: Sequence[int],
    coord1: int,
    coord2: int,
    ax: Axes,
    labels: Optional[Sequence[str]] = None,
):
    """Highlight selected points on sample scatter plot
    
    Parameters
    ----------
    samples: np.ndarray
        array of samples (rows are observations, columns are variables)
    thinned_idx: Sequence[int]
        indices of points in `samples` to highlight
    coord1: int
        index of variable to plot on the x axis
    coord2: int
        index of variable to plot on the y axis
    ax: Axes
        axes to plot on
    labels: Optional[Sequence[str]]
        variable names to use as axes labels

    Returns
    -------
    None
    """
    ax.scatter(samples[:, coord1], samples[:, coord2], color='lightgray', s=1);
    ax.scatter(samples[thinned_idx, coord1], samples[thinned_idx, coord2], color='red', s=4);
    if labels is not None:
        ax.set_xlabel(labels[coord1]);
        ax.set_ylabel(labels[coord2]);
    else:
        ax.set_xlabel(f'$x_{coord1 + 1}$');
        ax.set_ylabel(f'$x_{coord2 + 1}$');


def plot_sample_thinned(
    traces: Sequence[np.ndarray],
    thinned_idx: Sequence[Sequence[int]],
    titles: Optional[Sequence[str]],
    labels: Optional[Sequence[str]] = None,
) -> Figure:
    """Highlight selected points on sample scatter plot for multiple traces
    
    Parameters
    ----------
    traces: Sequence[np.ndarray]
        samples for each chain
    thinned_idx: Sequence[Sequence[int]]
        for each chain, indices of points in `samples` to highlight
    titles: Optional[Sequence[str]]
        titles for rows of plots
    labels: Optional[Sequence[str]]
        variable names to use as axes labels

    Returns
    -------
    Figure
        figure used for plotting
    """
    fig = plt.figure(constrained_layout=True, figsize=(12, 5 * len(traces)))
    subfigs = fig.subfigures(nrows=len(traces), ncols=1)
    for i in range(len(traces)):
        if len(traces) > 1:
            subfig = subfigs[i]
        else:
            subfig = subfigs
        if titles is not None:
            subfig.suptitle(titles[i]);
        axs = subfig.subplots(nrows=1, ncols=2);
        plot_thinned_coords(traces[i], thinned_idx[i], 0, 1, axs[0], labels=labels)
        plot_thinned_coords(traces[i], thinned_idx[i], 2, 3, axs[1], labels=labels)

    return fig


def plot_density(
    f: Callable[[np.ndarray], np.ndarray],
    ax: Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    mesh_size: int = 200,
    levels: int = 200,
):
    """Contour plot of probability density

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        density function to plot
    ax: Axes
        axes to plot on
    xlim: Tuple[float, float]
        limits for the x axis
    ylim: Tuple[float, float]
        limits for the y axis
    title: str
        plot title
    mesh_size: int
        number of points along each dimension at which to evaluate `f`
    levels: int
        number of color levels to show on the plot
    """
    x = np.linspace(*xlim, mesh_size)
    y = np.linspace(*ylim, mesh_size)
    xy = np.moveaxis(np.stack(np.meshgrid(x, y)), 0, 2).reshape(mesh_size * mesh_size, 2)
    z = f(xy).reshape(mesh_size, mesh_size)

    ax.contourf(x, y, z, levels=levels);
    ax.set_title(title);


def highlight_points(
    sample: np.ndarray,
    indices: Iterable[int],
    show_labels: bool = False,
    ax: Axes = None
):
    """Highlight points on a scatter plot of a sample

    Parameters
    ----------
    sample: np.ndarray
        points to plot
    indices: Iterable[int]
        indices of points to highlight
    show_labels: bool
        if True, add index labels on the plot. Default: False
    ax: Axes
        axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, color='gray')
    ax.scatter(sample[indices, 0], sample[indices, 1], color='red')
    if show_labels:
        for i, ind in enumerate(indices):
            ax.text(sample[ind, 0], sample[ind, 1], str(ind))
