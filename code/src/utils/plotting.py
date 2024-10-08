from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib
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
        add_legend: bool = True,
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
    add_legend: bool
        if True, add legend onto each subplot. Default: True

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

    if add_legend:
        ax.legend()

    return ax.figure


def highlight_points(
    sample: np.ndarray,
    indices: Iterable[int],
    coord_spec: Sequence[Sequence[int]],
    axs: Sequence[Axes] = None,
    var_labels: Sequence[str] = None,
    show_labels: bool = False,
    sample_point_size: float = 1,
    highlighted_point_size: Optional[float] = None,
    sample_point_color: str = 'lightgray',
    highlighted_point_color: str = 'red',
    sample_point_alpha: float = 0.3,
) -> Figure:
    """Highlight points on a scatter plot of a sample

    Parameters
    ----------
    sample: np.ndarray
        points to plot
    indices: Iterable[int]
        indices of points to highlight
    coord_spec: Sequence[Sequence[int]]
        coordinates to display on plots. The number of elements will correspond to the number of plots.
        Each element is a two-element sequence specifying the coordinates to plot on the x and y axes.
    ax: Axes
        axes to plot on. If not provided, a new figure will be created with the number of subplots matching
        `coord_spec`
    var_labels: Sequence[str]
        labels to use for the coordinates
    show_labels: bool
        if True, add index labels on the plot. Default: False
    sample_point_size: float
        size to use for plotting sample points. Default: 1
    highlighted_point_size: float
        size to use for plotting the highlighted points. Default: the matplotlib default
    sample_point_color: str
        color to use for plotting sample points. Default: `'lightgray'`
    highlighted_point_color: str
        color to use for plotting the highlighted points. Default: `'red'`
    sample_point_alpha: float
        transparency to use for plotting sample points. Default: 0.3

    Returns
    -------
    Figure
        figure used for plotting
    """
    if axs is None:
        fig, axs = plt.subplots(1, len(coord_spec))
        if len(coord_spec) == 1:
            axs = [axs]
    else:
        fig = axs[0].figure

    for i, coords in enumerate(coord_spec):
        ax = axs[i]
        ax.scatter(
            sample[:, coords[0]],
            sample[:, coords[1]],
            alpha=sample_point_alpha,
            s=sample_point_size,
            color=sample_point_color,
        )
        ax.scatter(
            sample[indices, coords[0]],
            sample[indices, coords[1]],
            s=highlighted_point_size,
            color=highlighted_point_color,
        )

        if show_labels:
            for i, ind in enumerate(indices):
                ax.text(sample[ind, 0], sample[ind, 1], str(ind))

        if var_labels is not None:
            ax.set_xlabel(var_labels[coords[0]])
            ax.set_ylabel(var_labels[coords[1]])

    return fig


def plot_sample_thinned(
    traces: Sequence[np.ndarray],
    thinned_idx: Sequence[Sequence[int]],
    titles: Optional[Sequence[str]],
    labels: Optional[Sequence[str]] = None,
    n_points: Optional[int] = None,
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
    n_points: int
        number of points to display. Default: display all points

    Returns
    -------
    Figure
        figure used for plotting
    """
    fig = plt.figure(constrained_layout=True, figsize=(12, 5 * len(traces)))
    subfigs = fig.subfigures(nrows=len(traces), ncols=1)
    for i, trace in enumerate(traces):
        if len(traces) > 1:
            subfig = subfigs[i]
        else:
            subfig = subfigs
        if titles is not None:
            subfig.suptitle(titles[i])
        axs = subfig.subplots(nrows=1, ncols=2)
        idx = thinned_idx[i][:n_points]
        highlight_points(trace, idx, [(0, 1), (2, 3)], axs, labels, highlighted_point_size=4)

    return fig


def plot_density(
    f: Callable[[np.ndarray], np.ndarray],
    ax: Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    mesh_size: int = 200,
    levels: int = 200,
    fill: bool = True,
    level_labels: bool = False,
    label_format: str = '%2.1f',
    label_font_size: int = 6,
    colorbar: bool = False,
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
    fill: bool
        if True, use the filled contour plot, otherwise use the outline
    level_labels: bool
        if True, show labels for levels on the plot
    label_format: str
        the format to use for level labels. Default: %2.1f
    label_font_size: int
        the font size to use for level labels. Default: 6
    colorbar: bool
        if True, display the colorbar
    """
    x = np.linspace(*xlim, mesh_size)
    y = np.linspace(*ylim, mesh_size)
    xy = np.moveaxis(np.stack(np.meshgrid(x, y)), 0, 2).reshape(mesh_size * mesh_size, 2)
    z = f(xy).reshape(mesh_size, mesh_size)

    if fill:
        cs = ax.contourf(x, y, z, levels=levels)
    else:
        cs = ax.contour(x, y, z, levels=levels)
    
    if level_labels:
        ax.clabel(cs, fmt=label_format, fontsize=label_font_size)
    
    if colorbar:
        norm = matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, ticks=cs.levels)

    ax.set_title(title)


def centered_subplots(fig: Figure, row_specs: Sequence[int]) -> Sequence[Axes]:
    """Create a grid of axes following the provided specification

    The code was adapted from the answer https://stackoverflow.com/questions/53361373/center-the-third-subplot-in-the-middle-of-second-row-python

    Parameters
    ----------
    fig: Figure
        figure to create plots in
    row_specs: Sequence[int]
        number of plots to create in each row

    Returns
    -------
    Sequence[Axes]
        flat sequence of axes corresponding to plots in the grid
    """
    max_cols = max(row_specs)
    grid_shape = (len(row_specs), 2 * max_cols)

    axs = []

    for i_row, n_cols in enumerate(row_specs):
        offset = max_cols - n_cols if n_cols < max_cols else 0
        for i_col in range(n_cols):
            ax_position = (i_row, 2 * i_col + offset)
            ax = plt.subplot2grid(grid_shape, ax_position, fig=fig, colspan=2)
            axs.append(ax)

    return axs
