from typing import Callable

import numpy as np
import scipy.stats as stats


def sample_chain(
    theta_sampler: Callable[[np.ndarray], np.ndarray],
    theta_init: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Sample a single chain of given length using the starting the values provided

    Parameters
    ----------
    theta_sampler: Callable[[np.ndarray], np.ndarray]
        transition kernel: function returning the next point given the current point
    theta_init: np.ndarray
        starting point
    n_samples: int
        required chain length

    Returns
    -------
    np.ndarray
        chain of samples: rows are samples, columns are variables
    """
    # set the starting values
    theta = np.array(theta_init, copy=True)

    # create an array for the trace
    trace = np.empty((n_samples + 1, theta_init.shape[0]))

    # store the initial values
    trace[0, :] = theta

    # sample variables
    for i in range(n_samples):
        # sample new theta
        theta = theta_sampler(theta)

        # record the value in the trace
        trace[i + 1, :] = theta

    return trace


def metropolis_random_walk_step(
    log_target_density: Callable[[np.ndarray], float],
    proposal_sampler: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build Metropolis-Hastings transition kernel

    Parameters
    ----------
    log_target_density: Callable[[np.ndarray], float]
        log density of the target distribution
    proposal_sampler: Callable[[np.ndarray], np.ndarray]
        function returning the proposed next point given the current position
    rng: np.random.Generator
        random number generator to use

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        transition kernel that can be used for sampling
    """
    def sampler(theta: np.ndarray) -> np.ndarray:
        # propose a new value
        theta_proposed = proposal_sampler(theta)

        # decide whether to accept the new value
        log_acceptance_probability = np.minimum(
            0, 
            log_target_density(theta_proposed) - log_target_density(theta)
        )
        u = rng.random()
        if u == 0 or np.log(u) < log_acceptance_probability:
            return theta_proposed
        else:
            return theta

    return sampler


def rw_proposal_sampler(
    step_size: float,
    rng: np.random.Generator,
    n_dim: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a Gaussian random walk proposal sampler

    Parameters
    ----------
    step_size: float
        scale of the step distribution
    rng: np.random.Generator
        random number generator to use
    n_dim: int
        dimension of the sample space

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        proposal sampler that draws a points from a multivariate
        Gaussian distribution with a diagonal matrix scaled by
        the provided value and centered at the point
    """
    G = step_size * np.identity(n_dim)
    def sampler(theta: np.ndarray) -> np.ndarray:
        xi = stats.norm.rvs(size=n_dim, random_state=rng)
        return theta + G @ xi
    return sampler
