# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import scipy.stats as stats

import arviz as az

import stan

import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jacobian

import dask.array as da
from dask.distributed import Client, progress

from stein_thinning.thinning import thin

# %%
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
# We create a Dask client in order to parallelise calculations where possible:

# %%
client = Client(processes=True, threads_per_worker=4, n_workers=4, memory_limit='2GB')
client


# %% [markdown]
# A helper function to parallelise computation over elements of an iterable:

# %%
def map_parallel(func, iterable, client):
    """Apply the function to each element of the iterable in parallel"""
    results = [client.submit(func, item) for item in iterable]
    return client.gather(results)


# %% [markdown]
# A helper function to process chunks or rows or columns of an array in parallel:

# %%
def apply_along_axis_parallel(func1d, axis, arr, chunk_size, client):
    """Apply function along the given axis and parallelise in chunks"""
    assert axis in (0, 1)

    # define a generator to obtain chunks of the matrix
    n_chunks = (arr.shape[1 - axis] - 1) // chunk_size + 1
    def chunker():
        for i in range(n_chunks):
            index_slice = slice(i * chunk_size, (i + 1) * chunk_size)
            if axis == 1:
                chunk = arr[index_slice, :]
            else:
                chunk = arr[:, index_slice]
            yield chunk

    # define the function to apply to chunks
    def func(chunk):
        return np.apply_along_axis(func1d, axis, chunk)
    
    # apply function to chunks in parallel
    results = map_parallel(func, chunker(), client)
    
    # reshape the results
    for i, result in enumerate(results):
        if len(result.shape) == 1:
            if axis == 1:
                results[i] = result.reshape(-1, 1)
            else:
                results[i] = result.reshape(1, -1)
    
    return np.concatenate(results, axis=1 - axis)


# %% [markdown]
# # Generate synthetic data

# %% [markdown]
# The Lotka-Volterra model is given by the equations:
# $$\begin{split}
# \frac{du_1}{dt} &= \theta_1 u_1 - \theta_2 u_1 u_2, \\
# \frac{du_2}{dt} &= \theta_4 u_1 u_2 - \theta_3 u_2,
# \end{split}$$
# where $u_1$ and $u_2$ are populations of prey and preditor, respectively, and $\theta_1, \theta_2, \theta_3, \theta_4$ are model parameters. All the quantities are positive.

# %% [markdown]
# Define the Lotka-Volterra model:

# %%
def lotka_volterra(t, u, theta):
    theta1, theta2, theta3, theta4 = theta
    u1, u2 = u
    return [
        theta1 * u1 - theta2 * u1 * u2,
        theta4 * u1 * u2 - theta3 * u2,
    ]


# %% [markdown]
# Solve the coupled ODEs:

# %%
t_n = 2500  # number of time data points
t_span = [0, 25]  # the time span over which to integrate the system
theta = [0.67, 1.33, 1., 1.]  # parameters of the model
q = 2  # number of state variables
d = len(theta)  # dimension of the parameter space
u_init = [1., 1.]  # initial values


# %%
def solve_lotka_volterra(theta):
    sol = solve_ivp(lotka_volterra, t_span, u_init, args=(theta,), dense_output=True)
    t = np.linspace(t_span[0], t_span[1], t_n)
    return t, sol.sol(t).T


# %%
t, u = solve_lotka_volterra(theta)

# %% [markdown]
# Add Gaussian noise:

# %%
rng = np.random.default_rng(12345)

# %%
means = [0, 0]
C = np.diag([0.2 ** 2, 0.2 ** 2])

# %%
eps = stats.multivariate_normal.rvs(mean=means, cov=C, size=len(u), random_state=rng)
y = u + eps

# %% [markdown]
# Plot the resulting values:

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for i in range(2):
    axs[i].plot(t, y[:, i], color='lightgray');
    axs[i].plot(t, u[:, i], color='black');
    axs[i].set_xlabel('t');
    axs[i].set_ylabel(f'$u_{i + 1}(t)$');

# %%
save_data = False
if save_data:
    filepath = Path('../data') / 'generated' / 'lotka_volterra_gaussian_noise.csv'
    df = pd.DataFrame({'u1': y[:, 0], 'u2': y[:, 1]}, index=pd.Index(t, name='t'))
    df.to_csv(filepath)


# %% [markdown]
# # Sample using a handwritten random-walk Metropolis-Hastings algorithm

# %% [markdown]
# Implement random-walk Metropolis-Hastings algorithm by hand:

# %%
def sample_chain(theta_sampler, theta_init, n_samples):
    """Sample a single chain of given length using the starting the values provided"""
    # set the starting values
    theta = np.array(theta_init, copy=True)

    # create an array for the trace
    trace = np.empty((n_samples + 1, d))

    # store the initial values
    trace[0, :] = theta

    # sample variables
    for i in range(n_samples):
        # sample new theta
        theta = theta_sampler(theta)

        # record the value in the trace
        trace[i + 1, :] = theta

    return trace


# %%
def to_arviz(chains, var_names):
    """Convert output to arviz format"""
    assert len(chains) > 0
    return az.from_dict({
        var_name: np.stack([chain[:, i] for chain in chains]) for i, var_name in enumerate(var_names)
    })


# %%
def metropolis_random_walk_step(log_target_density, proposal_sampler, rng):
    """Perform a Metropolis-Hastings random walk step"""
    def sampler(theta):
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


# %%
def log_target_density(theta):
    _, u = solve_lotka_volterra(np.exp(theta))
    log_likelihood = np.sum(stats.multivariate_normal.logpdf(y - u, mean=means, cov=C))
    log_prior = np.sum(stats.norm.logpdf(theta))
    return log_likelihood + log_prior


# %%
def rw_proposal_sampler(step_size, rng):
    G = step_size * np.identity(d)
    def sampler(theta):
        xi = stats.norm.rvs(size=d, random_state=rng)
        return theta + G @ xi
    return sampler


# %%
n_samples_rw = 500_000

# %%
# TODO consider selecting step size automatically following Gelman, Roberts, Gilks (1996) Efficient Metropolis Jumping Rules.
step_size = 0.0025

# %% [markdown]
# We use the initial values from Table S3 in Supplementary Material:

# %%
theta_inits = [
    np.array([0.55, 1, 0.8, 0.8]),
    np.array([1.5, 1., 0.8, 0.8]),
    np.array([1.3, 1.33, 0.5, 0.8]),
    np.array([0.55, 3., 3., 0.8]),
    np.array([0.55, 1., 1.5, 1.5]),
]

# %%
rw_seed = 12345
def run_rw_sampler(theta_init):
    rng = np.random.default_rng(rw_seed)
    theta_sampler = metropolis_random_walk_step(log_target_density, rw_proposal_sampler(step_size, rng), rng)
    return sample_chain(theta_sampler, np.log(theta_init), n_samples_rw)


# %%
# %%time
rw_samples = map_parallel(run_rw_sampler, theta_inits)


# %% [markdown]
# Reproduce the first column in Figure S17 from the Supplementary Material:

# %%
def plot_trace(samples, axs=None):
    if axs is None:
        n_cols = samples.shape[1]
        fig, axs = plt.subplots(1, samples.shape[1], figsize=(3 * n_cols, 3), constrained_layout=True)
    for i in range(d):
        axs[i].plot(samples[:, i]);
        axs[i].set_xscale('log');
        axs[i].set_xlabel('n');
        axs[i].set_ylabel(f'Parameter {i + 1}');


# %%
def plot_traces(traces):
    assert len(traces) > 0
    n_rows = len(traces)
    n_cols = traces[0].shape[1]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), constrained_layout=True, sharex=True)
    for i, trace in enumerate(traces):
        plot_trace(trace, axs[i] if n_rows > 1 else axs)


# %%
plot_traces(rw_samples)


# %%
def plot_paths(traces, theta_inits, idx1=0, idx2=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots();
    for i, trace in enumerate(traces):
        p = ax.plot(trace[:, idx1], trace[:, idx2], label=f'Chain {i + 1}');
        ax.scatter(theta_inits[i][idx1], theta_inits[i][idx2], color=p[0].get_color(), marker='x');
        ax.set_xlabel(f'$x_{idx1 + 1}$');
        ax.set_ylabel(f'$x_{idx2 + 1}$');
    ax.legend();


# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plot_paths(rw_samples, np.log(theta_inits), idx1=0, idx2=1, ax=axs[0])
plot_paths(rw_samples, np.log(theta_inits), idx1=2, idx2=3, ax=axs[1])


# %%
def acceptance_rate(sample):
    """Fraction of accepted samples"""
    return (np.sum(np.any(sample[1:] != sample[:-1], axis=1)) + 1) / sample.shape[0]


# %%
[acceptance_rate(sample) for sample in rw_samples]

# %% [markdown]
# Save traces for future analysis:

# %%
save_rw_results = False
if save_rw_results:
    for i, sample in enumerate(rw_samples):
        filepath = Path('../data') / 'generated' / f'rw_chain_{i}_seed_{rw_seed}.csv'
        np.savetxt(filepath, sample, delimiter=',')

# %% [markdown]
# ## Convergence diagnostics

# %%
az.summary(to_arviz(rw_samples, var_names=[f'theta{i + 1}' for i in range(d)]))

# %% [markdown]
# # Sample using Stan

# %% [markdown]
# The implementation follows https://mc-stan.org/docs/stan-users-guide/odes.html#estimating-system-parameters-and-initial-state.

# %%
stan_model_spec = """
functions {
  vector lotka_volterra(real t, vector u, vector theta) {
    vector[2] dudt;
    dudt[1] = exp(theta[1]) * u[1] - exp(theta[2]) * u[1] * u[2];
    dudt[2] = exp(theta[4]) * u[1] * u[2] - exp(theta[3]) * u[2];
    return dudt;
  }
}
data {
  int<lower=1> T;
  array[T] vector[2] y;
  real t0;
  array[T] real ts;
  vector[2] u0;
  vector<lower=0>[2] sigma;
}
parameters {
  vector[4] theta;
}
model {
  array[T] vector[2] u = ode_rk45(lotka_volterra, u0, t0, ts, theta);
  theta ~ std_normal();
  for (t in 1:T) {
    y[t] ~ normal(u[t], sigma);
  }
}
"""

# %%
data = {
    'T': t_n - 1,  # the first time is 0, for which the initial values are fixed
    'y': y[1:, :],
    't0': t_span[0],
    'ts': t[1:],
    'u0': u_init,
    'sigma': np.diag(C),  # TODO pass a matrix and use a multivariate normal in the Stan model
}

# %%
n_samples_hmc = 10_000


# %%
def extract_chains(stan_sample, param):
    """Extract chains from PyStan fit"""
    param_indices = stan_sample._parameter_indexes(param)
    return [stan_sample._draws[param_indices, :, i_chain].T for i_chain in range(stan_sample.num_chains)]


# %%
# %%time
inference_model = stan.build(stan_model_spec, data=data, random_seed=12345)
stan_sample = inference_model.sample(
    num_chains=len(theta_inits),
    num_samples=n_samples_hmc,
    save_warmup=True,
    init=[{'theta': np.log(theta_init)} for theta_init in theta_inits]
)

# %%
hmc_samples = extract_chains(stan_sample, 'theta')

# %%
plot_traces(hmc_samples)

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plot_paths(hmc_samples, np.log(theta_inits), idx1=0, idx2=1, ax=axs[0])
plot_paths(hmc_samples, np.log(theta_inits), idx1=2, idx2=3, ax=axs[1])

# %%
[acceptance_rate(sample) for sample in hmc_samples]

# %% [markdown]
# Save traces for further analysis:

# %%
save_hmc_results = False
if save_hmc_results:
    for i, sample in enumerate(hmc_samples):
        filepath = Path('../data') / 'generated' / f'hmc_chain_{i}_seed_{rw_seed}.csv'
        np.savetxt(filepath, sample, delimiter=',')

# %%
az.summary(az.from_pystan(stan_sample))


# %% [markdown]
# # Sensitivity analysis

# %% [markdown]
# ## Forward sensitivity equations

# %% [markdown]
# Given a system of ODEs of the form:
# $$\frac{du_r}{dt} = F_q(t, u_1, \dots, u_q; x),\qquad r=1,\dots,q,$$
# the sensitivities can be found by solving forward sensitivity equations (this is equation (35) in the Supplementary Material):
# $$\frac{d}{dt}\left(\frac{\partial u_r}{\partial x_s}\right) = \frac{dF_r}{dx_s} + \sum_{l=1}^q \frac{\partial F_r}{\partial u_l} \frac{\partial u_l}{\partial x_s}$$
# with initial conditions
# $$\frac{\partial u_r}{\partial x_s}(0) = 0.$$
#
# For the Lotka-Volterra model, the forward sensitivity equations are:
# $$\begin{split}
# \frac{d}{dt}\left(\frac{\partial u_1}{\partial \theta_1}\right) &= u_1 + (\theta_1 - \theta_2 u_2) \frac{\partial u_1}{\partial \theta_1} - \theta_2 u_1 \frac{\partial u_2}{\partial \theta_1}, \\
# \frac{d}{dt}\left(\frac{\partial u_1}{\partial \theta_2}\right) &= - u_1 u_2 + (\theta_1 - \theta_2 u_2) \frac{\partial u_1}{\partial \theta_2} - \theta_2 u_1 \frac{\partial u_2}{\partial \theta_2}, \\
# \frac{d}{dt}\left(\frac{\partial u_1}{\partial \theta_3}\right) &= (\theta_1 - \theta_2 u_2) \frac{\partial u_1}{\partial \theta_3} - \theta_2 u_1 \frac{\partial u_2}{\partial \theta_3}, \\
# \frac{d}{dt}\left(\frac{\partial u_1}{\partial \theta_4}\right) &= (\theta_1 - \theta_2 u_2) \frac{\partial u_1}{\partial \theta_4} - \theta_2 u_1 \frac{\partial u_2}{\partial \theta_4}, \\
# \frac{d}{dt}\left(\frac{\partial u_2}{\partial \theta_1}\right) &= \theta_4 u_2 \frac{\partial u_1}{\partial \theta_1} + (\theta_4 u_1 - \theta_3) \frac{\partial u_2}{\partial \theta_1}, \\
# \frac{d}{dt}\left(\frac{\partial u_2}{\partial \theta_2}\right) &= \theta_4 u_2 \frac{\partial u_1}{\partial \theta_2} + (\theta_4 u_1 - \theta_3) \frac{\partial u_2}{\partial \theta_2}, \\
# \frac{d}{dt}\left(\frac{\partial u_2}{\partial \theta_3}\right) &= -u_2 + \theta_4 u_2 \frac{\partial u_1}{\partial \theta_3} + (\theta_4 u_1 - \theta_3) \frac{\partial u_2}{\partial \theta_3}, \\
# \frac{d}{dt}\left(\frac{\partial u_2}{\partial \theta_4}\right) &= u_1 u_2 + \theta_4 u_2 \frac{\partial u_1}{\partial \theta_4} + (\theta_4 u_1 - \theta_3) \frac{\partial u_2}{\partial \theta_4}. \\
# \end{split}$$
#
# Denoting $w_1 = \frac{\partial u_1}{\partial \theta_1}$, $w_2 = \frac{\partial u_1}{\partial \theta_2}$, $w_3 = \frac{\partial u_1}{\partial \theta_3}$, $w_4 = \frac{\partial u_1}{\partial \theta_4}$ and $w_5 = \frac{\partial u_2}{\partial \theta_1}$, $w_6 = \frac{\partial u_2}{\partial \theta_2}$, $w_7 = \frac{\partial u_2}{\partial \theta_3}$, $w_8 = \frac{\partial u_2}{\partial \theta_4}$, we have the following additional equations:
# $$\begin{split}
# \frac{dw_1}{dt} &= u_1 + (\theta_1 - \theta_2 u_2) w_1 - \theta_2 u_1 w_5, \\
# \frac{dw_2}{dt} &= - u_1 u_2 + (\theta_1 - \theta_2 u_2) w_2 - \theta_2 u_1 w_6, \\
# \frac{dw_3}{dt} &= (\theta_1 - \theta_2 u_2) w_3 - \theta_2 u_1 w_7, \\
# \frac{dw_4}{dt} &= (\theta_1 - \theta_2 u_2) w_4 - \theta_2 u_1 w_8, \\
# \frac{dw_5}{dt} &= \theta_4 u_2 w_1 + (\theta_4 u_1 - \theta_3) w_5, \\
# \frac{dw_6}{dt} &= \theta_4 u_2 w_2 + (\theta_4 u_1 - \theta_3) w_6, \\
# \frac{dw_7}{dt} &= -u_2 + \theta_4 u_2 w_3 + (\theta_4 u_1 - \theta_3) w_7, \\
# \frac{dw_8}{dt} &= u_1 u_2 + \theta_4 u_2 w_4 + (\theta_4 u_1 - \theta_3) w_8. \\
# \end{split}$$

# %%
def lotka_volterra_sensitivity(t, uw, theta):
    theta1, theta2, theta3, theta4 = theta
    u1, u2, w1, w2, w3, w4, w5, w6, w7, w8 = uw
    return [
        # model equations
        theta1 * u1 - theta2 * u1 * u2,
        theta4 * u1 * u2 - theta3 * u2,
        # sensitivities
        u1 + (theta1 - theta2 * u2) * w1 - theta2 * u1 * w5,
        -u1 * u2 + (theta1 - theta2 * u2) * w2 - theta2 * u1 * w6,
        (theta1 - theta2 * u2) * w3 - theta2 * u1 * w7,
        (theta1 - theta2 * u2) * w4 - theta2 * u1 * w8,
        theta4 * u2 * w1 + (theta4 * u1 - theta3) * w5,
        theta4 * u2 * w2 + (theta4 * u1 - theta3) * w6,
        -u2 + theta4 * u2 * w3 + (theta4 * u1 - theta3) * w7,
        u1 * u2 + theta4 * u2 * w4 + (theta4 * u1 - theta3) * w8,
    ]


# %%
uw_init = np.concatenate([np.array(u_init), np.zeros(d * q)])

# %%
# %%time
sol = solve_ivp(lotka_volterra_sensitivity, t_span, uw_init, args=(theta,), dense_output=True)
sensitivity_forward = sol.sol(t).T

# %% [markdown]
# Plot the solution again:

# %%
fig, axs = plt.subplots(1, q, figsize=(10, 4), constrained_layout=True)
for i in range(q):
    axs[i].plot(t, sensitivity_forward[:, i]);
    axs[i].set_xlabel('t');
    axs[i].set_ylabel(f'$u_{i + 1}(t)$');

# %% [markdown]
# Plot the sensitivities:

# %%
fig, axs = plt.subplots(q, d, figsize=(14, 6), constrained_layout=True)
for i in range(q):
    for j in range(d):
        axs[i][j].plot(t, sensitivity_forward[:, q + i * d + j]);
        axs[i][j].set_xlabel('t');
        axs[i][j].set_ylabel(f'$\\partial u_{{{i + 1}}} / \\partial \\theta_{{{j + 1}}}$');


# %% [markdown]
# ## Numerical Jacobian calculation

# %% [markdown]
# Redefine the function since ``jax.experimental.ode.odeint`` passes the state variable in the first argument and time in the second argument:

# %%
def lotka_volterra2(u, t, theta):
    return [
        theta[0] * u[0] - theta[1] * u[0] * u[1],
        theta[3] * u[0] * u[1] - theta[2] * u[1],
    ]


# %%
def solve_lotka_volterra2(theta):
    return odeint(lotka_volterra2, jnp.array(u_init), jnp.array(t), jnp.array(theta))


# %%
sol2 = solve_lotka_volterra2(theta)

# %% [markdown]
# Plot the solution first:

# %%
fig, axs = plt.subplots(1, q, figsize=(10, 4), constrained_layout=True)
for i in range(q):
    axs[i].plot(t, sol2[:, i]);
    axs[i].set_xlabel('t');
    axs[i].set_ylabel(f'$u_{i + 1}(t)$');

# %% [markdown]
# Calculate the sensitivities:

# %%
# %%time
sensitivity_jax = jacobian(solve_lotka_volterra2)(theta)

# %% [markdown]
# We confirm that the numerical method agrees with the results from forward sensitivity equations:

# %%
fig, axs = plt.subplots(q, d, figsize=(14, 6), constrained_layout=True)
for i in range(2):
    for j in range(d):
        axs[i][j].plot(t, sensitivity_forward[:, 2 + i * d + j], label='Forward equations');
        axs[i][j].plot(t, sensitivity_jax[j][:, i], label='JAX');
        axs[i][j].set_xlabel('t');
        axs[i][j].set_ylabel(f'$\\partial u_{{{i + 1}}} / \\partial \\theta_{{{j + 1}}}$');
        axs[i][j].legend();

# %% [markdown]
# ## Calculating the gradient of the log-posterior

# %% [markdown]
# The Stein Thinning methog requires the gradient of the log-posterior $\nabla \log p$ as input. Below we follow section S3 of the Supplementary Material to derive it.
#
# Since $p(x) \propto \mathcal{L}(x) \pi(x)$, we have 
# $$\nabla \log p(x) = \nabla \log \mathcal{L}(x) + \nabla \log \pi(x).$$
# Assuming independent errors in observations yields
# $$\mathcal{L}(x) = \prod_{i=1}^N \phi_i(u(t_i)),$$
# and thus 
# $$
# \frac{\partial}{\partial x_s} \log \mathcal{L}(x) 
# = \sum_{i=1}^N \frac{\partial}{\partial x_s} \log \phi_i(u(t_i))
# = \sum_{i=1}^N \sum_{r=1}^q \frac{\partial}{\partial u_r} (\log \phi_i) \frac{\partial u_r}{\partial x_s},
# $$
# which can be written in matrix notation as
# $$(\nabla \log \mathcal{L})(x) = \sum_{i=1}^N \left(\frac{\partial u}{\partial x}\right)^T (t_i) (\nabla \log \phi_i)(u(t_i)),$$
# where
# $$\left(\frac{\partial u}{\partial x}\right)_{r,s} = \frac{\partial u_r}{\partial x_s}$$
# is the Jacobian matrix of sensitivities, as obtained earlier.
#
# Note that this does not match the expression provided on page 16 of the Supplementary Material:
# $$(\nabla \log \mathcal{L})(x) = -\sum_{i=1}^N \frac{\partial u}{\partial x}(t_i) (\nabla \log \phi_i)(u(t_i)),$$
# where the Jacobian is not transposed and there is a minus sign in front of the expression.
#
# For a multivariate normal distribution of the errors:
# $$\phi_i(u(t_i)) \propto \exp\left( -\frac{1}{2} (y_i - u(t_i))^T C^{-1} (y_i - u(t_i)) \right)$$
# we obtain
# $$(\nabla \log \phi_i)(u(t_i)) = C^{-1}(y_i - u(t_i)).$$
#
# We assume independent standard normal priors for all parameters, therefore
# $$\pi(x) = \prod_{i=1}^d \pi_i(x_i) \propto \exp\left(-\frac{1}{2}\sum_{i=1}^d x_i^2\right)$$
# and
# $$\nabla \log \pi(x) = -x.$$

# %% [markdown]
# We calculate the gradient of the log-likelihood from the Jacobian obtained previosly:

# %%
# reshape the Jacobian so it can be multiplied by the gradient of log phi
J = sensitivity_forward[:, q:].reshape(len(t), -1, q, order='F')
J.shape

# %%
# calculate the gradient of log phi and reshape it
grad_log_phi = (inv(C) @ (y - sensitivity_forward[:, :q]).T).T[:, :, np.newaxis]
grad_log_phi.shape

# %%
grad_log_lik = np.sum(np.squeeze(J @ grad_log_phi), axis=0)
grad_log_lik


# %% [markdown]
# Now put the calculation into a function so we can use it later:

# %%
def grad_log_likelihood(theta):
    """Solve the system of ODEs and calculate the log-likelihood"""
    sol = solve_ivp(lotka_volterra_sensitivity, t_span, uw_init, args=(theta,), dense_output=True)
    sensitivity_forward = sol.sol(t).T
    J = sensitivity_forward[:, q:].reshape(len(t), -1, q, order='F')
    grad_log_phi = (inv(C) @ (y - sensitivity_forward[:, :q]).T).T[:, :, np.newaxis]
    return np.sum(np.squeeze(J @ grad_log_phi), axis=0)


# %%
# %%time
grad_log_likelihood(theta)


# %% [markdown]
# We check the numbers against the numerical gradient:

# %%
def grad_log_likelihood_jax(theta):
    import jax.scipy.stats as jstats
    def log_likelihood(theta):
        sol = odeint(lotka_volterra2, jnp.array(u_init), jnp.array(t), jnp.array(theta))
        return jnp.sum(jstats.multivariate_normal.logpdf(jnp.array(y) - sol, mean=jnp.array(means), cov=jnp.array(C)))
    return jacobian(log_likelihood)(theta)


# %%
# %%time
grad_log_likelihood_jax(theta)


# %% [markdown]
# We use the gradient calculation based on solving forward sensitivity equations in what follows.

# %%
def grad_log_posterior(theta):
    return grad_log_likelihood(theta) - theta


# %%
grad_log_posterior(theta)

# %% [markdown]
# ## Sequential calculation of gradients

# %% [markdown]
# We can save time by calculating the gradients for unique samples only:

# %%
unique_samples, inverse_index = np.unique(np.exp(rw_samples[0]), axis=0, return_inverse=True)
unique_samples.shape

# %%
# %%time
grad = np.apply_along_axis(grad_log_posterior, 1, unique_samples)

# %% [markdown]
# ## Parallel calculation of gradients

# %% [markdown]
# We note that calculating gradients after a MCMC run is what is called "embarrassingly parallelisable" and the time required for this step can be effectively eliminated given sufficient computational resources. This is in contrast to the MCMC run itself, which is inherently sequential.
#
# Here we demonstrate how the popular package ``Dask`` can be used to parallelise this computation across the cores of a CPU on a single machine.

# %%
# %%time
grad_parallel = apply_along_axis_parallel(grad_log_posterior, 1, unique_samples, 200, client)

# %% [markdown]
# Confirm the values are identical to the sequential run:

# %%
assert np.max(np.abs(grad_parallel - grad)) < 1e-9


# %% [markdown]
# A helper function to calculate gradients using unique values only:

# %%
def calculate_gradient(sample, chunk_size=200):
    """Calculate gradients for samples"""
    unique_samples, inverse_index = np.unique(sample, axis=0, return_inverse=True)
    grad = apply_along_axis_parallel(grad_log_posterior, 1, unique_samples, chunk_size, client)
    return grad[inverse_index]


# %%
# %%time
rw_grad = calculate_gradient(np.exp(rw_samples[0]))

# %%
assert np.all(rw_grad == grad_parallel[inverse_index])

# %% [markdown]
# # Apply Stein Thinning

# %% [markdown]
# ### Random-walk sample

# %%
n_points_thinned = 20

# %%
# %%time
rw_thinned_idx = thin(np.exp(rw_samples[0]), rw_grad, n_points_thinned, pre='med')


# %% [markdown]
# This reproduces the results shown in Figure S20 in the Supplementary Material:

# %%
def plot_sample_thinned(sample, thinned_idx):
    fig, ax = plt.subplots()
    ax.scatter(sample[:, 0], sample[:, 1], color='lightgray', s=1);
    ax.scatter(sample[thinned_idx, 0], sample[thinned_idx, 1], color='red', s=4);
    ax.set_xlabel('$x_1$');
    ax.set_ylabel('$x_2$');


# %%
plot_sample_thinned(rw_samples[0], rw_thinned_idx)

# %% [markdown]
# ### HMC sample

# %%
# %%time
hmc_grad = calculate_gradient(np.exp(hmc_samples[0]))
hmc_thinned_idx = thin(np.exp(hmc_samples[0]), hmc_grad, n_points_thinned, pre='med')

# %%
plot_sample_thinned(hmc_samples[0], hmc_thinned_idx)
