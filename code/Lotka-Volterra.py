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
import time
start_time = time.time()

# %%
import logging
from pathlib import Path

import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import scipy.stats as stats
from scipy.stats import multivariate_normal as mvn

import arviz as az

import stan

from jax import jacobian
from jax.experimental.ode import odeint
import jax.numpy as jnp

from dask.distributed import Client

import dcor

from stein_thinning.thinning import thin, thin_gf

from mcmc import sample_chain, metropolis_random_walk_step, rw_proposal_sampler
import utils.caching
from utils.caching import cached, cached_batch, subscriptable
from utils.parallel import apply_along_axis_parallel, get_map_parallel
from utils.plotting import highlight_points, plot_paths, plot_sample_thinned, plot_traces
from utils.sampling import to_arviz

# %%
logging.basicConfig()
logging.getLogger(utils.caching.__name__).setLevel(logging.DEBUG)

# %%
import nest_asyncio
nest_asyncio.apply()

# %%
figures_path = Path('../report') / 'figures'

# %% [markdown]
# Directory where results of expensive calculations will be stored:

# %%
generated_data_dir = Path('../data') / 'generated'
utils.caching.cache_dir = generated_data_dir

# %%
recalculate = False  # True => perform expensive calculations, False => use stored results
save_data = recalculate

# %% [markdown]
# We create a Dask client in order to parallelise calculations where possible:

# %%
client = Client(processes=True, threads_per_worker=4, n_workers=4, memory_limit='2GB')
client

# %%
map_parallel = get_map_parallel(client)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
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
t_n = 2400  # number of time data points
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
#fig.suptitle('Lotka-Volterra solution with added Gaussian noise');
for i in range(2):
    axs[i].plot(t, y[:, i], color='lightgray');
    axs[i].plot(t, u[:, i], color='black');
    axs[i].set_xlabel('t');
    axs[i].set_ylabel(f'$u_{i + 1}(t)$');

fig.savefig(figures_path / 'lotka-volterra.pdf');

# %%
if save_data:
    filepath = generated_data_dir / 'lotka_volterra_gaussian_noise.csv'
    df = pd.DataFrame({'u1': y[:, 0], 'u2': y[:, 1]}, index=pd.Index(t, name='t'))
    df.to_csv(filepath)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Sample using a handwritten random-walk Metropolis-Hastings algorithm

# %% [markdown]
# Implement random-walk Metropolis-Hastings algorithm by hand:

# %%
def log_target_density(log_theta):
    _, u = solve_lotka_volterra(np.exp(log_theta))
    log_likelihood = np.sum(stats.multivariate_normal.logpdf(y - u, mean=means, cov=C))
    log_prior = np.sum(stats.norm.logpdf(log_theta))
    return log_likelihood + log_prior


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
    theta_sampler = metropolis_random_walk_step(log_target_density, rw_proposal_sampler(step_size, rng, d), rng)
    return sample_chain(theta_sampler, np.log(theta_init), n_samples_rw)


# %%
@subscriptable(n=len(theta_inits))
@cached_batch(item_type=np.ndarray, recalculate=recalculate, persist=True)
def rw_samples() -> list[np.ndarray]:
    return map_parallel(run_rw_sampler, theta_inits)


# %%
# force calculation in parallel
rw_samples[0];

# %% [markdown]
# Reproduce the first column in Figure S17 from the Supplementary Material:

# %%
titles = [f'$\\theta^{{(0)}} = ({theta[0]}, {theta[1]}, {theta[2]}, {theta[3]})$' for theta in theta_inits]
var_labels = [f'$\\log \\theta_{i + 1}$' for i in range(d)]

# %%
fig = plot_traces(rw_samples, titles=titles, var_labels=var_labels);
fig.suptitle('Traces from the random-walk Metropolis-Hasting algorithm');

# %% [markdown]
# Produce a figure for the report:

# %%
fig = plot_traces(rw_samples, titles=[f'Chain {i + 1}' for i in range(len(rw_samples))], var_labels=var_labels);
fig.savefig(figures_path / 'lotka-volterra-trace-plots.pdf');

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
plot_paths(rw_samples, np.log(theta_inits), idx1=0, idx2=1, ax=axs[0], label1='$\\log \\theta_1$', label2='$\\log \\theta_2$');
plot_paths(rw_samples, np.log(theta_inits), idx1=2, idx2=3, ax=axs[1], label1='$\\log \\theta_3$', label2='$\\log \\theta_4$');
fig.savefig(figures_path / 'lotka-volterra-chain-paths.png', dpi=600);
fig.suptitle('Traversal paths from the random-walk Metropolis-Hastings algorithm');


# %%
def acceptance_rate(sample):
    """Fraction of accepted samples"""
    return (np.sum(np.any(sample[1:] != sample[:-1], axis=1)) + 1) / sample.shape[0]


# %%
[acceptance_rate(sample) for sample in rw_samples]


# %% [markdown]
# ## Convergence diagnostics

# %% [markdown]
# ``arviz`` implements $\hat{R}$ and the expected sample size as recommended in _Vehtari et al. (2021) Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC_. The paper suggests the minimum ESS of 50 for each chain and the threshold value of 1.01 for $\hat{R}$. Based on these thresholds, the chains would be deemed not to have converged:

# %%
@cached(recalculate=recalculate, persist=True)
def rw_az_summary() -> pd.DataFrame:
    return az.summary(to_arviz(rw_samples, var_names=[f'log_theta{i + 1}' for i in range(d)]))


# %%
rw_az_summary()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Sample using Stan

# %% [markdown]
# The implementation follows https://mc-stan.org/docs/stan-users-guide/odes.html#estimating-system-parameters-and-initial-state.

# %%
stan_model_spec = """
functions {
  vector lotka_volterra(real t, vector u, vector log_theta) {
    vector[2] dudt;
    dudt[1] = exp(log_theta[1]) * u[1] - exp(log_theta[2]) * u[1] * u[2];
    dudt[2] = exp(log_theta[4]) * u[1] * u[2] - exp(log_theta[3]) * u[2];
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
  vector[4] log_theta;
}
model {
  array[T] vector[2] u = ode_rk45(lotka_volterra, u0, t0, ts, log_theta);
  log_theta ~ std_normal();
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
hmc_seed = 12345


# %%
@subscriptable(n=len(theta_inits))
@cached_batch(item_type=np.ndarray, recalculate=recalculate, persist=True)
def hmc_samples() -> list[np.ndarray]:
    inference_model = stan.build(
        stan_model_spec,
        data=data,
        random_seed=hmc_seed,
    )
    stan_sample = inference_model.sample(
        num_chains=len(theta_inits),
        num_samples=n_samples_hmc,
        save_warmup=True,
        init=[{'log_theta': np.log(theta_init)} for theta_init in theta_inits]
    )
    return extract_chains(stan_sample, 'log_theta')


# %%
# force the calculation
hmc_samples[0];

# %%
fig = plot_traces(hmc_samples, titles=titles, var_labels=var_labels);
fig.suptitle('Traces from the HMC algorithm');

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Traversal paths from the HMC algorithm');
plot_paths(hmc_samples, np.log(theta_inits), idx1=0, idx2=1, ax=axs[0], label1='$\\log \\theta_1$', label2='$\\log \\theta_2$');
plot_paths(hmc_samples, np.log(theta_inits), idx1=2, idx2=3, ax=axs[1], label1='$\\log \\theta_3$', label2='$\\log \\theta_4$');

# %%
[acceptance_rate(sample) for sample in hmc_samples]


# %% [markdown]
# Based on the thresholds in _Vehtari et al. (2021) Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC_, the diagnostics do not suggest any convergence issues:

# %%
@cached(recalculate=recalculate, persist=True)
def hmc_az_summary() -> pd.DataFrame:
    return az.summary(to_arviz(hmc_samples, var_names=[f'log_theta{i + 1}' for i in range(d)]))


# %%
hmc_az_summary()

# %% [markdown]
# ### Validation HMC sample

# %% [markdown]
# We generate an additional sample that we will use to evaluate the quality of fit for the proposed methods.

# %%
validation_hmc_seed = 98765


# %%
@subscriptable(n=len(theta_inits))
@cached_batch(item_type=np.ndarray, recalculate=recalculate, persist=True)
def validation_hmc_samples() -> list[np.ndarray]:
    inference_model = stan.build(
        stan_model_spec,
        data=data,
        random_seed=validation_hmc_seed,
    )
    stan_sample = inference_model.sample(
        num_chains=len(theta_inits),
        num_samples=n_samples_hmc,
        save_warmup=False,
        init=[{'log_theta': np.log(theta_init)} for theta_init in theta_inits]
    )
    return extract_chains(stan_sample, 'log_theta')


# %%
# force the calculation
validation_hmc_samples[0];

# %%
fig = plot_traces(validation_hmc_samples, titles=titles, var_labels=var_labels);
fig.suptitle('Traces from the validation sample');

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Traversal paths for the validation sample');
plot_paths(validation_hmc_samples, np.log(theta_inits), idx1=0, idx2=1, ax=axs[0], label1='$\\log \\theta_1$', label2='$\\log \\theta_2$');
plot_paths(validation_hmc_samples, np.log(theta_inits), idx1=2, idx2=3, ax=axs[1], label1='$\\log \\theta_3$', label2='$\\log \\theta_4$');

# %%
validation_sample = np.concatenate(validation_hmc_samples, axis=0)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
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
fig.suptitle('Lotka-Volterra solution from sensitivity model');
for i in range(q):
    axs[i].plot(t, sensitivity_forward[:, i]);
    axs[i].set_xlabel('t');
    axs[i].set_ylabel(f'$u_{i + 1}(t)$');

# %% [markdown]
# Plot the sensitivities:

# %%
fig, axs = plt.subplots(q, d, figsize=(14, 6), constrained_layout=True)
fig.suptitle('Sensitivities in Lotka-Volterra model');
for i in range(q):
    for j in range(d):
        axs[i][j].plot(t, sensitivity_forward[:, q + i * d + j]);
        axs[i][j].set_xlabel('t');
        axs[i][j].set_ylabel(f'$\\partial u_{{{i + 1}}} / \\partial \\theta_{{{j + 1}}}$');


# %% [markdown]
# ## Numerical Jacobian calculation

# %% [markdown]
# We need to redefine the function since ``jax.experimental.ode.odeint`` passes the state variable in the first argument and time in the second argument:

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
fig.suptitle('Lotka-Volterra solution from sensitivity model implemented with JAX');
for i in range(q):
    axs[i].plot(t, sol2[:, i]);
    axs[i].set_xlabel('t');
    axs[i].set_ylabel(f'$u_{i + 1}(t)$');


# %% [markdown]
# Calculate the sensitivities:

# %%
@cached(recalculate=recalculate, persist=True)
def sensitivity_jax() -> np.ndarray:
    return np.stack(jacobian(solve_lotka_volterra2)(theta))


# %% [markdown]
# We confirm that the numerical method agrees with the results from forward sensitivity equations:

# %%
fig, axs = plt.subplots(q, d, figsize=(14, 6), constrained_layout=True)
fig.suptitle('Comparison of sensitivities from forward equations and numerical differentiation');
for i in range(2):
    for j in range(d):
        axs[i][j].plot(t, sensitivity_forward[:, 2 + i * d + j], label='Forward equations');
        axs[i][j].plot(t, sensitivity_jax()[j, :, i], label='JAX');
        axs[i][j].set_xlabel('t');
        axs[i][j].set_ylabel(f'$\\partial u_{{{i + 1}}} / \\partial \\theta_{{{j + 1}}}$');
        axs[i][j].legend();

# %% [markdown]
# ## Calculating the gradient of the log-posterior

# %% [markdown]
# The Stein Thinning methog requires the gradient of the log-posterior $\nabla \log p$ as input. Below we follow section S3 of the Supplementary Material to derive it.
#
# Since $p(\pmb{\theta}) \propto \mathcal{L}(\pmb{\theta}) \pi(\pmb{\theta})$, we have 
# $$\nabla_{\pmb{\theta}} \log p(\pmb{\theta}) = \nabla_{\pmb{\theta}} \log \mathcal{L}(\pmb{\theta}) + \nabla_{\pmb{\theta}} \log \pi(\pmb{\theta}).$$
# Assuming independent errors in observations yields
# $$\mathcal{L}(\pmb{\theta}) = \prod_{i=1}^N \phi_i(u(t_i)),$$
# and thus 
# $$
# \frac{\partial}{\partial \theta_s} \log \mathcal{L}(\pmb{\theta}) 
# = \sum_{i=1}^N \frac{\partial}{\partial \theta_s} \log \phi_i(u(t_i))
# = \sum_{i=1}^N \sum_{r=1}^q \frac{\partial}{\partial u_r} (\log \phi_i) \frac{\partial u_r}{\partial \theta_s},
# $$
# which can be written in matrix notation as
# $$(\nabla_{\pmb{\theta}} \log \mathcal{L})(\pmb{\theta}) = \sum_{i=1}^N \left(\frac{\partial \mathbf{u}}{\partial \pmb{\theta}}\right)^T\! (t_i)\, (\nabla_u \log \phi_i)(u(t_i)),$$
# where
# $$\left(\frac{\partial \mathbf{u}}{\partial \pmb{\theta}}\right)_{r,s} = \frac{\partial u_r}{\partial \theta_s}$$
# is the matrix of sensitivities, as obtained earlier.
#
# Note that this does not match the expression provided on page 16 of the Supplementary Material:
# $$(\nabla \log \mathcal{L})(x) = -\sum_{i=1}^N \frac{\partial u}{\partial x}(t_i) (\nabla \log \phi_i)(u(t_i)),$$
# where the Jacobian is not transposed and there is a minus sign in front of the expression.
#
# For a multivariate normal distribution of the errors:
# $$\phi_i(u(t_i)) \propto \exp\left( -\frac{1}{2} (y_i - u(t_i))^T C^{-1} (y_i - u(t_i)) \right)$$
# we obtain
# $$(\nabla_u \log \phi_i)(u(t_i)) = C^{-1}(y_i - u(t_i)).$$
#
# We assume independent standard normal priors for all components $\xi_i = \log \theta_i$, therefore
# $$\pi(\pmb{\theta}) = \prod_{i=1}^d \pi_i(\log \theta_i) \propto \exp\left(-\frac{1}{2}\sum_{i=1}^d (\log \theta_i)^2\right)$$
# and
# $$\nabla_{\pmb{\theta}} \log \pi(\pmb{\theta}) = -\frac{\log \pmb{\theta}}{\pmb{\theta}},$$
# where both the logarithm and division are performed component-wise.

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
    return grad_log_likelihood(theta) - np.log(theta) / theta


# %%
grad_log_posterior(theta)


# %% [markdown]
# ## Parallel calculation of gradients

# %% [markdown]
# We note that calculating gradients after a MCMC run is what is called "embarrassingly parallelisable" and the time required for this step can be effectively eliminated given sufficient computational resources. This is in contrast to the MCMC run itself, which is inherently sequential.
#
# Here we demonstrate how the popular package ``Dask`` can be used to parallelise this computation across cores of a local machine.
#
# See the notebook in ``examples/Dask_AWS.ipynb`` for a comparison between sequential calculation, parallel computation locally and on AWS.

# %% [markdown]
# We can save time by calculating the gradients for unique samples only:

# %% [markdown]
# A helper function to calculate gradients using unique values only:

# %%
def parallelise_for_unique(func, sample, row_chunk_size=200):
    """Calculate gradients for samples"""
    # we can save time by calculating gradients for unique samples only
    unique_samples, inverse_index = np.unique(sample, axis=0, return_inverse=True)
    res = apply_along_axis_parallel(func, 1, unique_samples, row_chunk_size, client)
    return res[inverse_index]


# %% [markdown]
# Calculate the gradients for the random-walk samples:

# %%
@subscriptable(n=len(theta_inits))
@cached(recalculate=recalculate, persist=True)
def rw_grads(i: int) -> np.ndarray:
    return parallelise_for_unique(grad_log_posterior, np.exp(rw_samples[i]))


# %% [markdown]
# Calculate the gradients for HMC samples:

# %%
@subscriptable(n=len(theta_inits))
@cached(recalculate=recalculate, persist=True)
def hmc_grads(i: int) -> np.ndarray:
    return parallelise_for_unique(grad_log_posterior, np.exp(hmc_samples[i]))


# %% [markdown]
# # Apply Stein thinning

# %% [markdown]
# ### Random-walk sample

# %%
n_points_calculate = 10_000
n_points_thinned = 20
n_points_display = 20


# %%
@subscriptable(n=len(theta_inits))
@cached_batch(item_type=np.ndarray, recalculate=recalculate, persist=True)
def rw_thinned_idx() -> list[np.ndarray]:
    # we have to instantiate the array here as the caching function currently cannot be serialised
    samples = list(rw_samples)
    gradients = list(rw_grads)
    def calculate(i):
        return thin(np.exp(samples[i]), gradients[i], n_points_calculate, preconditioner='med')
    return map_parallel(calculate, range(len(theta_inits)))


# %% [markdown]
# Force recalculation when necessary:

# %%
# %%time
#rw_thinned_idx.recalculate(persist=True);

# %% [markdown]
# This reproduces the results shown in Figure S20 in the Supplementary Material:

# %%
fig = plot_sample_thinned(rw_samples, rw_thinned_idx, titles, var_labels, n_points=n_points_display);
fig.savefig(figures_path / 'lotka-volterra-stein-thinning.png', dpi=300);
fig.suptitle('Results of applying Stein thinning to samples from the random-walk Metropolis-Hastings algorithm');


# %% [markdown]
# #### Log-transformation
#
# Since inference is performed in log-space, it is natural to try Stein thinning in log-space as well.
#
# If $\xi_i = \log \theta_i$, then by the chain rule we have
# $$\frac{\partial f}{\partial \xi_i} = \sum_{j=1}^d \frac{\partial f}{\partial \theta_j} \frac{\partial \theta_j}{\partial \xi_i},$$
# thus
# $$\nabla_{\pmb{\xi}} \log p(\pmb{\xi}) = J^{-T} \nabla_{\pmb{\theta}} \log p(\pmb{\theta}),$$
# where the Jacobian is $J = \text{diag}(\theta_1^{-1}, \dots, \theta_d^{-1})$, so $J^{-T} =  \text{diag}(\theta_1, \dots, \theta_d)$.

# %%
@subscriptable(n=len(theta_inits))
@cached_batch(item_type=np.ndarray, recalculate=recalculate, persist=True)
def rw_st_log_idx() -> list[np.ndarray]:
    # we have to instantiate the array here as the caching function currently cannot be serialised
    samples = list(rw_samples)
    gradients = list(rw_grads)
    def calculate(i):
        return thin(samples[i], np.exp(samples[i]) * gradients[i], n_points_calculate, preconditioner='med')
    return map_parallel(calculate, range(len(theta_inits)))


# %% [markdown]
# Force recalculation when necessary:

# %%
# %%time
#rw_st_log_idx.recalculate(persist=True);

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### HMC sample

# %%
@subscriptable
@cached(recalculate=recalculate, persist=True)
def hmc_thinned_idx(i: int) -> np.ndarray:
    return thin(np.exp(hmc_samples[i]), hmc_grads[i], n_points_thinned, preconditioner='med')


# %%
fig = plot_sample_thinned(hmc_samples, hmc_thinned_idx, titles, var_labels);
fig.suptitle('Results of applying Stein thinning to samples from the HMC algorithm');


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Naive thinning

# %% [markdown]
# The baseline for comparison is the naive thinning approach where we retain each $i$-th element of the sample.

# %%
def naive_thin(sample_size, thinned_size):
    return np.linspace(0, sample_size - 1, thinned_size).astype(int)


# %%
@subscriptable
@cached(recalculate=recalculate, persist=True)
def rw_naive_idx(i: int) -> np.ndarray:
    return naive_thin(rw_samples[i].shape[0], n_points_thinned)


# %%
fig = plot_sample_thinned(rw_samples, rw_naive_idx, titles, var_labels);
fig.savefig(figures_path / 'lotka-volterra-naive-thinning.png', dpi=300);
fig.suptitle('Results of applying naive thinning to samples from the random-walk Metropolis-Hastings algorithm');


# %% [markdown]
# # Gradient-free Stein thinning

# %% [markdown]
# We recalculate the (unnormalised) log target density for all samples. Note that in principle we could have stored it during the MCMC run rather than recalculating it.

# %%
@subscriptable
@cached(recalculate=recalculate, persist=True)
def rw_log_p(i: int) -> np.ndarray:
    return parallelise_for_unique(log_target_density, rw_samples[i])


# %%
@subscriptable
@cached(recalculate=recalculate, persist=True)
def hmc_log_p(i: int) -> np.ndarray:
    return parallelise_for_unique(log_target_density, hmc_samples[i])


# %% [markdown]
# ## Full sample

# %% [markdown]
# ### Laplace proxy

# %%
def laplace_approximation(logpdf, x0):
    res = minimize(lambda x: -logpdf(x), x0, method='BFGS', options={'gtol': 1e-3})
    assert res.success
    return res.x, res.hess_inv


# %%
# %%time
laplace_mean, laplace_cov = laplace_approximation(log_target_density, np.mean(rw_samples[0], axis=0))

# %%
laplace_mean

# %%
laplace_cov


# %%
def gaussian_thin(sample, log_p, mean, cov, thinned_size):
    log_q = mvn.logpdf(sample, mean=mean, cov=cov)
    gradient_q = -np.einsum('ij,kj->ki', np.linalg.inv(cov), sample - mean)
    return thin_gf(sample, log_p, log_q, gradient_q, thinned_size, range_cap=200)


# %% [markdown]
# The method clearly fails in this case:

# %%
gaussian_thin(rw_samples[0], rw_log_p[0], laplace_mean, laplace_cov, n_points_thinned)

# %%
points_to_highlight = [231]

fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
highlight_points(rw_samples[0], points_to_highlight, [(0, 1), (2, 3)], axs, var_labels);

# %% [markdown]
# We calculate the proxy density at element 231:

# %%
log_q = mvn.logpdf(rw_samples[0], mean=laplace_mean, cov=laplace_cov)

# %%
log_q[231]

# %%
rw_log_p[0][231]

# %% [markdown]
# #### Numerical stability of optimisation

# %% [markdown]
# Using the default parameters in `scipy.optimize.minimize` results in a failure to find the optimum:

# %%
x0 = np.mean(rw_samples[0], axis=0)
res = minimize(lambda x: -log_target_density(x), x0)
res

# %% [markdown]
# Nelder-Mead succeeds:

# %%
x0 = np.mean(rw_samples[0], axis=0)
res = minimize(lambda x: -log_target_density(x), x0, method='Nelder-Mead')
res

# %% [markdown]
# However, the Hessian evaluated at the maximum does not appear to be negative definite:

# %%
from numdifftools import Hessian

# %%
# %%time
hess = Hessian(log_target_density)(res.x)
hess


# %% [markdown]
# The returned Hessian matrix is not negative definite either:

# %%
def is_positive_definite(x):
    return np.all(np.linalg.eigvals(x) > 0)


# %%
is_positive_definite(-hess)

# %% [markdown]
# ### Gaussian proxy

# %%
sample_mean = np.mean(rw_samples[0], axis=0)
sample_cov = np.cov(rw_samples[0], rowvar=False, ddof=4)

# %%
sample_mean

# %%
sample_cov

# %%
idx = gaussian_thin(rw_samples[0], rw_log_p[0], sample_mean, sample_cov, n_points_thinned)
idx

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
highlight_points(rw_samples[0], idx, [(0, 1), (2, 3)], axs, var_labels, highlighted_point_size=4);

# %% [markdown]
# ### Student t proxy

# %%
from scipy.optimize import OptimizeResult
from scipy.special import gamma


# %%
def extract_t_params(par, d):
    # upper-triangular elements of an n-by-n matrix
    n_cov = d * (d + 1) // 2

    # the means of the multivariate t are in the first `d` elements
    mu = par[:d]
    # the upper triangular elements of A are in the following `n_cov` elements
    A = np.zeros((d, d))
    A[np.triu_indices(d)] = par[d:d + n_cov]
    # the scale matrix
    scale = A.T @ A
    # the degrees of freedom value is the last element of `par`
    df = par[d + n_cov]

    return mu, scale, df

def loglik_mvt(Y: np.ndarray, par: np.ndarray) -> float:
    """The log-likelihood of the multivariate t-distribution

    Parameters
    ----------
    Y: np.ndarray
        the input data: rows are observations, columns are variables
    par: np.ndarray
        the parameters of the multivariate t-distribution

    Returns
    -------
    float
        the log-likelihood of the multivariate t-distribution evaluated at the given point


    Notes
    -----
    The parameters are passed as a one-dimensional array: first the means,
    then the elements of the upper-triangular matrix A, where A.T @ A ~ Cov(Y),
    followed by the degrees of freedom.
    """
    mu, scale, df = extract_t_params(par, Y.shape[1])
    return -np.sum(stats.multivariate_t.logpdf(Y, loc=mu, shape=scale, df=df))

def fit_mvt(
        Y: np.ndarray,
        mu_bounds: tuple[float, float],
        a_bounds: tuple[float, float],
        df_bounds: tuple[float, float],
        df_init: float = 4.,
) -> OptimizeResult:
    """Fit a multivariate t-distribution using maximum likelihood

    Parameters
    ----------
    Y: np.ndarray
        the input data: rows are observations, columns are variables
    mu_bounds: Tuple[float, float]
        the lower and upper bounds for means
    a_bounds: Tuple[float, float]
        the lower and upper bounds for values in the matrix A, where A.T @ A ~ Cov(Y)
    df_bounds: Tuple[float, float]
        the lower and upper bounds for the degree of freedom parameter

    Returns
    -------
    OptimizeResult
        the result of fitting a multivariate t-distribution
    """
    d = Y.shape[1]  # the number of variables
    n_cov = d * (d + 1) // 2  # upper-triangular elements of an n-by-n matrix

    # the starting values for the search
    sample_mean = np.mean(Y, axis=0)
    sample_cov = np.cov(Y, rowvar=False, ddof=d)
    A = np.linalg.cholesky(sample_cov).T
    start = np.concatenate([sample_mean, A[np.triu_indices(d)], [df_init]])

    # the bounds for the search
    lower = np.array([mu_bounds[0]] * d + [a_bounds[0]] * n_cov + [df_bounds[0]])
    upper = np.array([mu_bounds[1]] * d + [a_bounds[1]] * n_cov + [df_bounds[1]])

    def objective_func(beta):
        return loglik_mvt(Y, beta)

    bounds = list(zip(lower, upper))

    return minimize(objective_func, start, method='L-BFGS-B', bounds=bounds)


# %%
@subscriptable(n=len(theta_inits))
@cached_batch(item_type=OptimizeResult, recalculate=recalculate, persist=True)
def rw_t_fit() -> list[np.ndarray]:
    # we have to instantiate the array here as the caching function currently cannot be serialised
    samples = list(rw_samples)
    def calculate(i):
        return fit_mvt(samples[i], mu_bounds=(-0.5, 0.5), a_bounds=(-0.1, 0.1), df_bounds=(2, 15), df_init=3.)
    return map_parallel(calculate, range(len(theta_inits)))


# %% [markdown]
# Force recalculation when necessary:

# %%
#rw_t_fit.recalculate(persist=True);

# %%
t_mu, t_scale, t_df = extract_t_params(rw_t_fit[0].x, d)

# %%
t_mu

# %%
t_scale

# %%
t_df

# %%
t_df = np.round(t_df)
t_df

# %%
log_q = stats.multivariate_t.logpdf(rw_samples[0], loc=t_mu, shape=t_scale, df=t_df)


# %% [markdown]
# The density of the multivaritate Student's t distribution is given by
# $$f(\mathbf{x}) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) \nu^{\frac{d}{2}} \pi^{\frac{d}{2}} |\Sigma|^{\frac{1}{2}}} \left[1 + \frac{1}{\nu} (\mathbf{x} - \pmb{\mu})^T \Sigma^{-1} (\mathbf{x} - \pmb{\mu})\right]^{-\frac{\nu + d}{2}}.$$
# The gradient of the log-density is then
# $$\nabla_{\mathbf{x}} \log f(\mathbf{x}) = -\frac{\nu + d}{\nu} \frac{\Sigma^{-1} (\mathbf{x} - \pmb{\mu})}{1 + \frac{1}{\nu} (\mathbf{x} - \pmb{\mu})^T \Sigma^{-1} (\mathbf{x} - \pmb{\mu})}$$

# %% [markdown]
# We implement the log-density of the multivariate t and confirm that it matches what is returned by `scipy`:

# %%
def t_log_pdf(x, mu, sigma, df):
    d = x.shape[1]
    sigma_inv = np.linalg.inv(sigma)
    x_mu = x - mu
    return (
        np.log(gamma((df + d) / 2))
        - np.log(gamma(df  / 2))
        - d * (np.log(df) + np.log(np.pi)) / 2
        - np.log(np.linalg.det(sigma)) / 2
        -(df + d) / 2 * np.log(1 + np.einsum('ij,jk,ik->i', x_mu, sigma_inv, x_mu) / df)
    )


# %%
np.testing.assert_allclose(t_log_pdf(rw_samples[0], t_mu, t_scale, t_df), log_q)


# %%
def t_grad_log_pdf(x, mu, sigma, df):
    d = x.shape[1]
    sigma_inv = np.linalg.inv(sigma)
    x_mu = x - mu
    direction_scaled = np.einsum('jk,ik->ij', sigma_inv, x_mu)
    mahalanobis_d = np.einsum('ij,jk,ik->i', x_mu, sigma_inv, x_mu)
    return -(df + d) / df / (1 + mahalanobis_d / df).reshape(-1, 1) * direction_scaled


# %%
gradient_q = t_grad_log_pdf(rw_samples[0], t_mu, t_scale, t_df)

# %%
idx = thin_gf(rw_samples[0], rw_log_p[0], log_q, gradient_q, n_points_thinned, range_cap=200)
idx

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
highlight_points(rw_samples[0], idx, [(0, 1), (2, 3)], axs, var_labels, highlighted_point_size=4);

# %% [markdown]
# ## Sample with burn-in removed manually

# %% [markdown]
# We plot the difference in log-probability versus the squared Euclidean distance from the sample mode:

# %%
i = 0
sample = rw_samples[i]
sample_mean = np.mean(sample, axis=0)
ref_idx = np.argmax(rw_log_p[i])
dists = cdist(sample[ref_idx].reshape(1, -1), sample).squeeze()
prob_diff = rw_log_p[i] - rw_log_p[i][ref_idx]

fig, ax = plt.subplots(constrained_layout=True)

ax.scatter(dists ** 2, prob_diff, s=1);
ax.set_xlabel('$\\|x - x^* \\|^2$');
ax.set_ylabel('$\\log p(x) - \\log p(x^*)$');

inset_xlim = [0, 0.01]
inset_ylim = [-20, 5]

ax_ins = ax.inset_axes([0.1, 0.1, 0.5, 0.5], xlim=inset_xlim, ylim=inset_ylim)
ax.indicate_inset_zoom(ax_ins, edgecolor="black")

ax_ins.scatter(dists ** 2, prob_diff, s=1);
ax_ins.set_xlim(inset_xlim);
ax_ins.set_ylim(inset_ylim);

# %% [markdown]
# We can us the threshold of -15 to locate the bulk of the sample.

# %%
cond = prob_diff > -15 

# %% [markdown]
# This retains most of the points:

# %%
np.sum(cond) / rw_samples[0].shape[0]

# %% [markdown]
# The resulting subsample:

# %%
subsample = rw_samples[0][cond]
subsample_log_p = rw_log_p[0][cond]

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
highlight_points(subsample, [], [(0, 1), (2, 3)], axs, var_labels, sample_point_color=None);

# %% [markdown]
# ### Laplace proxy

# %% [markdown]
# The parameters of the Laplace proxy do not change, since they are estimated from the posterior distribution rather than the sample.

# %% [markdown]
# Thinning fails in this case again:

# %%
gaussian_thin(subsample, subsample_log_p, laplace_mean, laplace_cov, n_points_thinned)

# %%
subsample_log_p[426876]

# %%
subsample_log_q = mvn.logpdf(subsample, mean=laplace_mean, cov=laplace_cov)

# %%
subsample_log_q[426876]

# %% [markdown]
# Again, the tail of the proxy distribution is too thin relative to the target.

# %% [markdown]
# ### Gaussian proxy

# %%
subsample_mean = np.mean(subsample, axis=0)
subsample_cov = np.cov(subsample, rowvar=False, ddof=4)

# %%
subsample_mean

# %%
subsample_cov

# %% [markdown]
# The values calculated from the subsample are very close to those obtained from the full sample:

# %%
sample_mean

# %%
sample_cov

# %%
idx = gaussian_thin(subsample, subsample_log_p, subsample_mean, subsample_cov, n_points_thinned)
idx

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
highlight_points(subsample, idx, [(0, 1), (2, 3)], axs, var_labels, highlighted_point_size=4);


# %% [markdown]
# # Energy distance comparison

# %%
def fit_quality(subsample, validation_sample_step=10):
    return np.sqrt(dcor.energy_distance(validation_sample[::validation_sample_step], subsample))


# %%
def naive_idx(n, m):
    return np.linspace(0, n - 1, m).astype(int)


# %%
thinned_size_series = []
thinned_size_series.append(np.linspace(5, 100, 50).astype(int))
thinned_size_series.append(np.linspace(100, n_points_calculate, 200).astype(int))
thinned_sizes = np.concatenate(thinned_size_series)


# %%
@cached(recalculate=recalculate, persist=True)
def rw_energy_distance(i_chain, idx_name) -> np.ndarray:
    sample = rw_samples[i_chain]
    idx = globals()[idx_name][i_chain]
    energy_distances = np.fromiter((fit_quality(sample[idx[:thinned_size]]) for thinned_size in thinned_sizes), float)
    return np.stack([thinned_sizes, energy_distances], axis=1)


# %%
@cached(recalculate=recalculate, persist=True)
def rw_energy_distance_naive(i_chain) -> np.ndarray:
    sample = rw_samples[i_chain]
    n = sample.shape[0]
    energy_distances = np.fromiter((fit_quality(sample[naive_idx(n, thinned_size)]) for thinned_size in thinned_sizes), float)
    return np.stack([thinned_sizes, energy_distances], axis=1)


# %%
indices_to_plot = {
    'rw_thinned_idx': 'Stein',
    'rw_st_log_idx': 'Stein log',
    'rw_naive': 'Naive',
}


# %%
def get_indices(name):
    if name == 'rw_naive':
        return rw_energy_distance_naive
    else:
        return lambda i: rw_energy_distance(i, name)


# %%
# %%time
fig, axs = plt.subplots(1, len(theta_inits), figsize=(17, 3), constrained_layout=True)
for j in range(len(theta_inits)):
    for idx_name, label in indices_to_plot.items():
        res = get_indices(idx_name)(j)
        axs[j].plot(res[:, 0], res[:, 1], label=label);
    axs[j].set_xlabel('Thinned sample size');
    axs[j].set_ylabel('Energy distane');
    axs[j].set_title(f'Chain {j + 1}');
    axs[j].legend();
    axs[j].set_xscale('log');
fig.savefig(figures_path / 'lotka-volterra-stein-thinning-energy-distance.pdf');

# %% [markdown]
# Notebook execution took:

# %%
time.time() - start_time
