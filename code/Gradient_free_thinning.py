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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import multivariate_normal as mvn

from jax import grad
import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal as jmvn
from jax.scipy.stats import gaussian_kde as jgaussian_kde

from stein_thinning.thinning import thin, thin_gf


# %% [markdown]
# ## Generate from a multivariate normal mixture model

# %% [markdown]
# For multivariate normal distributions with pdfs
# $$f_i(x) = \frac{1}{(2\pi)^{d/2} |\Sigma_i|^{1/2}}\exp\left(-\frac{1}{2}(x - \mu_i)^T \Sigma_i^{-1}(x-\mu_i)\right),$$
# where $x \in \mathbb{R}^d$, the mixture pdf with $k$ components is given by
# $$f(x) = \sum_{i=1}^k w_i f_i(x),$$
# thus the score function is obtained as
# $$\nabla \log f(x) = \frac{\sum_{i=1}^k w_i \nabla f_i(x)}{\sum_{i=1}^k w_i f_i(x)} = -\frac{\sum_{i=1}^k w_i f_i(x) \Sigma_i^{-1}(x - \mu_i)}{\sum_{i=1}^k w_i f_i(x)}.$$

# %% [markdown]
# Define the functions for the parameters of a multivariate Gaussian mixture:

# %%
def make_mvn_mixture(weights, means, covs):
    # invert covariances
    covs_inv = np.linalg.inv(covs)

    k, d = means.shape
    assert weights.shape == (k,)
    assert covs.shape == (k, d, d)
    
    def rvs(size, random_state):
        component_samples = [
            mvn.rvs(mean=means[i], cov=covs[i], size=size, random_state=random_state) for i in range(len(weights))
        ]
        indices = rng.choice(len(weights), size=size, p=weights)
        return np.take_along_axis(np.stack(component_samples, axis=1), indices.reshape(size, 1, 1), axis=1).squeeze()

    def logpdf(x):
        f = np.stack([mvn.pdf(x, mean=means[i], cov=covs[i]) for i in range(len(weights))])
        return np.log(np.einsum('i,il->l', weights, f))
    
    def score(x):
        # centered sample
        xc = x[np.newaxis, :, :] - means[:, np.newaxis, :]
        # pdf evaluations for all components and all elements of the sample
        f = np.stack([mvn.pdf(x, mean=means[i], cov=covs[i]) for i in range(len(weights))])
        # numerator or the score function
        num = np.einsum('i,il,ijk,ilk->lj', weights, f, covs_inv, xc)
        # denominator or the score function
        den = np.einsum('i,il->l', weights, f)
        return -num / den[:, np.newaxis]
    
    def logpdf_jax(x):
        probs = jmvn.pdf(x.reshape(-1, 1, d), mean=means.reshape(1, k, d), cov=covs.reshape(1, k, d, d))
        return jnp.squeeze(jnp.log(jnp.sum(weights * probs, axis=1)))
    
    return rvs, logpdf, score, logpdf_jax


# %% [markdown]
# Choose the parameters of the mixture:

# %%
w = np.array([0.3, 0.7])
means = np.array([
    [-1., -1.],
    [1., 1.],
])
covs = np.array([
    [
        [0.5, 0.25],
        [0.25, 1.],
    ],
    [
        [2.0, -np.sqrt(3.) * 0.8],
        [-np.sqrt(3.) * 0.8, 1.5],
    ]
])

# %%
rvs, logpdf, score, logpdf_jax = make_mvn_mixture(w, means, covs)

# %% [markdown]
# Obtain a sample from the mixture:

# %%
rng = np.random.default_rng(12345)
sample_size = 1000
sample = rvs(sample_size, random_state=rng)

# %%
fig, ax = plt.subplots()
ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3);
ax.set_title('Sample from a multivariate Gaussian mixture');

# %% [markdown]
# Verify log-pdf against the JAX implementation:

# %%
np.testing.assert_array_almost_equal(logpdf(sample), logpdf_jax(sample))

# %% [markdown]
# Verify the score function against the JAX implementation:

# %%
gradient = score(sample)

# %%
score_jax = grad(logpdf_jax)
gradient_jax = jnp.apply_along_axis(score_jax, 1, sample)

# %%
np.testing.assert_array_almost_equal(gradient, gradient_jax)

# %% [markdown]
# ## Thinning

# %% [markdown]
# Our aim here is to select a subsample of the posterior sample that best represents the posterior distribution.

# %%
thinned_size = 40

# %% [markdown]
# ### Importance resampling

# %% [markdown]
# The easiest way to obtain a subsample from the posterior sample is importance resampling. In this case, each point is selected independently.
#
# For resampling, we need need the posterior probability for each sample point:

# %%
log_p = logpdf(sample)
p = np.exp(log_p)

# %%
w = p / np.sum(p)
idx_ir = rng.choice(np.arange(sample.shape[0]), size=thinned_size, p=w)

# %% [markdown]
# ### Stein thinning

# %% [markdown]
# If the gradient of the log-posterior is available, we can use it to perform thinning based on kernel Stein discrepancy:

# %%
idx_st = thin(sample, gradient, thinned_size)

# %% [markdown]
# ### Simple gradient-free Stein thinning

# %% [markdown]
# When the gradient of the log-posterior is not available, we can resort to a gradient-free approximation. This requires us to select a proxy distribution whose gradient is easily computable. The simplest option is to select a multivariate Gaussian with moments matching the sample:

# %%
sample_mean = np.mean(sample, axis=0)
sample_cov = np.cov(sample, rowvar=False, ddof=1)

# %% [markdown]
# Gradient-free Stein thinning requires us to provide the log-pdf of the proxy distribution and its score function:

# %%
log_q = mvn.logpdf(sample, mean=sample_mean, cov=sample_cov)
gradient_q = -np.einsum('ij,kj->ki', np.linalg.inv(sample_cov), sample - sample_mean)

# %% [markdown]
# We get the indices of the points to select:

# %%
idx_gf = thin_gf(sample, log_p, log_q, gradient_q, thinned_size)

# %% [markdown]
# ### Gradient-free Stein thinning with a KDE proxy

# %% [markdown]
# We can obtain a better approximation of the posterior with a KDE of the sample:

# %%
kde = jgaussian_kde(sample.T, bw_method='silverman')


# %% [markdown]
# For simplicity, we obtain the gradient by numerical differentiation. Since the default choice of kernel for KDE is Gaussian, we could also obtain the gradient explicitly.

# %%
def logpdf_and_score(kde, sample):
    log_q = np.array(kde.logpdf(sample.T))
    kde_grad = grad(lambda x: kde.logpdf(x)[0])
    gradient_q = np.array(jnp.apply_along_axis(kde_grad, 1, sample))
    return log_q, gradient_q


# %%
log_q_kde, gradient_q_kde = logpdf_and_score(kde, sample)

# %%
idx_gf_kde = thin_gf(sample, log_p, log_q_kde, gradient_q_kde, thinned_size)

# %% [markdown]
# ### Gradient-free Stein thinning with a weighted KDE proxy

# %% [markdown]
# A further improvement on the KDE approach is to use the posterior probabilities of the sample points as weights in the KDE approximation:

# %%
wkde = jgaussian_kde(sample.T, bw_method='silverman', weights=w)
log_q_wkde, gradient_q_wkde = logpdf_and_score(wkde, sample)

# %%
idx_gf_wkde = thin_gf(sample, log_p, log_q_wkde, gradient_q_wkde, thinned_size)

# %% [markdown]
# ### Comparison

# %%
entries = [
    (idx_ir, 'Importance resampling'),
    (idx_st, 'Stein thinning'),
    (idx_gf, 'Gradient-free Stein thinning: Gaussian proxy'),
    (idx_gf_kde, 'Gradient-free Stein thinning: KDE proxy'),
    (idx_gf_wkde, 'Gradient-free Stein thinning: weighted KDE proxy'),
]

# %% [markdown]
# The number of unique point selected:

# %%
pd.Series([len(np.unique(idx)) for idx, _ in entries], index=[title for _, title in entries])

# %% [markdown]
# Plot the selected points:

# %%
n_cols = 2
n_rows = (len(entries) - 1) // n_cols + 1
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4));
for i, (idx, title) in enumerate(entries):
    ax = axs[i // 2][i % 2]
    ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, color='gray');
    ax.scatter(sample[idx, 0], sample[idx, 1], color='red');
    ax.set_title(title);