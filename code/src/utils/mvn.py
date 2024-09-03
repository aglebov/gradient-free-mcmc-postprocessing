import numpy as np
from scipy.stats import multivariate_normal as mvn
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as jmvn


def make_mvn_mixture(weights, means, covs):
    # invert covariances
    covs_inv = np.linalg.inv(covs)

    k, d = means.shape
    assert weights.shape == (k,)
    assert covs.shape == (k, d, d)
    
    def rvs(size, random_state):
        component_samples = [
            mvn.rvs(mean=means[i], cov=covs[i], size=size, random_state=random_state)
            for i in range(len(weights))
        ]
        indices = random_state.choice(len(weights), size=size, p=weights)
        return np.take_along_axis(
            np.stack(component_samples, axis=1),
            indices.reshape(size, 1, 1),
            axis=1,
        ).squeeze()

    def logpdf(x, axes=slice(None)):
        f = np.stack([mvn.pdf(x, mean=means[i][axes], cov=covs[i][axes, axes]) for i in range(len(weights))]).reshape(len(weights), -1)
        return np.log(np.einsum('i,il->l', weights, f))
    
    def score(x):
        # centered sample
        xc = x[np.newaxis, :, :] - means[:, np.newaxis, :]
        # pdf evaluations for all components and all elements of the sample
        f = np.stack([mvn.pdf(x, mean=means[i], cov=covs[i]) for i in range(len(weights))]).reshape(len(weights), -1)
        # numerator of the score function
        num = np.einsum('i,il,ijk,ilk->lj', weights, f, covs_inv, xc)
        # denominator of the score function
        den = np.einsum('i,il->l', weights, f)
        return -num / den[:, np.newaxis]
    
    def logpdf_jax(x):
        probs = jmvn.pdf(
            x.reshape(-1, 1, d),
            mean=means.reshape(1, k, d),
            cov=covs.reshape(1, k, d, d)
        )
        return jnp.squeeze(jnp.log(jnp.sum(weights * probs, axis=1)))
    
    return rvs, logpdf, score, logpdf_jax
