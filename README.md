# Gradient-Free Optimal Postprocessing of MCMC Output
The project aims to extend the work in 

> 1. `Riabiz, M., Chen, W. Y., Cockayne, J., Swietach, P., Niederer, S. A., Mackey, L., Oates, C. J. (2022). Optimal thinning of MCMC output. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(4), 1059-1081`.

by implementing the idea presented in

> 2. `Fisher, M. A., Oates, C. (2022). Gradient-free kernel Stein discrepancy. arXiv preprint arXiv:2207.02636.`

We replicate the results from [1] for the Lotka-Volterra model in ``code/lotka_volterra/Stein_thinning.ipynb``.

``code/notebooks/gaussian_mixture/Gaussian_mixture.ipynb`` demonstrates using gradient-free kernel Stein density as proposed in [2] for a bivariate Gaussian mixture.

The ``code/notebooks/examples`` directory also contains several examples of using the relevant Python packages.

To run the code, navigate to the `code` directory, create and activate a virtual environment and run the following command:
```
pip install -e .
```