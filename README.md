# Gradient-Free Optimal Postprocessing of MCMC Output
The project aims to extend the work in 

> 1. `Riabiz, M., Chen, W. Y., Cockayne, J., Swietach, P., Niederer, S. A., Mackey, L., Oates, C. J. (2022). Optimal thinning of MCMC output. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(4), 1059-1081`.

by implementing the proposals presented in

> 2. `Fisher, M. A., Oates, C. (2022). Gradient-free kernel Stein discrepancy. arXiv preprint arXiv:2207.02636.`

and

> 3. `Huang, C., Joseph, V. R. (2023). Enhancing Sample Quality through Minimum Energy Importance Weights. arXiv preprint arXiv:2310.07953.`

We replicate the results presented from [1] for the Lotka-Volterra model in ``code/Lotka-Volterra.ipynb``.

``code/Gradient_free_thinning.ipynb`` demonstrates using gradient-free kernel Stein density as proposed in [2] for a bivariate Gaussian mixture.

The ``code/examples`` directory also contains several examples of using the relevant Python packages.
