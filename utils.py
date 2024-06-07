import numpy as np
import random
import scipy
import torch


def set_random_seeds(seed=0):
    """Set random seeds across libraries for reproducibility.

    Parameters
    ----------
    seed : int, optional (default: 0)
        The seed value for pseudorandom number generation.
    """
    seed = abs(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kde(data):
    """Compute the kernel density estimation (KDE) for a given dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n, d), where n is the number of samples and d is the dimensionality.

    Returns
    -------
    scipy.stats.gaussian_kde
        A KDE object that has been fit to the input data.
    """
    return scipy.stats.gaussian_kde(data.T)


def kl_divergence(active_kde, quiescent_kde, n=2500):
    """Compute a Monte Carlo estimate of the Kullback-Leibler (KL) divergence.

    Computes the KL divergence between two kernel density estimations (KDEs), or a continuous
    distribution and a KDE, using a Monte Carlo estimate.

    Parameters
    ----------
    active_kde : scipy.stats.gaussian_kde
        A KDE object representing the active distribution.
    quiescent_kde : scipy.stats.gaussian_kde or scipy.stats.rv_continuous
        A KDE object or continuous distribution representing the quiescent distribution.
    n : int, optional (default: 2500)
        The number of points to resample for computing the KL divergence.

    Returns
    -------
    np.float64
        The KL divergence between the quiescent and active distributions.

    Notes
    -----
    1. The KL divergence is estimated using a Monte Carlo method:
       KL(P || Q) ≈ (1/n) * Σ[log(P(x_i) / Q(x_i))] for i=1 to n,
       where P(x_i) and Q(x_i) are the probability densities of the quiescent and active
       distributions respectively, and x_i are points sampled from the quiescent distribution.
    2. The KL divergence is undefined if the support of Q(x) is not a subset of the support of
       P(x). In such cases, this function may return NaN.
    3. It is assumed that all KDE objects are properly fitted.
    """
    if isinstance(quiescent_kde, scipy.stats.gaussian_kde):
        points = quiescent_kde.resample(n)
    elif isinstance(getattr(quiescent_kde, "dist", None), scipy.stats.rv_continuous):
        points = quiescent_kde.rvs((n, active_kde.d)).T
    else:
        raise NotImplementedError("unknown distribution type.")
    quiescent_pdf = quiescent_kde.pdf(points)
    active_pdf = active_kde.pdf(points)
    return np.log((quiescent_pdf / active_pdf).clip(1e-7, 1e7)).mean()


def output_variance(data, shift=1000):
    """Compute the variance summed over dimensions for output trajectories.

    Parameters
    ----------
    data : np.ndarray
        A batch of output trajectories for which to compute the metric.
    shift : int (default: 1000)
        The number of initial timesteps to ignore.

    Returns
    -------
    np.float64
        The variance summed over dimensions for output trajectories, averaged across the batch.
    """
    b, _, d = data.shape
    covs = np.array([np.cov(data[i, shift:].T) for i in range(b)])
    if d > 1:
        covs = np.trace(covs, axis1=1, axis2=2)
    return covs.mean()


def stepwise_distance(data):
    """Compute the average point-to-point distance for output trajectories.

    Parameters
    ----------
    data : np.ndarray
        A batch of output trajectories for which to compute the metric.

    Returns
    -------
    np.ndarray
        The average point-to-point distance for each output trajectory.
    """
    return ((np.diff(data, axis=1) ** 2).sum(axis=-1) ** 0.5).mean(axis=-1)
