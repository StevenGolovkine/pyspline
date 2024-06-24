#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Formatter
---------

"""

import numpy as np
import numpy.typing as npt


def format_X_y(
    X: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    weights: npt.NDArray[np.float_] | None = None,
) -> tuple[
    list[npt.NDArray[np.float_]], npt.NDArray[np.float_], npt.NDArray[np.float_]
]:
    """Input formatter for multidimensional estimator.

    Parameters
    ----------
    X: npt.NDArray[np.float_], shape=(n_obs, n_dimension)
        An array containing the predictor variable values.
    y: npt.NDArray[np.float_], shape=(n_obs,)
        An array containing the response variable values.
    weights: npt.NDArray[np.float_] | None, default=None
        An array containing the sampled weights with the same shape as `y`. If
        `None`, the weights are set to 1 for the observed values and 0
        otherwise.

    Returns
    -------
    tuple[
        list[npt.NDArray[np.float_]],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_]
    ]
        A tuple containing the formatted X, y and an array of weights.

    """
    new_X = [np.unique(column) for column in X.T]
    X_matrices = np.meshgrid(*new_X, indexing="ij")

    new_y = np.zeros_like(X_matrices[0])
    for x, obs in zip(X, y):
        indices = tuple(
            np.flatnonzero(points == point)[0]
            for point, points in zip(x, new_X)
        )
        new_y[indices] = obs

    if weights is None:
        weights = np.ones_like(new_y)
        weights[new_y == 0] = 0
    return new_X, new_y, weights
