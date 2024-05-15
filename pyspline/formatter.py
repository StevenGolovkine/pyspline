#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Formatter
---------

"""

import numpy as np
import numpy.typing as npt


def format_X_y(
    X: npt.NDArray[np.float_], y: npt.NDArray[np.float_]
) -> tuple[list[npt.NDArray[np.float_]], npt.NDArray[np.float_]]:
    """Input formatter for multidimentsional estimator.

    Parameters
    ----------
    X: npt.NDArray[np.float_], shape=(n_obs, n_dimension)
        An array containing the predictor variable values.
    y: npt.NDArray[np.float_], shape=(n_obs,)
        An array containing the response variable values.

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
    return new_X, new_y
