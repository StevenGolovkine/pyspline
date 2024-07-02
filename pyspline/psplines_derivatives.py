#!/usr/bin/env python
# -*-coding:utf8 -*
"""
P-Splines derivatives
---------------------

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Tuple

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)

from .arrays import rotated_h_transform
from .basis import basis_bsplines
from .psplines import PSplines


class PSplinesDerivative(BaseEstimator, RegressorMixin):  # type: ignore
    """P-Splines Derivative Smoothing.

    Parameters
    ----------
    order_derivative: int, default=1
        The order of the derivatives to estimate.
    penalty: Tuple[float], default=(1.0,)
        A tuple of penalty parameters for each dimension.
    n_segments: Tuple[int], default=(10,)
        The number of evenly spaced segments.
    degree: Tuple[int], default=(3,)
        The number of the degree of the basis.
    order_penalty: int, default=2
        The number of the order of the difference penalty.

    Notes
    -----
    This code is adapted from _[2]. See [1]_ for more details.

    References
    ----------
    .. [1] Eilers, P., Marx, B.D., (2021) Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X.,
        (2023) JOPS: Practical Smoothing with P-Splines.

    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(
        self,
        order_derivative: int = 1,
        penalty: Tuple[float] = (1.0,),
        *,
        n_segments: Tuple[int] = (10,),
        degree: Tuple[int] = (3,),
        order_penalty: int = 2,
    ):
        """Initializa PSplines object."""
        self.order_derivative = order_derivative
        self.penalty = penalty
        self.n_segments = n_segments
        self.degree = degree
        self.order_penalty = order_penalty

    def fit(
        self,
        X: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_],
        sample_weights: npt.NDArray[np.float_] | None = None,
        domains: list[tuple[np.float_]] | tuple[np.float_] | None = None,
    ) -> PSplinesDerivative:
        """Fit a P-splines model to the given data.

        The method fits a P-splines model to the given data using a B-splines
        basis and an optional weights matrix.

        Parameters
        ----------
        X: npt.NDArray[np.float_], shape=(n_obs, n_dimension)
            An array containing the predictor variable values.
        y: npt.NDArray[np.float_], shape=(n_obs,)
            An array containing the response variable values.
        sample_weights: npt.NDArray[np.float64] | None, default=None
            An array of shape `(n_obs,)` containing the weights for each
            observation. If not provided, all observations are assumed to have
            equal weight.
        domains: list[tuple[np.float_]] | tuple[np.float_] | None, default=None
            The domains of the B-splines basis.

        Returns
        -------
        self: PSplinesDerivative
            Returns self.

        """
        X, y = check_X_y(X, y)
        dimension = X.shape[1]

        if dimension > 1:
            raise NotImplementedError("Not implemented for dimension > 1.")

        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights, X, dtype=X.dtype
            )

        # Estimate the model
        if domains is None:
            domains = (np.min(X), np.max(X))
        ps = PSplines(
            penalty=self.penalty,
            n_segments=self.n_segments,
            degree=self.degree,
            order_penalty=self.order_penalty,
        )
        ps.fit(X=X, y=y, sample_weights=sample_weights, domains=domains)

        # Estimate the derivative
        n_func = self.n_segments[0] + self.degree[0] - self.order_derivative
        basis = basis_bsplines(
            argvals=X.squeeze(),
            n_functions=n_func,
            degree=self.degree[0] - self.order_derivative,
            domain_min=domains[0],
            domain_max=domains[1],
        )
        beta_hat = (
            np.diff(ps.beta_hat_, n=self.order_derivative)
            / ((domains[1] - domains[0]) / self.n_segments)
            ** self.order_derivative
        )
        y_hat = basis.T @ beta_hat

        # Export results
        self.is_fitted_ = True
        self.dimension_ = dimension
        self.basis_ = basis
        self.domains_ = domains if isinstance(domains, list) else [domains]
        self.y_hat_ = y_hat
        self.beta_hat_ = beta_hat
        # self.diagnostics_ = {
        #     "hat_matrix": results.get("hat_matrix", None),
        #     "roughness": results.get("roughness", None),
        #     "residuals_std": results.get("residuals_std", None),
        # }
        return self

    def predict(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Predict the response variable values.

        The method predicts the response variable values for the given predictor
        variable values using the fitted P-splines model. If `X` is not
        provided, the method returns the fitted values.

        Parameters
        ----------
        X: npt.NDArray[np.float_]
            An array containing the predictor variable values.

        Returns
        -------
        npt.NDArray[np.float_]
            An array containing the estimated response variable values.

        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Build the B-splines basis
        new_X = [np.unique(column) for column in X.T]
        basis = [
            basis_bsplines(
                argvals=argvals,
                n_functions=n_segments + degree - self.order_derivative,
                degree=degree - self.order_derivative,
                domain_min=domain[0],
                domain_max=domain[1],
            )
            for argvals, n_segments, degree, domain in zip(
                new_X, self.n_segments, self.degree, self.domains_
            )
        ]

        if self.dimension_ == 1:
            y_pred = self.beta_hat_ @ basis[0]
        else:
            y_pred = rotated_h_transform(basis[0].T, self.beta_hat_)
            for idx in np.arange(1, len(basis)):
                y_pred = rotated_h_transform(basis[idx].T, y_pred)

        return y_pred
