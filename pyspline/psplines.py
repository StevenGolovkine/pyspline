#!/usr/bin/env python
# -*-coding:utf8 -*
"""
P-Splines
---------

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
from .formatter import format_X_y
from .psplines_inner import fit_one_dimensional, fit_n_dimensional


class PSplines(BaseEstimator, RegressorMixin):  # type: ignore
    """P-Splines Smoothing.

    Parameters
    ----------
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
        penalty: Tuple[float] = (1.0,),
        *,
        n_segments: Tuple[int] = (10,),
        degree: Tuple[int] = (3,),
        order_penalty: int = 2,
    ):
        """Initialize PSplines object."""
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
    ) -> PSplines:
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
        self: PSplines
            Returns self.

        """
        X, y = check_X_y(X, y)
        dimension = X.shape[1]

        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights, X, dtype=X.dtype
            )

        if dimension == 1:
            if domains is None:
                domains = (np.min(X), np.max(X))
            basis = basis_bsplines(
                argvals=X.squeeze(),
                n_functions=self.n_segments[0] + self.degree[0],
                degree=self.degree[0],
                domain_min=domains[0],
                domain_max=domains[1],
            )
            results = fit_one_dimensional(
                data=y,
                basis=basis,
                sample_weights=sample_weights,
                penalty=self.penalty,
                order_penalty=self.order_penalty,
            )
        else:
            # Modify y in order to have the right shape to fit in the array algo
            X, y, sample_weights = format_X_y(X, y, sample_weights)
            if domains is None:
                domains = [(np.min(xx), np.max(xx)) for xx in X]

            basis = [
                basis_bsplines(
                    argvals=argvals,
                    n_functions=n_segments + degree,
                    degree=degree,
                    domain_min=domain[0],
                    domain_max=domain[1],
                )
                for argvals, n_segments, degree, domain in zip(
                    X, self.n_segments, self.degree, domains
                )
            ]
            results = fit_n_dimensional(
                data=y,
                basis_list=basis,
                sample_weights=sample_weights,
                penalties=self.penalty,
                order_penalty=self.order_penalty,
            )

        # Export results
        self.is_fitted_ = True
        self.dimension_ = dimension
        self.basis_ = basis
        self.domains_ = domains if isinstance(domains, list) else [domains]
        self.y_hat_ = results.get("y_hat", None)
        self.beta_hat_ = results.get("beta_hat", None)
        self.diagnostics_ = {
            "hat_matrix": results.get("hat_matrix", None),
            "eff_dimension": results.get("eff_dimension", None),
            "roughness": results.get("roughness", None),
            "residuals_std": results.get("residuals_std", None),
            "se_eta": results.get("se_eta", None),
            "inv_mat": results.get("inv_mat", None),
        }
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
                n_functions=n_segments + degree,
                degree=degree,
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

    def errors(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Estimate the standard errors of the fitted values.

        Parameters
        ----------
        X: npt.NDArray[np.float_]
            An array containing the predictor variable values.

        Returns
        -------
        npt.NDArray[np.float_]
            An array containing standard errors of the fitted values.

        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        if self.dimension_ > 1:
            raise NotImplementedError("Not implemented for dimension > 1.")

        # Build the B-splines basis
        new_X = [np.unique(column) for column in X.T]
        basis = [
            basis_bsplines(
                argvals=argvals,
                n_functions=n_segments + degree,
                degree=degree,
                domain_min=domain[0],
                domain_max=domain[1],
            )
            for argvals, n_segments, degree, domain in zip(
                new_X, self.n_segments, self.degree, self.domains_
            )
        ]

        temp = np.diag(basis[0].T @ self.diagnostics_["inv_mat"] @ basis[0])
        se_eta = np.sqrt(self.diagnostics_["residuals_std"] ** 2 * temp)
        return se_eta

    def derivative(self, X: npt.NDArray[np.float_], order_derivative: int = 1):
        """Estimate the derivative of the data.

        Parameters
        ----------
        X: npt.NDArray[np.float_]
            An array containing the predictor variable values.
        order_derivative: int, default=1
            Order of the derivative to compute.

        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        if self.dimension_ > 1:
            raise NotImplementedError("Not implemented for dimension > 1.")

        n_functions = self.n_segments[0] + self.degree[0] - order_derivative
        basis = basis_bsplines(
            argvals=X.squeeze(),
            n_functions=n_functions,
            degree=self.degree[0] - order_derivative,
            domain_min=self.domains_[0][0],
            domain_max=self.domains_[0][1],
        )
        beta_hat = (
            np.diff(self.beta_hat_, n=order_derivative)
            / ((self.domains_[0][1] - self.domains_[0][0]) / self.n_segments[0])
            ** order_derivative
        )
        return basis.T @ beta_hat
