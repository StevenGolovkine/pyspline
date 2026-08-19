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

from numpy.typing import NDArray

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
    """
    P-Splines Smoothing.

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

        self.basis_: NDArray[np.float64] | list[NDArray[np.float64]
                                                ] = np.empty((0, 0))

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weights: npt.NDArray[np.float64] | None = None,
        domains: list[tuple[float, float]] | tuple[float, float] | None = None,
    ) -> PSplines:
        """
        Fit a P-splines model to the given data.

        The method fits a P-splines model to the given data using a B-splines
        basis and an optional weights matrix.

        Parameters
        ----------
        X: npt.NDArray[np.float64], shape=(n_obs, n_dimension)
            An array containing the predictor variable values.
        y: npt.NDArray[np.float64], shape=(n_obs,)
            An array containing the response variable values.
        sample_weights: npt.NDArray[np.float64] | None, default=None
            An array of shape `(n_obs,)` containing the weights for each
            observation. If not provided, all observations are assumed to have
            equal weight.
        domains: list[tuple[float, float]] | tuple[float,
            float] | None, default=None
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
                domains = (float(np.min(X)), float(np.max(X)))
            elif not isinstance(domains, tuple):
                raise TypeError("For 1D, domains must be tuple[float, float]")

            basis_one_dimensional = basis_bsplines(
                argvals=X.squeeze(),
                n_functions=self.n_segments[0] + self.degree[0],
                degree=self.degree[0],
                domain_min=domains[0],
                domain_max=domains[1],
            )
            results = fit_one_dimensional(
                data=y,
                basis=basis_one_dimensional,
                sample_weights=sample_weights,
                penalty=self.penalty[0],
                order_penalty=self.order_penalty,
            )
            self.basis_ = basis_one_dimensional
        else:
            # Modify y in order to have the right shape to fit in the array algo
            new_X, y, sample_weights = format_X_y(X, y, sample_weights)
            if domains is None:
                domains = [(float(np.min(xx)), float(np.max(xx)))
                           for xx in new_X]

            if not isinstance(domains, list):
                raise TypeError("For multi-dim, domains must be " \
                    "list[tuple[float,float]]")

            basis_n_dimensional = [
                basis_bsplines(
                    argvals=argvals,
                    n_functions=n_segments + degree,
                    degree=degree,
                    domain_min=domain[0],
                    domain_max=domain[1],
                )
                for argvals, n_segments, degree, domain in zip(
                    new_X, self.n_segments, self.degree, domains
                )
            ]
            results = fit_n_dimensional(
                data=y,
                basis_list=list(basis_n_dimensional),
                sample_weights=sample_weights,
                penalties=self.penalty,
                order_penalty=self.order_penalty,
            )
            self.basis_ = basis_n_dimensional

        # Export results
        self.is_fitted_ = True
        self.dimension_ = dimension
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

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predict the response variable values.

        The method predicts the response variable values for the given predictor
        variable values using the fitted P-splines model. If `X` is not
        provided, the method returns the fitted values.

        Parameters
        ----------
        X: npt.NDArray[np.float64]
            An array containing the predictor variable values.

        Returns
        -------
        npt.NDArray[np.float64]
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

        if self.beta_hat_ is None:
            raise ValueError("self.beta_hat_ cannot be None")
        if self.dimension_ == 1:
            y_pred = self.beta_hat_ @ basis[0]
        else:
            y_pred = rotated_h_transform(basis[0].T, self.beta_hat_)
            for idx in np.arange(1, len(basis)):
                y_pred = rotated_h_transform(basis[idx].T, y_pred)
        return y_pred

    def errors(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Estimate the standard errors of the fitted values.

        Parameters
        ----------
        X: npt.NDArray[np.float64]
            An array containing the predictor variable values.

        Returns
        -------
        npt.NDArray[np.float64]
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

        if self.diagnostics_["inv_mat"] is None:
            raise ValueError("self.diagnostics_['inv_mat'] cannot be None")
        temp = np.diag(basis[0].T @ self.diagnostics_["inv_mat"] @ basis[0])
        if self.diagnostics_['residuals_std'] is None:
            raise ValueError("self.diagnostics_['residuals_std']" \
                             "cannot be None")
        se_eta = np.sqrt(self.diagnostics_["residuals_std"] ** 2 * temp)
        if not isinstance(se_eta, np.ndarray):
            raise TypeError("se_eta must be a np.ndarray")
        if not np.issubdtype(se_eta.dtype, np.floating):
            raise TypeError("Array elements must be floats")
        return se_eta

    def derivative(
        self,
        X: npt.NDArray[np.float64],
        order_derivative: int = 1,
        dim: Tuple[int] = (0,)
    ) -> npt.NDArray[np.float64]:
        """
        Estimate the derivative of the data.

        Parameters
        ----------
        X: npt.NDArray[np.float64]
            An array containing the predictor variable values.
        order_derivative: int, default=1
            Order of the derivative to compute.
        dim: Tuple[int], default=(0,)
            dimension along which to compute the derivative

        Returns
        -------
        npt.NDArray[np.float64]
            An array containing the derivatives

        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        if self.beta_hat_ is None:
            raise ValueError("self.beta_hat_ cannot be None")

        if self.dimension_ == 1:
            n_functions = (self.n_segments[0] + self.degree[0]
                           - order_derivative)
            b = basis_bsplines(
                argvals=X.squeeze(),
                n_functions=n_functions,
                degree=self.degree[0] - order_derivative,
                domain_min=self.domains_[0][0],
                domain_max=self.domains_[0][1],
            )
            beta_hat = (
                np.diff(self.beta_hat_, n=order_derivative)
                / ((self.domains_[0][1] - self.domains_[0][0])
                   / self.n_segments[0])
                ** order_derivative
            )
            derivative = b.T @ beta_hat

        else:
            # Build the B-splines basis
            basis_list: list[npt.NDArray[np.float64]] = []
            for i, (argvals, n_segments, deg, domain) in enumerate(zip(X.T,
                                self.n_segments, self.degree, self.domains_)):
                if i in dim:
                    deg = deg - order_derivative
                b = basis_bsplines(
                    argvals=argvals,
                    n_functions=n_segments + deg,
                    degree=deg,
                    domain_min=domain[0],
                    domain_max=domain[1],
                )
                basis_list.append(b)

            # Beta hat
            diff = self.beta_hat_
            h = 1.0
            for d in dim:
                diff = np.diff(diff, n=int(order_derivative), axis=int(d))
                h *= ((self.domains_[d][1] - self.domains_[d][0]) /
                      self.n_segments[d])** int(order_derivative)
            beta_hat = diff/h

            # Derivative
            n_dims = beta_hat.ndim
            basis_indices = [chr(ord('a') + d) for d in range(n_dims)]
            einsum_str = ''.join(basis_indices) + ',' + ','.join(f"{i}z"
                                                for i in basis_indices) + '->z'
            derivative = np.einsum(einsum_str, beta_hat, *basis_list)

        return derivative
