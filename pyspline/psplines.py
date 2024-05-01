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

from .basis import basis_bsplines


class PSplines(BaseEstimator, RegressorMixin):  # type: ignore
    """P-Splines Smoothing.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

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
        """Initializa PSplines object."""
        self.penalty = penalty
        self.n_segments = n_segments
        self.degree = degree
        self.order_penalty = order_penalty

    def fit(
        self,
        X: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_],
        sample_weight: npt.NDArray[np.float_] | None = None,
    ) -> PSplines:
        """Fit a P-splines model to the given data.

        The method fits a P-splines model to the given data using a B-splines
        basis and an optional weights matrix.

        Parameters
        ----------
        X: Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            A 1D or a list of 1D arrays of shape `(n1,), (n2,), ..., (nk,)`
            containing the predictor variable values.
        y: npt.NDArray[np.float64]
            An nD array of shape `(n1, n2, ..., nk)` containing the response
            variable values.
        sample_weights: npt.NDArray[np.float64] | None, default=None
            An N-dimensional array of shape `(n1, n2, ..., nk)` containing the
            weights for each observation. If not provided, all observations are
            assumed to have equal weight.

        Returns
        -------
        self: PSplines
            Returns self.

        """
        X, y = check_X_y(X, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype
            )

        dimension = X.shape[1]

        # Build the B-splines basis
        basis = [
            basis_bsplines(
                argvals=argvals, n_functions=n_segments + degree, degree=degree
            )
            for argvals, n_segments, degree in zip(
                X,
                self.n_segments,
                self.degree,
            )
        ]

        self.is_fitted_ = True
        self.dimension_ = dimension
        self.basis = basis
        # `fit` should always return `self`
        return self

    def predict(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Predict the response variable values.

        The method predicts the response variable values for the given predictor
        variable values using the fitted P-splines model. If `X` is not
        provided, the method returns the fitted values.

        Parameters
        ----------
        X: npt.NDArray[np.float64]]
            A 1D or a list of one-dimensional arrays of shape
            `(n1,), (n2,), ..., (nk,)` containing the predictor variable values.

        Returns
        -------
        npt.NDArray[np.float64]
            An nD array of shape `(n1, n2, ..., nk)` containing the predicted
            response variable values.

        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        return np.ones(X.shape[0], dtype=np.int64)
