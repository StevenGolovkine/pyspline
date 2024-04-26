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


class PSplines(BaseEstimator, RegressorMixin):  # type: ignore
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

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
        penalty: float = 1.0,
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
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype
            )

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        return np.ones(X.shape[0], dtype=np.int64)
