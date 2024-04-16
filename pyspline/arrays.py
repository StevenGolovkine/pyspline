#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Array manipulation function
---------------------------

"""
import numpy as np
import numpy.typing as npt


def row_tensor(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.float64]:
    """
    Compute the row-wise tensor product of two 2D arrays.

    The row-wise tensor product of two 2D arrays `x` and `y` is a 2D array `z`
    such that each row of `z` is the Kronecker product of the corresponding row
    of `x` and `y`. If `y` is not provided, it defaults to `x`. Note that `x`
    and `y` must have the same number of rows.

    Parameters
    ----------
    x: npt.NDArray[np.float64]
        A 2D array of shape `(m, n)`.
    y: npt.NDArray[np.float64] | None, default=None
        A 2D array of shape `(m, q)`. If not provided, it defaults to `x`.

    Returns
    -------
    npt.NDArray[np.float64]
        A 2D array of shape `(m, n*q)` or `(m, n*n)` if `y` is not provided.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[5, 6, 7], [7, 8, 9]])
    >>> _row_tensor(x, y)
    array([
        [ 5.,  6.,  7., 10., 12., 14.],
        [21., 24., 27., 28., 32., 36.]
    ])
    >>> _row_tensor(x)
    array([
        [ 1.,  2.,  2.,  4.],
        [ 9., 12., 12., 16.]
    ])

    Notes
    -----
    This function is adapted from [1]_.

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized
        Linear Array Models with Applications to Multidimensional Smoothing.
        Journal of the Royal Statistical Society. Series B (Statistical
        Methodology) 68, pp.259--280.

    """
    if y is None:
        y = x
    if x.shape[0] != y.shape[0]:
        raise ValueError("`x` and `y` must have the same number of rows.")
    onex = np.ones((1, x.shape[1]))
    oney = np.ones((1, y.shape[1]))
    return np.kron(x, oney) * np.kron(onex, y)
