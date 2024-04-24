#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Array manipulation function
---------------------------

"""

import numpy as np
import numpy.typing as npt


def row_tensor(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_] | None = None
) -> npt.NDArray[np.float_]:
    """
    Compute the row-wise tensor product of two 2D arrays.

    The row-wise tensor product of two 2D arrays `x` and `y` is a 2D array `z`
    such that each row of `z` is the Kronecker product of the corresponding row
    of `x` and `y`. If `y` is not provided, it defaults to `x`. Note that `x`
    and `y` must have the same number of rows.

    Parameters
    ----------
    x: npt.NDArray[np.float_]
        A 2D array of shape `(m, n)`.
    y: npt.NDArray[np.float_] | None, default=None
        A 2D array of shape `(m, q)`. If not provided, it defaults to `x`.

    Returns
    -------
    npt.NDArray[np.float_]
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
    >>> row_tensor(x)
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


def h_transform(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Compute the H-transform of a nD array `y` with respect to a 2D array `x`.

    The H-transform of a nD array `y` with respect to a 2D array `x` is a nD
    array `z`. The H-transform generalizes the pre-multiplication of vectors and
    matrices by a matrix.

    Parameters
    ----------
    x: npt.NDArray[np.float_]
        A 2D array of shape `(n, m)`.
    y: npt.NDArray[np.float_]
        A nD array of shape `(m, n1, n2, ..., nk)`.

    Returns
    -------
    npt.NDArray[np.float_]
        A nD array of shape `(n, n1, n2, ..., nk)`.

    Notes
    -----
    This function is adapted from [1]_.

    Examples
    --------
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> h_transform(x, y)
    array([[22, 28]])

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized
        Linear Array Models with Applications to Multidimensional Smoothing.
        Journal of the Royal Statistical Society. Series B (Statistical
        Methodology) 68, pp.259--280.

    """
    if x.shape[1] != y.shape[0]:
        raise ValueError(
            "The second dimension of `x` must be equal to the first dimension",
            " of `y`.",
        )
    y_dim = y.shape
    y_reshape = y.reshape(y_dim[0], np.prod(y_dim[1:]))
    xy_product = x @ y_reshape
    return xy_product.reshape((xy_product.shape[0], *y_dim[1:]))


def rotate(x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Rotate the axes of a multi-dimensional array to the right.

    The rotation of a nD array moves the first axis of a multi-dimensional array
    to the last position. This is equivalent to rotating the axes of the array
    to the right. It generalizes the transpose operation to nD arrays.

    Parameters
    ----------
    x: npt.NDArray[np.float_]
        A multi-dimensional array of shape `(n1, n2, ..., nk)`.

    Returns
    -------
    npt.NDArray[np.float_]
        A multi-dimensional array of shape `(n2, ..., nk, n1)`.

    Notes
    -----
    This function is adapted from [1]_.

    Examples
    --------
    >>> x = np.array([[[1, 2], [3, 4], [5, 6]], [[5, 6], [7, 8], [9, 0]]])
    >>> rotate(x)
    array([
        [[1, 5],[2, 6]],
        [[3, 7],[4, 8]],
        [[5, 9],[6, 0]]
    ])

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized
        Linear Array Models with Applications to Multidimensional Smoothing.
        Journal of the Royal Statistical Society. Series B (Statistical
        Methodology) 68, pp.259--280.

    """
    return np.moveaxis(x, 0, -1)


def rotated_h_transform(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Compute the rotated H-transform of a nD array with respect to a 2D array.

    The rotated H-transform of a nD array `y` with respect to a 2D array `x` is
    a nD array `z`. This function performed a H-transform of `x` and `y`, and
    then rotate the results.

    Parameters
    ----------
    x: npt.NDArray[np.float64]
        A 2D array of shape `(n, m)`.
    y: npt.NDArray[np.float64]
        A nD array of shape `(m, n1, n2, ..., nk)`.

    Notes
    -----
    This function is adapted from [1]_.

    Returns
    -------
    npt.NDArray[np.float64]
        A nD array of shape `(n1, n2, ..., nk, m)`.

    Examples
    --------
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> rotated_h_transform(x, y)
    array([
        [22],
        [28]
    ])

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized
        Linear Array Models with Applications to Multidimensional Smoothing.
        Journal of the Royal Statistical Society. Series B (Statistical
        Methodology) 68, pp.259--280.

    """
    return rotate(h_transform(x, y))


def create_permutation(p: int, k: int) -> npt.NDArray[np.float_]:
    """
    Create a permutation array for a given number of factors and levels.

    This function creates a permutation array for a given number of factors `p`
    and levels `k`. The resulting array is a 1D array of shape `(k*p,)` that
    contains the indices of all possible combinations of `p` factors with `k`
    levels each.

    Parameters
    ----------
    p: int
        The number of factors.
    k: int
        The number of levels.

    Returns
    -------
    npt.NDArray[np.float_]
        A 1D array of shape `(k*p,)` that contains the indices of all possible
        combinations of `p` factors with `k` levels each.

    Examples
    --------
    >>> np.tile(np.arange(3), 2)
    array([0, 1, 2, 0, 1, 2])
    >>> _create_permutation(3, 2)
    array([0, 3, 1, 4, 2, 5])
    >>> np.repeat(np.arange(3), 2)
    array([0, 0, 1, 1, 2, 2])
    >>> _create_permutation(2, 3)
    array([0, 2, 4, 1, 3, 5])

    """
    a = np.arange(0, k)
    b = np.arange(0, p)
    m = np.add.outer(a * p, b)
    return m.flatten("F")
