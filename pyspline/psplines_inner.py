#!/usr/bin/env python
# -*-coding:utf8 -*

"""
P-Splines inner functions
-------------------------

"""

import numpy as np
import numpy.typing as npt


def tensor_product_penalties(
    penalties: list[npt.NDArray[np.float_]],
) -> list[npt.NDArray[np.float_]] | npt.NDArray[np.float_]:
    """
    Compute the tensor product of a list of penalty matrices.

    The `tensor_product_penalties` function computes the tensor product of a
    list of penalty matrices. The resulting list contains the tensor product of
    each penalty matrix with itself and with the identity matrices of the same
    size as the other penalty matrices. If a penalty matrix is square, its
    tensor product with itself is symmetrized by taking the mean of the matrix
    and its transpose.

    Parameters
    ----------
    penalties: list[npt.NDArray[np.float_]]
        A list of penalty matrices.

    Returns
    -------
    list[npt.NDArray[np.float_]]
        A list of tensor product matrices.

    Notes
    -----
    This function is adapted from the function `tensor.prod.penalties` in [1]_.

    Examples
    --------
    >>> penalties = [
    ...     np.array([
    ...         [ 1., -1.,  0.],
    ...         [-1.,  2., -1.],
    ...         [ 0., -1.,  2.]
    ...     ]),
    ...     np.array([
    ...         [ 1., -1.],
    ...         [-1.,  2.]
    ...     ])
    ... ]
    >>> _tensor_product_penalties(penalties)
    [
        array([
           [ 1.,  0., -1., -0.,  0.,  0.],
           [ 0.,  1., -0., -1.,  0.,  0.],
           [-1., -0.,  2.,  0., -1., -0.],
           [-0., -1.,  0.,  2., -0., -1.],
           [ 0.,  0., -1., -0.,  2.,  0.],
           [ 0.,  0., -0., -1.,  0.,  2.]
        ]),
        array([
           [ 1., -1.,  0., -0.,  0., -0.],
           [-1.,  2., -0.,  0., -0.,  0.],
           [ 0., -0.,  1., -1.,  0., -0.],
           [-0.,  0., -1.,  2., -0.,  0.],
           [ 0., -0.,  0., -0.,  1., -1.],
           [-0.,  0., -0.,  0., -1.,  2.]
        ]
    )]

    References
    ----------
    .. [1] Wood, S. (2023). mgcv: Mixed GAM Computation Vehicle with Automatic
        Smoothness Estimation.

    """
    n_penalties = len(penalties)
    eyes = [np.eye(penalty.shape[1]) for penalty in penalties]

    if n_penalties == 1:
        return penalties[0]
    else:
        tensors_list = []
        for idx in range(n_penalties):
            left = penalties[0] if idx == 0 else eyes[0]
            for j in range(1, n_penalties):
                right = penalties[j] if idx == j else eyes[j]
                left = np.kron(left, right)
            # Make sure the matrix is symmetric
            if left.shape[0] == left.shape[1]:
                left = (left + left.T) / 2
            tensors_list.append(left)
        return tensors_list
