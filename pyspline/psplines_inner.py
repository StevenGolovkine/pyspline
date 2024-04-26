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


def fit_one_dimensional(
    data: npt.NDArray[np.float_],
    basis: npt.NDArray[np.float_],
    sample_weights: npt.NDArray[np.float_] | None = None,
    penalty: float = 1.0,
    order_penalty: int = 2,
) -> dict[str, npt.NDArray[np.float_]]:
    """
    Fit a one-dimensional P-splines model to the given data.

    The function fits a one-dimensional P-splines model to the given data using
    a basis matrix and an optional weight matrix. The function returns a
    dictionary containing the fitted values, the estimated coefficients, and the
    hat matrix.

    Parameters
    ----------
    data: npt.NDArray[np.float64]
        A one-dimensional array of shape `(n_obs,)` containing the response
        variable values.
    basis: npt.NDArray[np.float64]
        A two-dimensional array of shape `(n_basis, n_obs)` containing the basis
        matrix.
    sample_weights: npt.NDArray[np.float64] | None, default=None
        A one-dimensional array of shape `(n_obs,)` containing the weights for
        each observation. If not provided, all observations are assumed to have
        equal weight.
    penalty: float, default=1.0
        The penalty parameter for the P-splines model.
    order_penalty: int, default=2
        The order of the penalty difference matrix.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        A dictionary containing the following keys:
        - `y_hat`: A one-dimensional array of shape `(n_obs,)` containing the
        fitted values.
        - `beta_hat`: A one-dimensional array of shape `(n_basis,)` containing
        the estimated coefficients.
        - `hat_matrix`: A one-dimensional array of shape `(n_obs,)` containing
        the diagonal of the hat matrix.

    Notes
    -----
    The implementation of adapted from [2]_. See [1]_ for more details.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> basis = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]])
    >>> _fit_one_dimensional(data, basis)
    {
        'y_hat': array(
            [1.25143869, 1.95484728, 2.83532537, 3.89287295, 5.12749004]
        ),
        'beta_hat': array([0.7250996 , 0.43780434, 0.08853475]),
        'hat_matrix': array(
            [0.3756529, 0.3549800, 0.2669322, 0.2788401, 0.7545816]
        )
    }

    References
    ----------
    .. [1] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X.
        (2023). JOPS: Practical Smoothing with P-Splines.

    """
    # Get parameters.
    n_basis, n_obs = basis.shape

    # Construct the penalty.
    pen_mat = np.diff(np.eye(n_basis), n=order_penalty, axis=0)

    # Build the different part of the model.
    if sample_weights is None:
        sample_weights = np.ones(n_obs)
    weight_mat = np.diag(sample_weights)

    bwb_mat = basis @ weight_mat @ basis.T
    pen_mat = penalty * pen_mat.T @ pen_mat
    bwy_mat = basis @ weight_mat @ data

    # Fit the model
    inv_mat = np.linalg.pinv(bwb_mat + pen_mat)
    beta_hat = inv_mat @ bwy_mat
    y_hat = basis.T @ beta_hat

    # Compute the hat matrix
    hat_matrix = np.diag(basis.T @ inv_mat @ basis @ weight_mat)

    return {"y_hat": y_hat, "beta_hat": beta_hat, "hat_matrix": hat_matrix}
