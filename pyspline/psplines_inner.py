#!/usr/bin/env python
# -*-coding:utf8 -*

"""
P-Splines inner functions
-------------------------

"""

import numpy as np
import numpy.typing as npt

from .arrays import create_permutation, rotated_h_transform, row_tensor


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
    data: npt.NDArray[np.float_]
        A one-dimensional array of shape `(n_obs,)` containing the response
        variable values.
    basis: npt.NDArray[np.float_]
        A two-dimensional array of shape `(n_basis, n_obs)` containing the basis
        matrix.
    sample_weights: npt.NDArray[np.float_] | None, default=None
        A one-dimensional array of shape `(n_obs,)` containing the weights for
        each observation. If not provided, all observations are assumed to have
        equal weight.
    penalty: float, default=1.0
        The penalty parameter for the P-splines model.
    order_penalty: int, default=2
        The order of the penalty difference matrix.

    Returns
    -------
    dict[str, npt.NDArray[np.float_]]
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
    diff_mat = np.diff(np.eye(n_basis), n=order_penalty, axis=0)

    # Build the different part of the model.
    if sample_weights is None:
        sample_weights = np.ones(n_obs)
    weight_mat = np.diag(sample_weights)

    bwb_mat = basis @ weight_mat @ basis.T
    pen_mat = penalty * diff_mat.T @ diff_mat
    bwy_mat = basis @ weight_mat @ data

    # Fit the model
    inv_mat = np.linalg.pinv(bwb_mat + pen_mat)
    beta_hat = inv_mat @ bwy_mat
    y_hat = basis.T @ beta_hat

    # Compute the hat matrix and effective dimension
    hat_matrix = np.diag(basis.T @ inv_mat @ basis @ weight_mat)
    eff_dimension = np.sum(hat_matrix)
    # Compute the roughness
    roughness = beta_hat @ diff_mat.T @ diff_mat @ beta_hat
    roughness = np.sqrt(roughness / (n_basis - order_penalty))
    # Compute standard deviation of the residuals
    residuals_std = np.sqrt(
        np.sum((data - y_hat) ** 2) / (n_obs - eff_dimension)
    )
    # Compute standard errors on the grid
    se_eta = np.sqrt(residuals_std**2 * hat_matrix)
    return {
        "y_hat": y_hat,
        "beta_hat": beta_hat,
        "hat_matrix": hat_matrix,
        "eff_dimension": eff_dimension,
        "roughness": roughness,
        "residuals_std": residuals_std,
        "se_eta": se_eta,
        "inv_mat": inv_mat,
    }


def fit_n_dimensional(
    data: npt.NDArray[np.float_],
    basis_list: list[npt.NDArray[np.float_]],
    sample_weights: npt.NDArray[np.float_] | None = None,
    penalties: tuple[float, ...] | None = None,
    order_penalty: int = 2,
) -> dict[str, npt.NDArray[np.float_]]:
    """
    Fit an nD P-splines model to the given data.

    The function fits an nD P-splines model to the given data using a list of
    basis matrices and an optional weights matrix. The function returns a
    dictionary containing the fitted values, the estimated coefficients, and the
    hat matrix.

    Parameters
    ----------
    data: npt.NDArray[np.float_]
        An nD array of shape `(n1, n2, ..., nk)` containing the response
        variable values.
    basis_list: list[npt.NDArray[np.float_]]
        A list of two-dimensional arrays of shape
        `(m1, n1), (m2, n2), ..., (mk, nk)` containing the basis matrices for
        each dimension.
    sample_weights: npt.NDArray[np.float_] | None, default=None
        An nD array of shape `(n1, n2, ..., nk)` containing the weights for each
        observation. If not provided, all observations are assumed to have equal
        weight.
    penalties: tuple[float, ...] | None, default=None
        A tuple of penalty parameters for each dimension. If not provided, the
        penalty is assumed to be the same for each dimension and equal to 1.
    order_penalty: int, default=2
        The order of the penalty difference matrix.

    Returns
    -------
    dict[str, npt.NDArray[np.float_]]
        A dictionary containing the following keys:
        - `y_hat`: An nD array of shape `(n1, n2, ..., nk)` containing the
        fitted values.
        - `beta_hat`: An nD array of shape `(m1, m2, ..., mk)` containing the
        estimated coefficients.
        - `hat_matrix`: A nD array of shape `(n1, n2, ..., nk)` containing the
        hat matrix.

    Notes
    -----
    The implementation of adapted from [2]_. See [1]_ for more details.

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> basis_list = [np.array([[1, 1], [1, 2]]), np.array([[1, 1], [2, 3]])]
    >>> _fit_n_dimensional(data, basis_list)
    {
        'y_hat': array([
            [1., 2.],
            [3., 4.]
        ]),
        'beta_hat': array([
            [-3.,  1.],
            [ 2.,  0.]
        ]),
        'hat_matrix': array([
            [1., 1.],
            [1., 1.]
        ])
    }

    References
    ----------
    .. [1] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X.
        (2023). JOPS: Practical Smoothing with P-Splines.

    """
    if sample_weights is None:
        sample_weights = np.ones_like(data)
    if penalties is None:
        penalties = len(data.shape) * (1,)

    n_basis = tuple(basis.shape[0] for basis in basis_list)
    tensor_list = [row_tensor(basis.T) for basis in basis_list]

    bwb_mat = rotated_h_transform(tensor_list[0].T, sample_weights)
    for idx in np.arange(1, len(tensor_list)):
        bwb_mat = rotated_h_transform(tensor_list[idx].T, bwb_mat)
    bwb_mat = (
        bwb_mat.reshape(np.repeat(n_basis, 2))
        .transpose(create_permutation(2, len(n_basis)))
        .reshape((np.prod(n_basis), np.prod(n_basis)))
    )

    # Penalty
    eyes_mats = [np.eye(n) for n in n_basis]
    diff_mats = [
        np.diff(eyes_mat, n=order_penalty, axis=0) for eyes_mat in eyes_mats
    ]
    prod_diff_mats = [diff_mat.T @ diff_mat for diff_mat in diff_mats]
    pen_mats = tensor_product_penalties(prod_diff_mats)

    penalty_mat = np.sum(
        [penalty * pen_mat for (penalty, pen_mat) in zip(penalties, pen_mats)],
        axis=0,
    )

    # Last part of the equation
    bwy_mat = rotated_h_transform(basis_list[0], data * sample_weights)
    for idx in np.arange(1, len(basis_list)):
        bwy_mat = rotated_h_transform(basis_list[idx], bwy_mat)
    bwy_mat = bwy_mat.reshape(np.prod(n_basis))

    # Fit
    fit = np.linalg.lstsq(bwb_mat + penalty_mat, bwy_mat, rcond=None)
    y_hat = rotated_h_transform(basis_list[0].T, fit[0].reshape(n_basis))
    for idx in np.arange(1, len(basis_list)):
        y_hat = rotated_h_transform(basis_list[idx].T, y_hat)

    # Compute the H matrix
    rot_hat_mat = np.linalg.pinv(bwb_mat + penalty_mat)
    rot_hat_mat = (
        rot_hat_mat.reshape(np.repeat(n_basis, 2))
        .transpose(create_permutation(2, len(n_basis)))
        .reshape(tuple(n**2 for n in n_basis))
    )

    hat_matrix = rotated_h_transform(tensor_list[0], rot_hat_mat)
    for idx in np.arange(1, len(tensor_list)):
        hat_matrix = rotated_h_transform(tensor_list[idx], hat_matrix)
    hat_matrix = sample_weights * hat_matrix

    return {
        "y_hat": y_hat,
        "beta_hat": fit[0].reshape(n_basis),
        "hat_matrix": hat_matrix,
    }
