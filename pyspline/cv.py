#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.typing as npt
from typing import cast, Any
from sklearn.utils.validation import (
    check_X_y,
)
from functools import reduce

from .psplines import PSplines
from .basis import basis_bsplines

def cv(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    params: list[tuple[float]],
    n_segments: tuple[int] = (10,),
    degree: tuple[int] = (3,),
    order_penalty: int = 2,
    domains: list[tuple[float,float]] | tuple[float,float] | None = None,

) -> npt.NDArray[np.float64]:
    """
    Leave-one-out standard error prediction.

    Parameters
    ----------
    X: npt.NDArray[np.float64], shape=(n_obs, n_dimension)
        An array containing the predictor variable values.
    y: npt.NDArray[np.float64], shape=(n_obs,)
        An array containing the response variable values.
    params: list[tuple[np.float64]]
        The penalty parameters
    n_segments: Tuple[int], default=(10,)
        The number of evenly spaced segments.
    degree: Tuple[int], default=(3,)
        The number of the degree of the basis.
    order_penalty: int, default=2
        The number of the order of the difference penalty.
    domains: list[tuple[float, float]] | tuple[float] | None, default=None
        The domains of the B-splines basis.

    Returns
    -------
    npt.NDArray[np.float64]
        An array containing leave-one-out standard error prediction.

    """
    X, y = check_X_y(X, y)
    dimension = X.shape[1]
    nb_obs = len(y)
    cv = np.zeros(len(params), dtype=float)

    if dimension == 1:
        for idx, p in enumerate(params):
            ps = PSplines(n_segments=n_segments, degree=degree, penalty=p,
                    order_penalty=order_penalty)
            ps.fit(X, y, domains=domains)
            if ps.y_hat_ is None:
                raise ValueError("ps.y_hat_ cannot be None")
            if isinstance(ps.basis_, list):
                raise ValueError("Expected 1D basis, got multi-dimensional")

            diff = np.diff(np.eye(ps.basis_.T.shape[1]), order_penalty).T
            btb = ps.basis_ @ ps.basis_.T
            h_diag = np.sum(ps.basis_ * (np.linalg.solve(
                btb + np.eye(btb.shape[0]) * 1e-4 + p * diff.T @ diff,
                ps.basis_)), 0)
            cv[idx] = np.sqrt(np.sum(
                ((y - ps.y_hat_) / (1- h_diag)) **2)/nb_obs)
    else:
        for idx, p in enumerate(params):
            ps = PSplines(n_segments=n_segments, degree=degree, penalty=p,
                order_penalty=order_penalty)
            ps.fit(X, y, domains=domains)
            if ps.domains_ is None:
                raise ValueError("ps.domains_ cannot be None")

            y_pred = np.array([ps.predict(X[i,:].reshape(1,-1))[0][0]
                               for i in range(len(X))])
            basis = [
                basis_bsplines(
                    argvals=argvals,
                    n_functions=n_segments + degree,
                    degree=degree,
                    domain_min=float(domain[0]),
                    domain_max=float(domain[1]),
                    ).T
                    for argvals, n_segments, degree, domain in zip(
                        X.T, ps.n_segments, ps.degree, ps.domains_
                    )
                    ]

            # Penalty matrix
            total_penalty = penalties(basis, order_penalty, ps.penalty)

            # Kronecker product
            b_kron = kronecker_product(basis)
            h_diag = np.sum(b_kron.T * (np.linalg.solve(b_kron.T @ b_kron +
                                                total_penalty, b_kron.T)), 0)

            cv[idx] = np.sqrt(np.sum(((y - y_pred) / (1-h_diag)) **2)/nb_obs)
    return cv

def cv_derivative(
    X: npt.NDArray[np.float64],
    deriv: npt.NDArray[np.float64],
    params: list[tuple[float]],
    n_segments: tuple[int] = (10,),
    degree: tuple[int] = (3,),
    order_penalty: int = 2,
    domains: list[tuple[float,float]] | tuple[float,float] | None = None,

) -> npt.NDArray[np.float64]:
    """
    Leave-one-out standard error prediction.

    Parameters
    ----------
    X: npt.NDArray[np.float64], shape=(n_obs, n_dimension)
        An array containing the predictor variable values.
    deriv: npt.NDArray[np.float64], shape=(n_obs,)
        An array containing the estimated derivatives of the fitted values
    params: list[tuple[np.float64]]
        The penalty parameters
    n_segments: Tuple[int], default=(10,)
        The number of evenly spaced segments.
    degree: Tuple[int], default=(3,)
        The number of the degree of the basis.
    order_penalty: int, default=2
        The number of the order of the difference penalty.
    domains: list[tuple[float, float]] | tuple[float] | None, default=None
        The domains of the B-splines basis.

    Returns
    -------
    npt.NDArray[np.float64]
        An array containing leave-one-out standard error prediction.

    """
    X, deriv = check_X_y(X, deriv)
    dimension = X.shape[1]
    nb_obs = len(deriv)
    cv = np.zeros(len(params), dtype=float)

    if dimension == 1:
        if domains is None:
            domains = (float(np.min(X)), float(np.max(X)))
        elif not isinstance(domains, tuple):
            raise TypeError("For 1D, domains must be tuple[float, float]")

        for idx, p in enumerate(params):
            basis_one_dimensional = basis_bsplines(
                    argvals=X.squeeze(),
                    n_functions=n_segments[0] + degree[0],
                    degree=degree[0],
                    domain_min=float(domains[0]),
                    domain_max=float(domains[1]),
                    ).T

            if isinstance(basis_one_dimensional, list):
                raise ValueError("Expected 1D basis, got multi-dimensional")

            diff = np.diff(np.eye(basis_one_dimensional.shape[1]),
                           order_penalty).T
            btb = basis_one_dimensional.T @ basis_one_dimensional

            a = np.linalg.solve(btb + np.eye(btb.shape[0]) * 1e-4
                                + p * diff.T @ diff, basis_one_dimensional.T)
            h_diag = np.sum(basis_one_dimensional.T * a, 0)
            deriv_pred = basis_one_dimensional @ a @ deriv

            cv[idx] = np.sqrt(np.sum(
                ((deriv - deriv_pred) / (1- h_diag)) **2)/nb_obs)
    else:
        if domains is None:
            domains = [(float(np.min(xx)), float(np.max(xx)))
                           for xx in X]
        if not isinstance(domains, list):
            raise TypeError("For multi-dim, domains must be " \
                    "list[tuple[float,float]]")
        for idx, p in enumerate(params):
            basis = [
                basis_bsplines(
                    argvals=argvals,
                    n_functions=n_segments + degree,
                    degree=degree,
                    domain_min=float(domain[0]),
                    domain_max=float(domain[1]),
                    ).T
                    for argvals, n_segments, degree, domain in zip(
                        X.T, n_segments, degree, domains
                    )
                    ]

            # Penalty matrix
            total_penalty = penalties(basis, order_penalty, p)

            # Kronecker product
            b_kron = kronecker_product(basis)

            a = np.linalg.solve(b_kron.T @ b_kron + total_penalty, b_kron.T)
            h_diag = np.sum(b_kron.T * a, 0)
            deriv_pred = b_kron @ a @ deriv

            # CV
            cv[idx] = np.sqrt(np.sum(((
                deriv - deriv_pred) / (1-h_diag)) **2)/nb_obs)

    return cv

def gcv(
    p: float | tuple[np.float64],
    X: npt.NDArray[np.float64],
    deriv: npt.NDArray[np.float64],
    n_segments: tuple[int] = (10,),
    degree: tuple[int] = (3,),
    order_penalty: int = 2,
    domains: list[tuple[float,float]] | tuple[float,float] | None = None,
    ) -> np.float64:
    """
    Generalized cross-validation.

    Parameters
    ----------
    p: float | tuple[float]
        The penalty hyperparameter
    X: npt.NDArray[np.float64], shape=(n_obs, n_dimension)
        An array containing the predictor variable values.
    deriv: npt.NDArray[np.float64], shape=(n_obs,)
        An array containing the derivatives values.
    n_segments: Tuple[int], default=(10,)
        The number of evenly spaced segments.
    degree: Tuple[int], default=(3,)
        The number of the degree of the basis.
    order_penalty: int, default=2
        The number of the order of the difference penalty.
    domains: list[tuple[float, float]] | tuple[float] | None, default=None
        The domains of the B-splines basis.

    Returns
    -------
    np.float64
        The GCV value.

    """
    X, deriv = check_X_y(X, deriv)
    dimension = X.shape[1]
    nb_obs = len(deriv)


    if dimension == 1:
        if domains is None:
            domains = (float(np.min(X)), float(np.max(X)))
        elif not isinstance(domains, tuple):
            raise TypeError("For 1D, domains must be tuple[float, float]")

        basis_one_dimensional = basis_bsplines(
                argvals=X.squeeze(),
                n_functions=n_segments[0] + degree[0],
                degree=degree[0],
                domain_min=float(domains[0]),
                domain_max=float(domains[1]),
                ).T

        if isinstance(basis_one_dimensional, list):
            raise ValueError("Expected 1D basis, got multi-dimensional")

        diff = np.diff(np.eye(basis_one_dimensional.shape[1]),
                        order_penalty).T
        btb = basis_one_dimensional.T @ basis_one_dimensional
        A = (basis_one_dimensional @ np.linalg.solve(
            btb + np.eye(btb.shape[0]) * 1e-3 + p * diff.T @ diff,
            basis_one_dimensional.T))

    else:
        if domains is None:
            domains = [(float(np.min(xx)), float(np.max(xx)))
                            for xx in X]
        if not isinstance(domains, list):
            raise TypeError("For multi-dim, domains must be " \
                    "list[tuple[float,float]]")
        if not isinstance(p, tuple):
            raise TypeError("For multi-dim, parameters must be " \
                    "tuple")

        basis = [
                basis_bsplines(
                    argvals=argvals,
                    n_functions=n_segments + degree,
                    degree=degree,
                    domain_min=float(domain[0]),
                    domain_max=float(domain[1]),
                    ).T
                    for argvals, n_segments, degree, domain in zip(
                        X.T, n_segments, degree, domains
                    )
                    ]

        # Penalty matrix
        total_penalty = penalties(basis, order_penalty, tuple(p))
        small_penalty = penalties(basis, order_penalty,
                                  tuple(np.repeat(1e-03, len(basis))))

        # Kronecker product
        b_kron = kronecker_product(basis)
        btb = b_kron.T @ b_kron + small_penalty
        A = b_kron @ np.linalg.solve(btb + total_penalty, b_kron.T)

    A_formatted = cast(np.ndarray[tuple[int, int], Any], A)
    gcv = ((1/nb_obs) * np.linalg.norm(
            (np.identity(n=len(A_formatted)) - A_formatted)@deriv) /
            ((1/nb_obs) *np.matrix.trace(np.identity(n=len(A))-A_formatted))**2)

    return np.float64(gcv)

def risk(
        p: float | tuple[np.float64],
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_segments: tuple[int] = (10,),
        degree: tuple[int] = (3,),
        order_penalty: int = 2,
        domains: list[tuple[float,float]] | tuple[float,float] | None = None,
        order_derivative: int = 1,
        variance: int = 1,
        dim: tuple[int] = (0,)
) -> float:
    """
    Risk estimation.

    Parameters
    ----------
    p: float | tuple[np.float64]
        The penalty hyperparameter
    X: npt.NDArray[np.float64], shape=(n_obs, n_dimension)
        An array containing the predictor variable values.
    y: npt.NDArray[np.float64], shape=(n_obs,)
        An array containing the response variable values.
    n_segments: Tuple[int], default=(10,)
        The number of evenly spaced segments.
    degree: Tuple[int], default=(3,)
        The number of the degree of the basis.
    order_penalty: int, default=2
        The number of the order of the difference penalty.
    domains: list[tuple[float, float]] | tuple[float] | None, default=None
        The domains of the B-splines basis.
    order_derivative: int, default=1
        The order of the derivative to compute.
    variance: int, default=1
        The noise variance, sigma^2
    dim: tuple[int], default(0,)
        The dimension along which to compute the derivative

    Returns
    -------
    float
        A float containing the risk estimation.

    """
    X, y = check_X_y(X, y)
    dimension = X.shape[1]

    if dimension == 1:
        if domains is None:
            domains = (float(np.min(X)), float(np.max(X)))
        elif not isinstance(domains, tuple):
            raise TypeError("For 1D, domains must be tuple[float, float]")
        basis_one_dimensional = basis_bsplines(
                argvals=X.squeeze(),
                n_functions=n_segments[0] + degree[0],
                degree=degree[0],
                domain_min=float(domains[0]),
                domain_max=float(domains[1]),
                ).T

        if isinstance(basis_one_dimensional, list):
            raise ValueError("Expected 1D basis, got multi-dimensional")
        btb = (basis_one_dimensional.T @ basis_one_dimensional
                + np.eye(basis_one_dimensional.T.shape[0]) * 1e-3 )
        diff = np.diff(np.eye(basis_one_dimensional.shape[1]),
                                order_penalty).T
        H = np.linalg.solve(btb + p * diff.T @ diff, btb)
        alpha =  np.linalg.solve(btb , basis_one_dimensional.T) @ y

        basis_one_dimensional_deriv = basis_bsplines(
                            argvals=X.squeeze(),
                            n_functions = (n_segments[0] + degree[0]
                                    - order_derivative),
                            degree=degree[0],
                            domain_min=float(domains[0]),
                            domain_max=float(domains[1]),
                            ).T
        diff_deriv = np.diff(np.eye(basis_one_dimensional.shape[1]),
                                                        order_derivative).T
        D_r = (basis_one_dimensional_deriv @ diff_deriv
                /((domains[1]- domains[0])
                    / n_segments[0])**order_derivative)

    else:
        if domains is None:
            domains = [(float(np.min(xx)), float(np.max(xx)))
                            for xx in X]
        if not isinstance(domains, list):
            raise TypeError("For multi-dim, domains must be " \
                    "list[tuple[float,float]]")
        if not isinstance(p, tuple):
            raise TypeError("For multi-dim, p must be " \
                    "tuple[float]")
        basis = [
            basis_bsplines(
                argvals=argvals,
                n_functions=n_segments + degree,
                degree=degree,
                domain_min=float(domain[0]),
                domain_max=float(domain[1])
            ).T
            for argvals, n_segments, degree, domain in zip(
                    X.T, n_segments, degree, domains
            )
        ]

        # Penalty matrix
        total_penalty = penalties(basis, order_penalty, p)
        small_penalty = penalties(basis, order_penalty,
                                    tuple(np.repeat(1e-03, len(basis))))

        # Kronecker product
        b_kron = kronecker_product(basis)

        btb = b_kron.T @ b_kron + small_penalty
        H = np.linalg.solve(btb + total_penalty, btb)
        alpha =  np.linalg.solve(btb, b_kron.T) @ y

        basis_deriv: list[npt.NDArray[np.float64]] = []
        for i, (argvals, n_seg, deg, domain) in enumerate(zip(X.T,
                            n_segments, degree, domains)):
            if i in dim:
                deg = deg - order_derivative
            b = basis_bsplines(
                argvals=argvals,
                n_functions=n_seg + deg,
                degree=deg,
                domain_min=domain[0],
                domain_max=domain[1],
            ).T
            basis_deriv.append(b)

        b_kron_deriv = kronecker_product(basis_deriv)

        h = 1.0
        for d in dim:
            h *= ((domains[d][1] - domains[d][0]) /
                    n_segments[d])** int(order_derivative)

        diff_list = []
        for i, b in enumerate(basis):
            if i in dim:
                diff_list.append(np.diff(np.eye(b.shape[1]),
                                            n=order_derivative).T)
            else:
                diff_list.append(np.eye(b.shape[1]))
        diff_deriv = reduce(np.kron, diff_list)
        D_r = (b_kron_deriv @ diff_deriv/h)

    result = (variance * np.trace(D_r @ np.linalg.solve(btb, D_r.T))
        + 2 * variance * np.trace(
        D_r.T @ D_r @ (H - np.identity(n=len(H))) @ np.linalg.inv(btb))
        + np.linalg.norm(D_r @ (H - np.identity(n=len(H))) @ alpha)**2)
    return float(result)

def kronecker_product(
    basis: list[npt.NDArray[np.float64]],
)->npt.NDArray[np.float64]:
    """
    Computes the Kronecker product.

    Parameters
    ----------
    basis: list[npt.NDArray[np.float64]]
        The P-splines basis.

    Returns
    -------
    npt.NDArray[np.float64]
        An array containing the kronecker product.

    """
    shape_b_kron = 1
    nb_obs = basis[0].shape[0]
    for b in basis:
        shape_b_kron *= b.shape[1]
    b_kron = np.zeros((nb_obs, shape_b_kron))
    for i in range(nb_obs):
        temp = basis[0][i,:]
        for idx_b in range(1, len(basis)):
            temp_cast = cast(np.ndarray[tuple[int, int], Any], temp)
            temp_cast = np.kron(basis[idx_b][i,:], temp_cast)
        b_kron[i,:] = temp_cast
    return b_kron

def penalties(
    basis: list[npt.NDArray[np.float64]],
    order_penalty: int = 2,
    lambda_param: tuple[float,...] = (1,1)
)->npt.NDArray[np.float64]:
    """
    Computes the penaltes.

    Parameters
    ----------
    basis: list[npt.NDArray[np.float64]]
        The P-splines basis
    order_penalty: int, default=2
        The number of the order of the difference penalty.
    lambda_param: tuple[float,...], default=(1,1)
        The penalty hyperparameter

    Returns
    -------
    npt.NDArray[np.float64]
        An array containing the penalties.

    """
    pen_mat = []
    for i in range(len(basis)):
        matrices = []
        for j, b in enumerate(basis):
            if j == i:
                matrices.append(np.diff(np.eye(b.shape[1]),
                                        order_penalty).T)
            else:
                matrices.append(np.eye(b.shape[1]))
        temp = reduce(np.kron, matrices)
        pen_mat.append(temp.T @ temp)
    total_penalty = np.zeros_like(pen_mat[0], dtype=np.float64)
    for i in range(len(basis)):
        total_penalty += lambda_param[i] * pen_mat[i]
    return total_penalty