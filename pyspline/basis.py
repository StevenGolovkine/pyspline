#!/usr/bin/env python
# -*-coding:utf8 -*
"""
Basis functions
---------------

"""

import numpy as np
import numpy.typing as npt

from math import gamma


def tpower(
    x: npt.NDArray[np.float_], knots: npt.NDArray[np.float_], p: int = 3
) -> npt.NDArray[np.float_]:
    """Compute truncated power functions.

    Parameters
    ----------
    x: npt.NDArray[np.float_], shape=(n, )
        An array on which the functions are calculated.
    knots: npt.NDArray[np.float_], shape=(m, )
        An array giving the truncation points.
    p: int, default=3
        Degree of the basis. The default gives cubic truncated power functions.

    Returns
    -------
    npt.NDArray[np.float_], shape=(n, m)
        An array containing the truncated power functions.

    """
    res = np.zeros((len(x), len(knots)))
    for idx, knot in enumerate(knots):
        res[:, idx] = np.power(x - knot, p) * (x >= knot)
    return res


def basis_bsplines(
    argvals: npt.NDArray[np.float_],
    n_functions: int = 10,
    degree: int = 3,
    domain_min: float | None = None,
    domain_max: float | None = None,
) -> npt.NDArray[np.float_]:
    """Define a B-splines basis of functions.

    Build a basis of :math:`n_functions` functions using B-splines basis on the
    interval defined by ``argvals``. We assume that the knots are regularly
    spaced. The number of knots is equal to ``n_functions - degree``.

    Parameters
    ----------
    argvals: npt.NDArray[np.float_]
        The values on which evaluated the B-splines.
    n_functions: int, default=10
        Number of considered B-splines.
    degree: int, default=3
        Degree of the B-splines. The default gives cubic splines.
    domain_min: float, default=None
        Minimum number for the argvals. If `None`, the value is set to
        `min(argvals)`.
    domain_max: float, default=None
        Maximum number for hte argvals. If `None`, the value is set to
        `max(argvals)`.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of a
        B-splines basis.

    Notes
    -----
    This function is adapted from the `bbase` function in the R package `JOPS`
    _[2]. It computes a proper B-splines basis function (see _[1], Section 8.1).

    Examples
    --------
    >>> basis_bsplines(argvals=np.arange(0, 1, 0.01), n_functions=10)

    References
    ----------
    .. [1] Eilers, P., Marx, B.D., (2021) Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X.,
        (2023) JOPS: Practical Smoothing with P-Splines.

    """
    # Set parameters
    if domain_min is None:
        domain_min = min(argvals)
    if domain_max is None:
        domain_max = max(argvals)

    # Compute the B-splines
    n_segments = n_functions - degree
    dx = (domain_max - domain_min) / n_segments
    knots = np.linspace(
        domain_min - degree * dx,
        domain_max + degree * dx,
        num=int(n_segments + 2 * degree) + 1,
        endpoint=True,
    )
    p_mat = tpower(argvals, knots, degree)
    d_mat = np.diff(np.eye(p_mat.shape[1]), n=degree + 1, axis=0) / (
        gamma(degree + 1) * np.power(dx, degree)
    )
    basis_mat = np.power(-1, degree + 1) * p_mat @ d_mat.T

    # Make B-splines exactly zero beyond their end knots
    sk = knots[np.arange(basis_mat.shape[1]) + degree + 1]
    mask = np.zeros((len(argvals), len(sk)))
    for idx, val in enumerate(argvals):
        mask[idx, :] = val < sk
    return (basis_mat * mask).T  # type: ignore
