#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of basis.py file."""

import numpy as np
import pytest

from pyspline.basis import tpower, basis_bsplines


@pytest.fixture
def data():
    x = np.linspace(0, 1, 11)
    knots = np.array([0.25, 0.5, 0.75])
    p = 1
    n_functions = 3
    return {"x": x, "knots": knots, "p": p, "n_functions": n_functions}


def test_tpower(data):
    expected_result = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.35, 0.1, 0.0],
            [0.45, 0.2, 0.0],
            [0.55, 0.3, 0.05],
            [0.65, 0.4, 0.15],
            [0.75, 0.5, 0.25],
        ]
    )
    result = tpower(data["x"], data["knots"], data["p"])
    np.testing.assert_almost_equal(result, expected_result)


def test_tpower_default(data):
    expected_result = np.array(
        [
            [0.00000e00, 0.00000e00, 0.00000e00],
            [0.00000e00, 0.00000e00, 0.00000e00],
            [0.00000e00, 0.00000e00, 0.00000e00],
            [1.25000e-04, 0.00000e00, 0.00000e00],
            [3.37500e-03, 0.00000e00, 0.00000e00],
            [1.56250e-02, 0.00000e00, 0.00000e00],
            [4.28750e-02, 1.00000e-03, 0.00000e00],
            [9.11250e-02, 8.00000e-03, 0.00000e00],
            [1.66375e-01, 2.70000e-02, 1.25000e-04],
            [2.74625e-01, 6.40000e-02, 3.37500e-03],
            [4.21875e-01, 1.25000e-01, 1.56250e-02],
        ]
    )
    result = tpower(data["x"], data["knots"])
    np.testing.assert_almost_equal(result, expected_result)


def test_basis_bsplines(data):
    expected_result = np.array(
        [
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
    )
    min = np.min(data["x"])
    max = np.max(data["x"])
    result = basis_bsplines(data["x"], data["n_functions"], data["p"], min, max)
    np.testing.assert_almost_equal(result, expected_result)


def test_basis_bsplines_default(data):
    expected_result = np.array(
        [
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
    )
    result = basis_bsplines(data["x"], data["n_functions"], data["p"])
    np.testing.assert_almost_equal(result, expected_result)
