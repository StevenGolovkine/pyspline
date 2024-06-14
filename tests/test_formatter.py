#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of formatter.py file."""

import numpy as np
import pytest

from pyspline.formatter import format_X_y


@pytest.fixture
def data():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    weights = np.ones_like(y)
    return {"x": x, "y": y, "weights": weights}


@pytest.fixture
def data_2d():
    x = np.array(
        [
            [0.0, -0.5],
            [0.0, 0.0],
            [0.0, 0.5],
            [0.5, -0.5],
            [1.0, 0.0],
            [1.0, 0.5],
        ]
    )
    y = np.array([1, 2, 3, 4, 5, 6])
    return {"x": x, "y": y}


###############################################################################
# Tests format_X_y
def test_format_X_y(data):
    X = data["x"].reshape(-1, 1)  # Consistent with PSplines input.
    new_X, new_y, weights = format_X_y(X, data["y"], data["weights"])

    expected_x = [data["x"]]
    expected_y = data["y"]
    expected_weights = data["weights"]

    np.testing.assert_array_almost_equal(new_X, expected_x)
    np.testing.assert_array_almost_equal(new_y, expected_y)
    np.testing.assert_array_almost_equal(weights, expected_weights)


def test_format_X_y_2d(data_2d):
    new_X, new_y, weights = format_X_y(data_2d["x"], data_2d["y"])

    expected_x = [np.array([0, 0.5, 1]), np.array([-0.5, 0, 0.5])]
    expected_y = np.array([[1, 2, 3], [4, 0, 0], [0, 5, 6]])
    expected_weights = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1]])

    np.testing.assert_array_almost_equal(new_X, expected_x)
    np.testing.assert_array_almost_equal(new_y, expected_y)
    np.testing.assert_array_almost_equal(weights, expected_weights)
