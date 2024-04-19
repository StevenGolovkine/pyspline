#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of arrays.py file."""

import numpy as np
import pytest

from pyspline.arrays import row_tensor


@pytest.fixture
def data():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6, 7], [7, 8, 9]])
    return {"x": x, "y": y}


@pytest.fixture
def data_wrong():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2]])
    return {"x": x, "y": y}


def test_row_tensor_with_y(data):
    expected_result = np.array([
        [5.0, 6.0, 7.0, 10.0, 12.0, 14.0], [21.0, 24.0, 27.0, 28.0, 32.0, 36.0]
    ])
    result = row_tensor(data["x"], data["y"])
    np.testing.assert_array_equal(result, expected_result)


def test_row_tensor_without_y(data):
    expected_result = np.array([[1.0, 2.0, 2.0, 4.0], [9.0, 12.0, 12.0, 16.0]])
    result = row_tensor(data["x"])
    np.testing.assert_array_equal(result, expected_result)


def test_row_tensor_with_mismatched_shapes(data_wrong):
    with pytest.raises(ValueError):
        row_tensor(data_wrong["x"], data_wrong["y"])
