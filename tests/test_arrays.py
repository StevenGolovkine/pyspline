#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of arrays.py file."""

import numpy as np
import pytest

from pyspline.arrays import (
    row_tensor,
    h_transform,
    rotate,
    rotated_h_transform,
    create_permutation,
)


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


###############################################################################
# Tests row_tensor
def test_row_tensor_with_y(data):
    expected_result = np.array(
        [
            [5.0, 6.0, 7.0, 10.0, 12.0, 14.0],
            [21.0, 24.0, 27.0, 28.0, 32.0, 36.0],
        ]
    )
    result = row_tensor(data["x"], data["y"])
    np.testing.assert_array_equal(result, expected_result)


def test_row_tensor_without_y(data):
    expected_result = np.array([[1.0, 2.0, 2.0, 4.0], [9.0, 12.0, 12.0, 16.0]])
    result = row_tensor(data["x"])
    np.testing.assert_array_equal(result, expected_result)


def test_row_tensor_with_mismatched_shapes(data_wrong):
    with pytest.raises(ValueError):
        row_tensor(data_wrong["x"], data_wrong["y"])


###############################################################################
# Tests h_transform
def test_h_transform(data):
    expected_result = np.array([[19, 22, 25], [43, 50, 57]])
    result = h_transform(data["x"], data["y"])
    np.testing.assert_array_equal(result, expected_result)


def test_h_transform_with_mismatched_shapes(data_wrong):
    with pytest.raises(ValueError):
        h_transform(data_wrong["x"], data_wrong["y"])


###############################################################################
# Tests rotate
def test_rotate(data):
    expected_result = np.array([[1, 3], [2, 4]])
    result = rotate(data["x"])
    np.testing.assert_array_equal(result, expected_result)


###############################################################################
# Tests rotated_h_transform
def test_rotated_h_transform(data):
    expected_result = np.array([[19, 43], [22, 50], [25, 57]])
    result = rotated_h_transform(data["x"], data["y"])
    np.testing.assert_array_equal(result, expected_result)


###############################################################################
# Tests create_permutation
def test_create_permutation():
    expected_result = np.array([0, 3, 1, 4, 2, 5])
    result = create_permutation(3, 2)
    np.testing.assert_array_equal(result, expected_result)

    expected_result = np.array([0, 2, 4, 1, 3, 5])
    result = create_permutation(2, 3)
    np.testing.assert_array_equal(result, expected_result)
