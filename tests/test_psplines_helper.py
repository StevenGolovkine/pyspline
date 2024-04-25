#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the psplines_inner.py file."""

import numpy as np
import pytest

from pyspline.psplines_inner import tensor_product_penalties


@pytest.fixture
def data():
    penalties = [
        np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]),
        np.array([[1.0, -1.0], [-1.0, 2.0]]),
    ]
    penalty = [np.array([[1.0, -1.0], [-1.0, 2.0]])]
    return {"penalties": penalties, "penalty": penalty}


###############################################################################
# Tests tensor_product_penalties
def test_tensor_product_penalties(data):
    expected_result = [
        np.array(
            [
                [1.0, 0.0, -1.0, -0.0, 0.0, 0.0],
                [0.0, 1.0, -0.0, -1.0, 0.0, 0.0],
                [-1.0, -0.0, 2.0, 0.0, -1.0, -0.0],
                [-0.0, -1.0, 0.0, 2.0, -0.0, -1.0],
                [0.0, 0.0, -1.0, -0.0, 2.0, 0.0],
                [0.0, 0.0, -0.0, -1.0, 0.0, 2.0],
            ]
        ),
        np.array(
            [
                [1.0, -1.0, 0.0, -0.0, 0.0, -0.0],
                [-1.0, 2.0, -0.0, 0.0, -0.0, 0.0],
                [0.0, -0.0, 1.0, -1.0, 0.0, -0.0],
                [-0.0, 0.0, -1.0, 2.0, -0.0, 0.0],
                [0.0, -0.0, 0.0, -0.0, 1.0, -1.0],
                [-0.0, 0.0, -0.0, 0.0, -1.0, 2.0],
            ]
        ),
    ]
    result = tensor_product_penalties(data["penalties"])
    np.testing.assert_allclose(result, expected_result)


def test_tensor_product_penalties_one(data):
    expected_result = np.array([[1.0, -1.0], [-1.0, 2.0]])
    result = tensor_product_penalties(data["penalty"])
    np.testing.assert_array_equal(result, expected_result)
