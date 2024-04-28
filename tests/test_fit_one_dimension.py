#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the psplines_inner.py file."""

import numpy as np
import pytest

from pyspline.psplines_inner import fit_one_dimensional


@pytest.fixture
def data():
    data = np.array([1, 2, 3, 4, 5])
    basis = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]])
    return {"data": data, "basis": basis}


###############################################################################
# Tests fit_one_dimensional
def test_fit_one_dimensional(data):
    expected_y_hat = np.array(
        [1.25143869, 1.95484728, 2.83532537, 3.89287295, 5.12749004]
    )
    expected_beta_hat = np.array([0.7250996, 0.43780434, 0.08853475])
    expected_hat_matrix = np.array(
        [0.3756529, 0.3549800, 0.2669322, 0.2788401, 0.7545816]
    )

    result = fit_one_dimensional(data["data"], data["basis"])
    np.testing.assert_array_almost_equal(result["y_hat"], expected_y_hat)
    np.testing.assert_array_almost_equal(result["beta_hat"], expected_beta_hat)
    np.testing.assert_array_almost_equal(
        result["hat_matrix"], expected_hat_matrix
    )
