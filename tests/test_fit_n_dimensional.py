#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the psplines_inner.py file."""

import numpy as np
import pytest

from pyspline.psplines_inner import fit_n_dimensional


@pytest.fixture
def data():
    data = np.array([[1, 2], [3, 4]])
    basis = [np.array([[1, 1], [1, 2]]), np.array([[1, 1], [2, 3]])]
    return {"data": data, "basis": basis}


###############################################################################
# Tests fit_n_dimensional
def test_fit_n_dimensional(data):
    expected_y_hat = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_beta_hat = np.array([[-3.0, 1.0], [2.0, 0.0]])
    expected_hat_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

    result = fit_n_dimensional(data["data"], data["basis"])
    np.testing.assert_array_almost_equal(result["y_hat"], expected_y_hat)
    np.testing.assert_array_almost_equal(result["beta_hat"], expected_beta_hat)
    np.testing.assert_array_almost_equal(
        result["hat_matrix"], expected_hat_matrix
    )
