#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the psplines.py file."""

import numpy as np
import pytest

from pyspline.psplines import PSplines


@pytest.fixture
def data():
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return {"x": X, "y": Y}


@pytest.fixture
def data_2d():
    return 0


###############################################################################
# Tests PSplines
def test_getter():
    ps = PSplines()

    np.testing.assert_equal(ps.penalty, (1.0,))
    np.testing.assert_equal(ps.n_segments, (10,))
    np.testing.assert_equal(ps.degree, (3,))
    np.testing.assert_equal(ps.order_penalty, 2)


def test_fit_one_dimensional(data):
    ps = PSplines(n_segments=(5,))
    ps.fit(data["x"].reshape(-1, 1), data["y"])

    expected_beta = np.array([0.2, 1.0, 1.8, 2.6, 3.4, 4.2, 5.0, 5.8])
    np.testing.assert_array_almost_equal(ps.beta_hat_, expected_beta)

    expected_y = data["y"]
    np.testing.assert_array_almost_equal(ps.y_hat_, expected_y)


def test_predict_one_dimensional(data):
    ps = PSplines(n_segments=(5,))
    ps.fit(data["x"].reshape(-1, 1), data["y"])
    pred = ps.predict(np.array([2.5, 4.5]).reshape(-1, 1))

    expected_pred = np.array([2.5, 4.5])
    np.testing.assert_array_almost_equal(pred, expected_pred)
