#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the psplines.py file."""

import numpy as np
import pytest

from pyspline.psplines import PSplines


@pytest.fixture
def data():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1, 1, 1, 1, 1])
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
# Tests PSplines
def test_getter():
    ps = PSplines()

    np.testing.assert_equal(ps.penalty, (1.0,))
    np.testing.assert_equal(ps.n_segments, (10,))
    np.testing.assert_equal(ps.degree, (3,))
    np.testing.assert_equal(ps.order_penalty, 2)


def test_fit_one_dimensional(data):
    ps = PSplines(n_segments=(5,))
    ps.fit(data["x"].reshape(-1, 1), data["y"], sample_weights=data["weights"])

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


def test_fit_n_dimensional(data_2d):
    ps = PSplines(penalty=(1, 1), n_segments=(4, 4), degree=(1, 1))
    ps.fit(data_2d["x"], data_2d["y"])

    expected_beta = np.array(
        [
            [1.13891101, 1.62504247, 2.07445424, 2.52024028, 2.96277288],
            [2.44441304, 2.70060251, 2.97906136, 3.28184828, 3.59633628],
            [3.64772374, 3.70634893, 3.84193587, 4.04057514, 4.27038025],
            [4.62438241, 4.59267883, 4.6360308, 4.79261388, 5.01368424],
            [5.52524285, 5.40086044, 5.35227626, 5.53343682, 5.82386187],
        ]
    )
    np.testing.assert_array_almost_equal(ps.beta_hat_, expected_beta)

    expected_y = np.array(
        [
            [1.13891101, 2.07445424, 2.96277288],
            [3.64772374, 3.84193587, 4.27038025],
            [5.52524285, 5.35227626, 5.82386187],
        ]
    )
    np.testing.assert_array_almost_equal(ps.y_hat_, expected_y)


def test_predict_n_dimensional(data_2d):
    ps = PSplines(penalty=(1, 1), n_segments=(4, 4), degree=(1, 1))
    ps.fit(data_2d["x"], data_2d["y"])
    pred = ps.predict(np.array([[0.25, -0.25], [0.75, 0.25]]))

    expected_pred = np.array(
        [[2.70060251, 3.28184828], [4.59267883, 4.79261388]]
    )
    np.testing.assert_array_almost_equal(pred, expected_pred)


def test_errors_one_dimensional(data):
    ps = PSplines(n_segments=(5,))
    ps.fit(data["x"].reshape(-1, 1), data["y"])
    pred = ps.errors(np.array([2.5, 4.5]).reshape(-1, 1))

    expected_pred = np.array([0, 0])
    np.testing.assert_array_almost_equal(pred, expected_pred)


def test_errors_n_dimensional(data_2d):
    ps = PSplines(penalty=(1, 1), n_segments=(4, 4), degree=(1, 1))
    ps.fit(data_2d["x"], data_2d["y"])
    with pytest.raises(NotImplementedError):
        ps.errors(X=data_2d["x"])


def test_derivative_one_dimensional(data):
    ps = PSplines(n_segments=(5,))
    ps.fit(data["x"].reshape(-1, 1), data["y"])
    pred = ps.derivative(
        np.array([2.5, 4.5]).reshape(-1, 1), order_derivative=1
    )

    expected_pred = np.array([1, 1])
    np.testing.assert_array_almost_equal(pred, expected_pred)


def test_derivative_n_dimensional(data_2d):
    ps = PSplines(penalty=(1, 1), n_segments=(4, 4), degree=(1, 1))
    ps.fit(data_2d["x"], data_2d["y"])
    with pytest.raises(NotImplementedError):
        ps.derivative(X=data_2d["x"], order_derivative=1)
