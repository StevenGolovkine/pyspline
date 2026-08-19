#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the cv.py file."""

import numpy as np
import pytest
from pyspline.cv import cv

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
# Tests cv

def test_cv_one_dimensional(data):
    pred = cv(data["x"].reshape(-1, 1), data["y"],  n_segments=(4,), 
        degree=(1,), order_penalty=2, domains=(1,5), params=[(0.1,), (1,)])
    expected_pred = np.array([0,0])
    np.testing.assert_array_almost_equal(pred, expected_pred)

def test_cv_n_dimensional(data_2d):
    pred = cv(data_2d["x"], data_2d["y"],  n_segments=(4,4), degree=(3,3), 
       order_penalty=2, domains=[(0,1),(-0.5,0.5)], params=[(0.1,0.1), (1,1)])
    expected_pred = np.array([1.557724, 1.56465329])
    np.testing.assert_array_almost_equal(pred, expected_pred)