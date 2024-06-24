#!/usr/bin/env python
# coding: utf-8
"""
B-spline fits having differing support (Motorcycle data)
========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyspline.psplines import PSplines


# Get the data
data = pd.read_csv("../data/mcycle.csv").dropna()
times = data["times"].to_numpy()
accel = data["accel"].to_numpy()


def make_grid(x, n=100):
    return np.linspace(np.min(x), np.max(x), n)


# Fit based on all data
new_times = make_grid(times, 1000)
ps = PSplines(n_segments=(5,), degree=(3,), penalty=(0,))
ps.fit(times.reshape(-1, 1), accel)
new_accel = ps.predict(new_times.reshape(-1, 1))


# Fit based on data where time is greater than 5ms
mask = times > 5
sample_weights = np.zeros_like(accel)
sample_weights[mask] = 1
new_times_subset = new_times[new_times > 5]
ps_subset = PSplines(n_segments=(5,), degree=(3,), penalty=(0,))
ps_subset.fit(times.reshape(-1, 1), accel, sample_weights=sample_weights)
new_accel_subset = ps_subset.predict(new_times_subset.reshape(-1, 1))


# Build the graph
plt.figure(figsize=(6, 4), dpi=300)
plt.scatter(times, accel, color="#000000", s=2, zorder=4)
plt.plot(new_times, new_accel, color="b", zorder=5)
plt.plot(
    new_times_subset, new_accel_subset, color="r", linestyle="dashed", zorder=5
)
plt.axhline(y=0, color="k", linewidth=0.2, zorder=3)

plt.title("Motor cycle helmet data")
plt.xlabel("Times (ms)")
plt.ylabel("Acceleration (g)")
plt.grid(linestyle="-", color="#EEEEEE", zorder=0)
plt.show()
