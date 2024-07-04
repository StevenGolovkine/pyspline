#!/usr/bin/env python
# coding: utf-8
"""
P-spline fit with twice se bands, optimal on CV (Motorcyle data)
================================================================
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

ps = PSplines(penalty=(0.8,), n_segments=(20,), degree=(3,), order_penalty=2)
ps.fit(times.reshape(-1, 1), accel)
new_accel = ps.predict(new_times.reshape(-1, 1))
errors = ps.errors(new_times.reshape(-1, 1))


# Build the graph
plt.figure(figsize=(6, 4), dpi=300)
plt.scatter(times, accel, color="#999999", s=4, zorder=4)
plt.plot(new_times, new_accel, color="b", zorder=5)
plt.plot(
    new_times, new_accel + 2 * errors, color="r", linestyle="dashed", zorder=5
)
plt.plot(
    new_times, new_accel - 2 * errors, color="r", linestyle="dashed", zorder=5
)
plt.axhline(y=0, color="k", linewidth=0.2, zorder=3)

plt.title("P-spline fit to motorcycle helmet data")
plt.xlabel("Times (ms)")
plt.ylabel("Acceleration (g)")
plt.grid(linestyle="-", color="#EEEEEE", zorder=0)
plt.show()
