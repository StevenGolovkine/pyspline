#!/usr/bin/env python
# coding: utf-8
"""
Compare a small and large number of B-splines (Motorcycle data)
===============================================================
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


# Small basis
new_times = make_grid(times, 1000)

ps = PSplines(n_segments=(10,), degree=(3,), penalty=(0,))
ps.fit(times.reshape(-1, 1), accel, domains=(0, 60))
new_accel_small = ps.predict(new_times.reshape(-1, 1))


# Large basis
new_times = make_grid(times, 1000)

ps = PSplines(n_segments=(20,), degree=(3,), penalty=(0,))
ps.fit(times.reshape(-1, 1), accel, domains=(0, 60))
new_accel_large = ps.predict(new_times.reshape(-1, 1))


# Build the graph
plt.figure(figsize=(6, 4), dpi=300)
plt.scatter(times, accel, color="#000000", s=2, zorder=4)
plt.plot(new_times, new_accel_large, color="b", zorder=5)
plt.plot(new_times, new_accel_small, color="r", linestyle="dashed", zorder=5)
plt.axhline(y=0, color="k", linewidth=0.2, zorder=3)

plt.title("Motorcycle helmet impact data")
plt.xlabel("Times (ms)")
plt.ylabel("Acceleration (g)")
plt.grid(linestyle="-", color="#EEEEEE", zorder=0)
plt.show()
