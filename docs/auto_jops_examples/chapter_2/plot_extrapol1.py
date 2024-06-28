#!/usr/bin/env python
# coding: utf-8
"""
Illustration of interpolation and extrapolation by penalty order
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from pyspline.psplines import PSplines


# Set RNG
rng = np.random.default_rng(42)


# Simulate data
m = 50
x = rng.uniform(0, 1, m)
y = np.sin(2.5 * x) + 0.05 * rng.normal(0, 1, m) + 0.2


# Delete some data
mask = np.array(
    [(xx > 0.2 and xx < 0.4) or (xx > 0.6 and xx < 0.8) for xx in x]
)
x_sub = x[mask]
y_sub = y[mask]


# Parameters
ndx = 25
deg = 3
pen = 1
xg = np.linspace(0, 1, 500)
knots = (np.arange(1, ndx + deg + 1) - 2) / ndx


# First order difference
ps_one = PSplines(
    penalty=(pen,), n_segments=(ndx,), degree=(deg,), order_penalty=1
)
ps_one.fit(X=x_sub.reshape(-1, 1), y=y_sub, domains=(0, 1))
y_one = ps_one.predict(X=xg.reshape(-1, 1))


# Second order difference
ps_two = PSplines(
    penalty=(pen,), n_segments=(ndx,), degree=(deg,), order_penalty=2
)
ps_two.fit(X=x_sub.reshape(-1, 1), y=y_sub, domains=(0, 1))
y_two = ps_two.predict(X=xg.reshape(-1, 1))


# Build the graph
fig = plt.figure(figsize=(6, 4), dpi=300)
axs = fig.subplots(1, 2, sharex=True)

axs[0].scatter(x_sub, y_sub, color="#AAAAAA", s=0.5, zorder=3)
axs[0].plot(xg, y_one, color="#0047AB", linewidth=1, zorder=4)
axs[0].scatter(
    knots, ps_one.beta_hat_, edgecolors="r", facecolors="none", s=5, zorder=5
)
axs[0].hlines(0, xmin=0, xmax=1, color="#000000", linewidth=0.5)
axs[0].grid(linestyle="-", color="#EEEEEE", zorder=0)
axs[0].set_title("First differences")

axs[1].scatter(x_sub, y_sub, color="#AAAAAA", s=0.5, zorder=3)
axs[1].plot(xg, y_two, color="#0047AB", linewidth=1, zorder=4)
axs[1].scatter(
    knots, ps_two.beta_hat_, edgecolors="r", facecolors="none", s=5, zorder=5
)
axs[1].hlines(0, xmin=0, xmax=1, color="#000000", linewidth=0.5)
axs[1].grid(linestyle="-", color="#EEEEEE", zorder=0)
axs[1].set_title("Second differences")

plt.show()
