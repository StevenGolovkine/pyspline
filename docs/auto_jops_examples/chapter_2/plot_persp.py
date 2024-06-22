#!/usr/bin/env python
# coding: utf-8
"""
B-splines in perspective
========================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pyspline.basis import basis_bsplines


# Basis on grid
ndx = 7
deg = 3
ng = 500
xmin = 0
xmax = 4
xg = np.linspace(xmin, xmax, ng)
Bg = basis_bsplines(
    xg, n_functions=ndx, degree=deg, domain_min=xmin, domain_max=xmax
)


# Make a matrix with B-splines scaled by coefficients
Bsc = Bg + np.outer(np.arange(1, ndx + 1), np.ones(ng))


# Select one row, for visualization
k = 160
xk = xg[k]
bk2 = Bg[:, k]
bk1 = bk2 + np.arange(1, ndx + 1)
bk2[bk2 < 1e-3] = np.nan


# For plotting
Bg[Bg < 1e-4] = np.nan


# Build the graphs
fig = plt.figure(figsize=(6, 4), dpi=300)
axs = fig.subplots(1, 2, sharex=True)

colors = iter(cm.rainbow(np.linspace(0, 1, ndx)))
for idx in np.arange(ndx):
    c = next(colors)
    axs[0].plot(xg, Bsc[idx].T, color=c, zorder=4)
    axs[0].scatter(xk, bk1[idx], color=c, zorder=5)
    axs[1].plot(xg, Bg[idx].T, color=c, zorder=4)
    axs[1].scatter(xk, bk2[idx], color=c, zorder=5)

axs[0].axvline(x=xk, color="k", linestyle="dashed", linewidth=1, zorder=3)
axs[0].axhline(y=0, color="k", linewidth=0.2, zorder=3)
axs[0].set_title("Perspective view")
axs[0].grid(linestyle="-", color="#EEEEEE", zorder=0)

axs[1].axvline(x=xk, color="k", linestyle="dashed", linewidth=1, zorder=3)
axs[1].axhline(y=0, color="k", linewidth=0.2, zorder=3)
axs[1].set_title("Columns of a B-splines basis")
axs[1].grid(linestyle="-", color="#EEEEEE", zorder=0)

plt.show()
