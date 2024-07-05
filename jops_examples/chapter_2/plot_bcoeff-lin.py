#!/usr/bin/env python
# coding: utf-8
"""
View of linear B-spline coefficients
====================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pyspline.psplines import PSplines
from pyspline.basis import basis_bsplines


# Set RNG
rng = np.random.default_rng(42)


# Simulate data
n = 50
x = np.linspace(0, 1, n)
y = 0.3 + np.sin(1.2 * x + 0.3) + 0.15 * rng.normal(size=n)


# Parameters
ndx = 13
deg = 1


# A basis for plotting the fit on the grid xg
ng = 500
xg = np.linspace(0, 1, ng)
Bg_wiggly = basis_bsplines(
    xg, n_functions=ndx, degree=deg, domain_min=0, domain_max=1
)
Bg_smooth = basis_bsplines(
    xg, n_functions=ndx, degree=deg, domain_min=0, domain_max=1
)


# (Wiggly) Estimate the coefficients and compute the fit on the grid
ps_wiggly = PSplines(penalty=0.1, n_segments=(ndx - deg,), degree=(deg,))
ps_wiggly.fit(X=x.reshape(-1, 1), y=y)
z_wiggly = ps_wiggly.predict(X=xg.reshape(-1, 1))


# (Smooth) Estimate the coefficients and compute the fit on the grid
ps_smooth = PSplines(penalty=3, n_segments=(ndx - deg,), degree=(deg,))
ps_smooth.fit(X=x.reshape(-1, 1), y=y)
z_smooth = ps_smooth.predict(X=xg.reshape(-1, 1))


# Positions of the peaks of the B-splines
dk = 1 / (ndx - deg)
xa = np.arange(1, ndx + 1) * dk - (deg + 1) * dk / 2


# Make a matrix with B-splines scaled by coefficients
Bsc_wiggly = np.diag(ps_wiggly.beta_hat_) @ Bg_wiggly
Bsc_wiggly[Bsc_wiggly < 1e-4] = np.nan

Bsc_smooth = np.diag(ps_smooth.beta_hat_) @ Bg_smooth
Bsc_smooth[Bsc_smooth < 1e-4] = np.nan


# Build the graph
fig = plt.figure(figsize=(6, 4), dpi=300)
axs = fig.subplots(1, 2, sharex=True)

axs[0].scatter(xa, ps_wiggly.beta_hat_, color="red", s=10, zorder=4)
axs[0].plot(xg, z_wiggly, color="#0047AB", linewidth=2, zorder=3)

colors = iter(cm.rainbow(np.linspace(0, 1, ndx)))
for idx in np.arange(ndx):
    c = next(colors)
    axs[0].plot(xg, Bsc_wiggly[idx], color=c, zorder=3)
axs[0].hlines(0, xmin=0, xmax=1, color="#000000", linewidth=0.5)
axs[0].grid(linestyle="-", color="#EEEEEE", zorder=0)
axs[0].set_title("Wiggly curve")

axs[1].scatter(xa, ps_smooth.beta_hat_, color="red", s=10, zorder=4)
axs[1].plot(xg, z_smooth, color="#0047AB", linewidth=2, zorder=3)

colors = iter(cm.rainbow(np.linspace(0, 1, ndx)))
for idx in np.arange(ndx):
    c = next(colors)
    axs[1].plot(xg, Bsc_smooth[idx], color=c, zorder=3)
axs[1].hlines(0, xmin=0, xmax=1, color="#000000", linewidth=0.5)
axs[1].grid(linestyle="-", color="#EEEEEE", zorder=0)
axs[1].set_title("Smooth curve")


plt.show()
