#!/usr/bin/env python
# coding: utf-8
"""
Show the essence of P-splines
=============================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pyspline.psplines import PSplines
from pyspline.basis import basis_bsplines

# Set RNG
rng = np.random.default_rng(42)


# Simulate data
n = 150
x = np.linspace(0, 1, n)
y = 0.3 + np.sin(1.2 * x + 0.3) + 0.15 * rng.normal(size=n)


# Make a matrix containing the B-spline basis
ndx = 15
deg = 3
B = basis_bsplines(x, n_functions=ndx, degree=deg, domain_min=0, domain_max=1)


# A basis for plotting the fit on the grid xg
ng = 500
xg = np.linspace(0, 1, ng)
Bg = basis_bsplines(xg, n_functions=ndx, degree=deg, domain_min=0, domain_max=1)


# Positions of the peaks of the B-splines
dk = 1 / (ndx - deg)
xa = np.arange(1, ndx + 1) * dk - (deg + 1) * dk / 2


# Estimate the coefficients and compute the fit on the grid
ps = PSplines(penalty=0.1, n_segments=(ndx - deg,), degree=(deg,))
ps.fit(X=x.reshape(-1, 1), y=y)
z = ps.predict(X=xg.reshape(-1, 1))


# Make a matrix with B-splines scaled by coefficients
Bsc = np.diag(ps.beta_hat_) @ Bg
Bsc[Bsc < 1e-4] = np.nan


# Build the graph
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(x, y, color="#AAAAAA", linewidth=0.5, zorder=3)
plt.scatter(x, y, color="#AAAAAA", s=0.5, zorder=3)
plt.plot(xg, z, color="#0047AB", linewidth=2, zorder=6)

colors = iter(cm.rainbow(np.linspace(0, 1, ndx)))
for idx in np.arange(ndx):
    c = next(colors)
    plt.scatter(xa[idx], ps.beta_hat_[idx], color=c, zorder=6)
    plt.plot(xg, Bsc[idx], color=c, zorder=3)
plt.hlines(0, xmin=-0.1, xmax=1.1, color="#000000", linewidth=0.5)
plt.grid(linestyle="-", color="#EEEEEE", zorder=0)
plt.show()
