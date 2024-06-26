#!/usr/bin/env python
# coding: utf-8
"""
B-spline fits with same basis having varying roughness
======================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pyspline.basis import basis_bsplines


# Set RNG
rng = np.random.default_rng(42)


# Make basis
m = 200
x = np.linspace(0, 1, m)
nseg = 10
deg = 3
n = nseg + deg
B = basis_bsplines(x, n_functions=n)


# Make coefficients
A1 = rng.uniform(0, 1, n)
A2 = 0.8 * np.sin(2 * np.arange(1, n + 1) / n) + rng.uniform(0, 1, n) * 0.2
A3 = np.arange(1, n + 1) / n
A4 = np.repeat(1, n)
A = np.vstack([A1, A2, A3, A4])
Z = A @ B


# Generate the plots
fig = plt.figure(figsize=(6, 4), dpi=300)
axs = fig.subplots(2, 2, sharex=True)

for idx in np.arange(4):
    # Compute roughness
    Aj = A[idx]
    R = np.sqrt(np.sum(np.diff(Aj) ** 2) / (n - 1))

    # Scaled basis
    Bsc = np.diag(Aj) @ B
    # Remove zero entries
    Bsc[Bsc < 1e-4] = np.nan

    knots = (np.arange(1, n + 1) - 2) / nseg

    axs[idx // 2, idx % 2].scatter(
        knots, Aj, edgecolors="r", facecolors="none", zorder=3
    )
    axs[idx // 2, idx % 2].plot(x, Z[idx], c="b", zorder=3)
    colors = iter(cm.rainbow(np.linspace(0, 1, n)))
    for j in np.arange(n):
        c = next(colors)
        axs[idx // 2, idx % 2].plot(x, Bsc[j], color=c, zorder=3)
    axs[idx // 2, idx % 2].grid(linestyle="-", color="#EEEEEE", zorder=0)
    axs[idx // 2, idx % 2].set_title(f"r = {R:.2}")
