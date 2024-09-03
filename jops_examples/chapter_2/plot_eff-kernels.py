#!/usr/bin/env python
# coding: utf-8
"""
Equivalent kernels of Whittaker smoother with second order penalty
==================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Create the impulse
n = 201
x = np.arange(0, n, 1)
y = np.zeros(n)
y[n // 2] = 1
lambdas = [1e0, 1e2, 1e4, 1e6]


# Apply the Whittaker smoother with a difference penalty
fig = plt.figure(figsize=(6, 4), dpi=300)
axs = fig.subplots(2, 2, sharex=True)

E = np.eye(n)
for idx_p, lamb in enumerate(lambdas):
    D = np.diff(E, n=2, axis=0)
    P = lamb * D.T @ D
    H = np.linalg.pinv(E + P)

    idxs = [0, 50, 100, 150, 200]
    colors = iter(cm.rainbow(np.linspace(0, 1, len(idxs))))
    for idx in idxs:
        c = next(colors)
        axs[idx_p // 2, idx_p % 2].plot(x, H[idx], color=c, zorder=4)
    axs[idx_p // 2, idx_p % 2].grid(linestyle="-", color='#EEEEEE', zorder=0)
    axs[idx_p // 2, idx_p % 2].set_title(f"$\lambda$ = {lamb:.0e}", size=5)
