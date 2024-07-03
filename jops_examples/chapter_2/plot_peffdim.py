#!/usr/bin/env python
# coding: utf-8
"""
Effective dimension with increased penalization
===============================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyspline.psplines import PSplines


# Get the data
data = pd.read_csv("../data/mcycle.csv").dropna()
times = data["times"].to_numpy()
accel = data["accel"].to_numpy()


# Loops for log(lambda) and order d
lambdas = np.arange(-5, 5.1, 0.1)
llambdas = 10**lambdas

ED_1 = np.zeros_like(llambdas)
for idx, lamb in enumerate(llambdas):
    ps = PSplines(
        penalty=(lamb,), n_segments=(20,), degree=(3,), order_penalty=1
    )
    ps.fit(
        X=times.reshape(-1, 1), y=accel, domains=(np.min(times), np.max(times))
    )
    ED_1[idx] = ps.diagnostics_["eff_dimension"]

ED_2 = np.zeros_like(llambdas)
for idx, lamb in enumerate(llambdas):
    ps = PSplines(
        penalty=(lamb,), n_segments=(20,), degree=(3,), order_penalty=2
    )
    ps.fit(
        X=times.reshape(-1, 1), y=accel, domains=(np.min(times), np.max(times))
    )
    ED_2[idx] = ps.diagnostics_["eff_dimension"]

ED_3 = np.zeros_like(llambdas)
for idx, lamb in enumerate(llambdas):
    ps = PSplines(
        penalty=(lamb,), n_segments=(20,), degree=(3,), order_penalty=3
    )
    ps.fit(
        X=times.reshape(-1, 1), y=accel, domains=(np.min(times), np.max(times))
    )
    ED_3[idx] = ps.diagnostics_["eff_dimension"]


# Build the graph
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(lambdas, ED_1, zorder=4, linestyle="solid", label="1")
plt.plot(lambdas, ED_2, zorder=4, linestyle="dashed", label="2")
plt.plot(lambdas, ED_3, zorder=4, linestyle="dotted", label="3")
plt.axhline(y=0, color="k", linewidth=0.2, zorder=3)
plt.ylim((-1, 25))

plt.title("Effective dimensions, across penalty order")
plt.xlabel("log10($\lambda$)")
plt.ylabel("ED")
plt.grid(linestyle="-", color="#EEEEEE", zorder=0)
plt.legend(title="Order penalty")
plt.show()
