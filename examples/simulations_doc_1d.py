#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspline.psplines import PSplines
import pandas as pd
from scipy.optimize import minimize

from pyspline.basis import basis_bsplines
from pyspline.cv import cv, cv_derivative, gcv, risk

ratio=0.05

# Parameters
params = [(0.001,), (0.01,), (0.05,), (0.1,), (0.5,), (1,), (5,), (10,), (50,),
          (100,)]
params_deriv = [(0.0001,), (0.001,), (0.01,), (0.1,), (0.5,), (1,),
                (5,), (10,), (50,), (100,), (150,), (200,), (250,), (300,),
                (400,), (500,), (1000,)]

nb_simu = 100
order_derivative = 1
n_segments = (30,)
degree = (3,)
domains = (0,1)
# ratios = [0.01, 0.05, 0.1]
nb_obs =[50, 100, 200]
ndx = n_segments[0] + degree[0]
x_basis = np.linspace(0, 1, 500)
METHODS = {"cv1": {"type": "cv", "penalty": "zero"},
           "cv2": {"type": "cv", "penalty": "best"},
           "gcv1": {"type": "gcv", "penalty": "zero"},
           "gcv2": {"type": "gcv", "penalty": "best"},
           "risk": {"type": "risk", "penalty": None}}

# Expected values
new_x = np.linspace(0,1,25)
# z_expected = np.sin(new_x)*np.cos(new_x)
# expected_deriv_x = np.cos(new_x) * np.cos(new_x) - np.sin(new_x
# ) * np.sin(new_x)
z_expected = np.sin(10 * new_x)
expected_deriv_x = 10 * np.cos(10 * new_x)

new_x_no_border = new_x[1:-1]
# z_expected_no_border = np.sin(new_x_no_border)*np.cos(new_x_no_border)
# expected_deriv_x_no_border = np.cos(new_x_no_border) * np.cos(new_x_no_border
#                     ) - np.sin(new_x_no_border) * np.sin(new_x_no_border)
z_expected_no_border = np.sin(10 * new_x_no_border)
expected_deriv_x_no_border = 10 * np.cos(10 * new_x_no_border)

# EQM
data_z = {}
data_deriv_x = {}
# basis_z = {}
basis_deriv_x = {}
error_z = []
error_deriv_x = []

for idx_nb_obs, n in enumerate(nb_obs):
    # for idx_ratio, ratio in enumerate(ratios):
    for method, config in METHODS.items():
        print(f"nb_obs: {n}, method: {method}")
        rng = np.random.default_rng(42)

        # Simulate data
        x = rng.uniform(0,1,n)
        # z_true = np.sin(x) * np.cos(x)
        z_true = np.sin(10 * x)
        variance = ratio * np.var(z_true)
        noise = rng.normal(loc=0, scale=np.sqrt(variance), size=n)
        z = z_true + noise

        # CV
        loocv = cv(
            X=x.reshape(-1,1),
            y=z,
            params=params,
            n_segments=n_segments,
            degree=degree,
            order_penalty= 2,
            domains=domains)
        best_penalty = params[np.argmin(loocv)]
        print(f"best lambda: {best_penalty}")

        # First derivative estimation
        if config["penalty"] == "zero":
            penalty_deriv = (0,0)
        elif config["penalty"] == "best":
            penalty_deriv = best_penalty

        # Hyperparameter selection for derivative estimation
        if config["type"] == "cv":
            ps_deriv = PSplines(n_segments=n_segments, degree=degree,
                                penalty=penalty_deriv, order_penalty=2)
            ps_deriv.fit(x.reshape(-1,1), z, domains=domains)
            estim_deriv_x = ps_deriv.derivative(new_x.reshape(-1,1),
                                order_derivative=order_derivative, dim=(0,))
            cv_deriv_x = cv_derivative(
                X=new_x.reshape(-1,1),
                deriv = estim_deriv_x,
                params=params_deriv,
                n_segments=n_segments,
                degree=degree,
                domains=domains)
            best_penalty_deriv_x = params_deriv[np.argmin(cv_deriv_x)]

        elif config["type"] == "gcv":
            ps_deriv = PSplines(n_segments=n_segments, degree=degree,
                                penalty=penalty_deriv, order_penalty=2)
            ps_deriv.fit(x.reshape(-1,1), z, domains=domains)
            estim_deriv_x = ps_deriv.derivative(new_x.reshape(-1,1),
                                order_derivative=order_derivative, dim=(0,))
            cv_deriv_x = minimize(
                fun=lambda p: gcv(
                    p,
                    X=new_x.reshape(-1,1),
                    deriv = estim_deriv_x,
                    n_segments=n_segments,
                    degree=degree,
                    domains=domains,
                    order_penalty =3), x0= best_penalty)
            best_penalty_deriv_x = cv_deriv_x.x

        elif config["type"] == "risk":
            cv_deriv_x = minimize(
                fun=lambda p: risk(
                    p,
                    X=x.reshape(-1,1),
                    y=z,
                    n_segments=n_segments,
                    degree=degree,
                    domains=domains,
                    order_penalty =3,
                    variance=variance,
                    order_derivative=1), x0= best_penalty)
            best_penalty_deriv_x = cv_deriv_x.x
            # best_penalty_deriv_x = risk(
            #                     X=x.reshape(-1,1),
            #                     y=z,
            #                     params=params_deriv,
            #                     n_segments=n_segments,
            #                     degree=degree,
            #                     domains=domains,
            #                     order_penalty =3,
            #                     variance=variance,
            #                     order_derivative=1)
        else:
            print("ERROR")

        for idx_simu in range(nb_simu):
            rng = np.random.default_rng(2*idx_simu)
            x = rng.uniform(0,1,n)
            # z_true = np.sin(x) * np.cos(x)
            z_true = np.sin(10 * x)
            noise = rng.normal(loc=0, scale=np.sqrt(ratio * np.var(z_true)),
                               size=n)
            z = z_true + noise

            # Fit the model
            ps = PSplines(n_segments=n_segments, degree=degree,
                          penalty=best_penalty, order_penalty=2)
            ps.fit(x.reshape(-1,1), z, domains=domains)

            # EQM derivatives
            ps_deriv = PSplines(n_segments=n_segments, degree=degree,
                            penalty=best_penalty_deriv_x, order_penalty=2)

            if config["penalty"] == "zero":
                ps_deriv.fit(x.reshape(-1,1), z, domains=domains)

                # EQM df/dx with borders
                estim_deriv = ps_deriv.derivative(new_x.reshape(-1,1),
                                order_derivative=order_derivative, dim=(0,))
                error = np.sum((expected_deriv_x - estim_deriv)**2)
                error_deriv_x.append({"method": method, "nb_obs": n,
                                    "error": error, "borders": 1})

                # EQM df/dx without borders
                estim_deriv = ps_deriv.derivative(
                    new_x_no_border.reshape(-1,1),
                    order_derivative=order_derivative, dim=(0,))
                error = np.sum((expected_deriv_x_no_border - estim_deriv)**2)
                error_deriv_x.append({"method": method, "nb_obs": n,
                                    "error": error, "borders": 0})

            elif config["penalty"] == "best":
                estim_deriv = ps.derivative(
                    new_x.reshape(-1,1),
                    order_derivative=order_derivative, dim=(0,))
                ps_deriv.fit(new_x.reshape(-1,1), estim_deriv, domains=domains)

                # EQM with borders
                new_estim_deriv = ps_deriv.predict(new_x.reshape(-1,1))
                error = np.sum((expected_deriv_x - new_estim_deriv)**2)
                error_deriv_x.append({"method": method, "nb_obs": n,
                                    "error": error, "borders": 1})

                # EQM without borders
                new_estim_deriv = ps_deriv.predict(
                    new_x_no_border.reshape(-1,1))
                error = np.sum(
                    (expected_deriv_x_no_border - new_estim_deriv)**2)
                error_deriv_x.append({"method": method, "nb_obs": n,
                                    "error": error, "borders": 0})

            elif config["penalty"] is None:
                basis_one_dimensional = basis_bsplines(
                                    argvals=x.squeeze(),
                                    n_functions=n_segments[0] + degree[0],
                                    degree=degree[0],
                                    domain_min=float(domains[0]),
                                    domain_max=float(domains[1]),
                                    ).T

                btb = (basis_one_dimensional.T @ basis_one_dimensional
                        + np.eye(basis_one_dimensional.T.shape[0]) * 1e-4 )
                diff = np.diff(np.eye(basis_one_dimensional.shape[1]),  n=2).T
                alpha_lambda = (np.linalg.solve(btb
                            + best_penalty_deriv_x * diff.T @ diff,
                            basis_one_dimensional.T) @ z)

                # EQM df/dx with borders
                basis_one_dimensional_deriv = basis_bsplines(
                                    argvals=new_x.squeeze(),
                                    n_functions = (n_segments[0] + degree[0]
                                            - order_derivative),
                                    degree=degree[0],
                                    domain_min=float(domains[0]),
                                    domain_max=float(domains[1]),
                                    ).T
                diff_deriv = np.diff(np.eye(basis_one_dimensional.shape[1]),
                                                    order_derivative).T
                D_r = (basis_one_dimensional_deriv @ diff_deriv
                        /((domains[1]- domains[0])
                            / n_segments[0])**order_derivative)
                estim_deriv = D_r @ alpha_lambda

                error = np.sum((expected_deriv_x - estim_deriv)**2)
                error_deriv_x.append({"method": method, "nb_obs": n,
                                    "error": error, "borders": 1})

                # EQM df/dx without borders
                basis_one_dimensional = basis_bsplines(
                                    argvals=new_x_no_border.squeeze(),
                                    n_functions=n_segments[0] + degree[0],
                                    degree=degree[0],
                                    domain_min=float(domains[0]),
                                    domain_max=float(domains[1]),
                                    ).T
                basis_one_dimensional_deriv = basis_bsplines(
                                    argvals=new_x_no_border.squeeze(),
                                    n_functions = (n_segments[0] + degree[0]
                                            - order_derivative),
                                    degree=degree[0],
                                    domain_min=float(domains[0]),
                                    domain_max=float(domains[1]),
                                    ).T
                diff_deriv = np.diff(np.eye(basis_one_dimensional.shape[1]),
                                                    order_derivative).T
                D_r = (basis_one_dimensional_deriv @ diff_deriv
                        /((domains[1]- domains[0])
                            / n_segments[0])**order_derivative)
                estim_deriv = D_r @ alpha_lambda

                error = np.sum((expected_deriv_x_no_border - estim_deriv)**2)
                error_deriv_x.append({"method": method, "nb_obs": n,
                                    "error": error, "borders": 0})

# Figures EQM
ratio_formatted = str(ratio).replace(".", "_")
# for error, title in zip([error_z, error_deriv_x],
#       [f"EQM_f_1d_all_ratio_{ratio_formatted}_{n_segments[0]}_segments",
#      f"EQM_df_dx_1d_all_ratio_{ratio_formatted}_{n_segments[0]}_segments"]):
title =  f"EQM_df_dx_1d_all_ratio_{ratio_formatted}_{n_segments[0]}_segments"
error = error_deriv_x

error = pd.DataFrame(error)
# fig, ax1 = plt.subplots()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 13))

sns.boxplot(ax=ax1, x=error.loc[error["borders"] == 1,"method"],
            y=error.loc[error["borders"] == 1,"error"],
            hue=error.loc[error["borders"] == 1,"nb_obs"])
ax1.set_yscale('log')
ax1.set_title("Bords: Oui")
ax1.set_ylabel("log(EQM)")
ax1.set_xlabel("Méthode")
ax1.set_xticks([0, 1, 2, 3, 4],
                ['cv 1', 'cv 2', 'gcv 1', 'gcv 2', 'risque'])
handles, _ = ax1.get_legend_handles_labels()
new_labels = ['n=50', 'n=100', 'n=200']
ax1.legend(handles=handles, labels=new_labels)

sns.boxplot(ax=ax2, x=error.loc[error["borders"] == 0,"method"],
            y=error.loc[error["borders"] == 0,"error"],
            hue=error.loc[error["borders"] == 0,"nb_obs"])
ax2.set_yscale('log')
ax2.set_title("Bords: Non")
ax2.set_ylabel("log(EQM)")
ax2.set_xlabel("")
ax2.set_xticks([0, 1, 2, 3, 4],
                ['cv 1', 'cv 2', 'gcv 1', 'gcv 2', 'risque'])
handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=new_labels)
ax2.set_xlabel("Méthode")

fig.savefig(title)
plt.tight_layout()
plt.show()
