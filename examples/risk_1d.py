#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspline.basis import basis_bsplines
from pyspline.cv import cv, risk
from scipy.optimize import minimize

cv_gamma = "risk"

# Parameters 
params = [(0.001,), (0.01,), (0.05,), (0.1,), (0.5,), (1,), (5,), (10,), (50,), 
          (100,)]
params_deriv = [(0.0001,), (0.001,), (0.01,), (0.1,), (0.5,), (1,), 
                (5,), (10,), (50,), (100,), (150,), (200,), (250,), (300,), 
                (400,), (500,), (600,), (700,), (800,), (900,), (1000,)]

nb_simu = 100
order_derivative = 1
n_segments = (30,)
degree = (3,)
domains = (0,1)
ratios = [0.01, 0.05, 0.1]
nb_obs =[50, 100, 200]
ndx = n_segments[0] + degree[0]
x_basis = np.linspace(0, 1, 500)

# Expected values 
new_x = np.linspace(0,1,25)
# z_expected = np.sin(new_x)*np.cos(new_x)
# expected_deriv_x = np.cos(new_x) * np.cos(new_x) - np.sin(new_x) * 
# np.sin(new_x)
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
basis_z = {}
basis_deriv_x = {}
error_z = []
error_deriv_x = []
data_test = {}

for idx_nb_obs, n in enumerate(nb_obs): 
    for idx_ratio, ratio in enumerate(ratios): 
        print(f"nb_obs: {n}, ratio: {ratio}")
        rng = np.random.default_rng(42)

        # Simulate data
        x = rng.uniform(0,1,n)
        # z_true = np.sin(x) * np.cos(x)
        z_true = np.sin(10 * x)
        variance = ratio * np.var(z_true)
        noise = rng.normal(loc=0, scale=np.sqrt(variance), size=n)
        z = z_true + noise

        # Estimate risk
        # best_penalty_deriv_x = risk(
        #             X=x.reshape(-1,1), 
        #             y=z,
        #             params=params_deriv,
        #             n_segments=n_segments,
        #             degree=degree,
        #             domains=domains, 
        #             order_penalty =3, 
        #             variance=variance,
        #             order_derivative=1)

        loocv = cv(
            X=x.reshape(-1,1),
            y=z,
            params=params,
            n_segments=n_segments,
            degree=degree,
            order_penalty= 3,
            domains=domains)
        best_penalty = params[np.argmin(loocv)]
        print(f"best lambda: {best_penalty}")

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
        

        for idx_simu in range(nb_simu): 
            rng = np.random.default_rng(2*idx_simu)
            x = rng.uniform(0,1,n)
            # z_true = np.sin(x) * np.cos(x)
            z_true = np.sin(10 * x)
            variance = ratio * np.var(z_true)
            noise = rng.normal(loc=0, scale=np.sqrt(variance), size=n)
            z = z_true + noise

            # EQM df/dx with borders
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
            error_deriv_x.append({"ratio": idx_ratio, "nb_obs": n, 
                                "error": error, "borders": 1})
            if idx_simu == 0:
                data_deriv_x[(idx_ratio, n)] = {"new_x": new_x, 
                            "pred": estim_deriv, 
                            "penalty": best_penalty_deriv_x[0], "eqm": error}
            
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
            error_deriv_x.append({"ratio": idx_ratio, "nb_obs": n, 
                                "error": error, "borders": 0})
        
# Figures EQM df/dx
error = error_deriv_x
title = f"EQM_df_dx_1d_risk_{n_segments[0]}_segments"
error = pd.DataFrame(error)
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 13))

sns.boxplot(ax=ax1, x=error.loc[error["borders"] == 1,"ratio"], 
            y=error.loc[error["borders"] == 1,"error"], 
            hue=error.loc[error["borders"] == 1,"nb_obs"])
ax1.set_yscale('log') 
ax1.set_title("Bords: Oui")
ax1.set_ylabel("log(EQM)")
ax1.set_xlabel("Ratio signal bruit")    
ax1.set_xticks([0, 1, 2], ['0.01', '0.05', '0.10']) 
handles, _ = ax1.get_legend_handles_labels()
new_labels = ['n=50', 'n=100', 'n=200']
ax1.legend(handles=handles, labels=new_labels)

sns.boxplot(ax=ax2, x=error.loc[error["borders"] == 0,"ratio"], 
            y=error.loc[error["borders"] == 0,"error"], 
            hue=error.loc[error["borders"] == 0,"nb_obs"])
ax2.set_yscale('log') 
ax2.set_title("Bords: Non")
ax2.set_ylabel("log(EQM)")
ax2.set_xlabel("")
ax2.set_xticks([0, 1, 2], ['0.01', '0.05', '0.10']) 
handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=new_labels)
ax2.set_xlabel("Ratio signal bruit")
fig.savefig(title)
plt.tight_layout()
plt.show()

# Build the graph df/dx
fig, axes = plt.subplots(3, 3, figsize=(10, 13))
for i, n in enumerate(nb_obs): 
    for j, ratio in enumerate(ratios): 
        axes[i,j].plot(new_x, expected_deriv_x, color="#ab0000", linewidth=2, 
                linestyle="dashed", label=r"$\partial f(x) / \partial x$", 
                zorder=4)
        
        axes[i,j].plot(data_deriv_x[(j,n)]["new_x"], 
                        data_deriv_x[(j,n)]["pred"], color="#0047AB", 
                        marker='o',  linewidth=2, label='Valeurs prédites', 
                        zorder=6)
        penalty = data_deriv_x[(j, n)]["penalty"]
        eqm = np.mean(data_deriv_x[(j, n)]["eqm"])

        axes[i, j].set_title(
            rf"$\gamma$={penalty:.4f}, EQM={eqm:.4f}"
        )

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels)
fig.savefig(f"df_dx_1d_risk_{n_segments[0]}_segments")
plt.tight_layout()
plt.show()
