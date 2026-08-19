#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pyspline.psplines import PSplines
from pyspline.basis import basis_bsplines

from scipy.optimize import minimize
import pandas as pd

from pyspline.cv import cv, cv_derivative, gcv

cv_gamma = "gcv"

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
        noise = rng.normal(loc=0, scale=np.sqrt(ratio * np.var(z_true)), size=n)
        z = z_true + noise
        
        # CV
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

        # CV derivative 
        ps_deriv = PSplines(n_segments=n_segments, degree=degree, 
                            penalty=(0,0), order_penalty=2)
        ps_deriv.fit(x.reshape(-1,1), z, domains=domains)

        # new_z = ps_deriv.predict(new_x.reshape(-1,1))   
        estim_deriv_x = ps_deriv.derivative(new_x.reshape(-1,1), 
                            order_derivative=order_derivative, dim=(0,))

        # data_test[(idx_ratio, n)] = {"x": new_x, "pred": estim_deriv_x}

        match cv_gamma: 
            case "cv":
                cv_deriv_x = cv_derivative(
                    X=new_x.reshape(-1,1),
                    deriv = estim_deriv_x,
                    params=params_deriv,
                    n_segments=n_segments,
                    degree=degree,
                    domains=domains, 
                    order_penalty =3)
                best_penalty_deriv_x = params_deriv[np.argmin(cv_deriv_x)]
            case "gcv": 
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

            case _ :
                best_penalty_deriv_x = best_penalty

        print(f"best gamma x: {best_penalty_deriv_x}")

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
                          penalty=best_penalty, order_penalty=3)
            ps.fit(x.reshape(-1,1), z, domains=domains)

            if idx_simu == 0:
                bx = basis_bsplines(argvals=x_basis, n_functions=ndx, 
                                    degree=degree[0]
                                )
                Bsc = np.diag(ps.beta_hat_) @ bx
                basis_z[(idx_ratio, n)] = Bsc

            # EQM f with borders 
            new_z = ps.predict(new_x.reshape(-1,1))   
            error = np.sum((z_expected - new_z)**2)
            error_z.append({"ratio": idx_ratio, "nb_obs": n, "error": error, 
                            "borders": 1})
            if idx_simu == 0:
                data_z[(idx_ratio, n)] = {"x": x, "simulated": z, 
                            "new_x": new_x, "pred": new_z, 
                            "penalty": best_penalty[0], "eqm": error}
                
            # EQM f without borders 
            new_z_no_border = ps.predict(new_x_no_border.reshape(-1,1))
            error = np.sum((z_expected_no_border - new_z_no_border)**2)
            error_z.append({"ratio": idx_ratio,  "nb_obs": n, "error": error, 
                            "borders": 0})
            
            ps = PSplines(n_segments=n_segments, degree=degree, 
                        penalty=best_penalty_deriv_x, order_penalty=3)
            ps.fit(x.reshape(-1,1), z, domains=domains)

            # EQM df/dx with borders
            estim_deriv = ps.derivative(new_x.reshape(-1,1), 
                                order_derivative=order_derivative, dim=(0,))
            error = np.sum((expected_deriv_x - estim_deriv)**2)
            error_deriv_x.append({"ratio": idx_ratio, "nb_obs": n, 
                                "error": error, "borders": 1})
            if idx_simu == 0:
                data_deriv_x[(idx_ratio, n)] = {"new_x": new_x, 
                            "pred": estim_deriv, 
                            "penalty": best_penalty_deriv_x[0], "eqm": error}
            
            # EQM df/dx without borders
            estim_deriv = ps.derivative(new_x_no_border.reshape(-1,1), 
                order_derivative=order_derivative, dim=(0,))
            error = np.sum((expected_deriv_x_no_border - estim_deriv)**2)
            error_deriv_x.append({"ratio": idx_ratio, "nb_obs": n, 
                                "error": error, "borders": 0})
        
# Figures EQM 
match cv_gamma: 
    case "cv": 
        title_deriv = f"EQM_df_dx_1_1d_{n_segments[0]}_segments"
    case "gcv": 
        title_deriv = f"EQM_df_dx_1_1d_gcv_{n_segments[0]}_segments"
    case "no": 
        title_deriv = f"EQM_df_dx_1_1d_no_cv_{n_segments[0]}_segments"

for error, title in zip([error_z, error_deriv_x], 
                        [f"EQM_f_1d_{n_segments[0]}_segments", title_deriv]):
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

# Build the graph f(x)
fig, axes = plt.subplots(3, 3, figsize=(10, 13))
for i, n in enumerate(nb_obs): 
    for j, ratio in enumerate(ratios): 
        sorted_indices = np.argsort(data_z[(j,n)]["x"])
        x_sorted = data_z[(j,n)]["x"][sorted_indices]
        y_sorted = data_z[(j,n)]["simulated"][sorted_indices]
        axes[i, j].plot(x_sorted, y_sorted, color="#AAAAAA", 
                        linewidth=0.5, label='Données simulées', zorder=3)
        axes[i,j].scatter(x_sorted, y_sorted, color="#AAAAAA", 
                s=0.5, zorder=3)
        
        axes[i,j].plot(new_x, z_expected, color="#ab0000", 
                linestyle='dashed',  linewidth=2, 
                label='f(x)', zorder=4)
        axes[i,j].plot(data_z[(j,n)]["new_x"], 
                data_z[(j,n)]["pred"], color="#0047AB", 
                marker='o',  linewidth=2, label='Valeurs prédites', 
                zorder=6)
        basis = basis_z[(j,n)]
        colors = iter(cm.rainbow(np.linspace(0, 1, ndx)))
        for idx in range(ndx):
            c = next(colors)
            axes[i,j].plot(x_basis, basis[idx], color =c) 

        penalty = data_z[(j, n)]["penalty"]
        eqm = np.mean(data_z[(j, n)]["eqm"])

        axes[i, j].set_title(
            rf"$\lambda$={penalty}, EQM={eqm:.4f}"
        )
        if j == 0: 
            axes[i,j].set_ylabel(f"nb obs: {n}")
        if i == 2:
            axes[i,j].set_xlabel(f"Ratio signal bruit: {ratio}")
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels)
fig.savefig(f"f_1d_{n_segments[0]}_segments")
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
match cv_gamma: 
    case "cv": 
        title = f"df_dx_1_1d_{n_segments[0]}_segments"
    case "gcv": 
        title = f"df_dx_1_1d_gcv_{n_segments[0]}_segments"
    case "no": 
        title = f"df_dx_1_1d_no_cv_{n_segments[0]}_segments"
fig.savefig(title)
plt.tight_layout()
plt.show()

# # Figure df/dx penalty = 0
# fig, axes = plt.subplots(3, 3, figsize=(10, 13))
# for i, n in enumerate(nb_obs): 
#     for j, ratio in enumerate(ratios): 
#         axes[i,j].plot(new_x, expected_deriv_x, color="#ab0000", linewidth=2, 
#                 linestyle="dashed", label=r"$\partial f(x) / \partial x$", 
#                 zorder=4)
#         sorted_indices = np.argsort(data_test[(j,n)]["x"])
#         x_sorted = data_test[(j,n)]["x"][sorted_indices]
#         y_sorted = data_test[(j,n)]["pred"][sorted_indices]
#         axes[i,j].plot(x_sorted.reshape(-1, 1), 
#                         y_sorted, color="#0047AB", marker='o',  linewidth=2, 
#                         label='Sans pénalité', zorder=6)
        
# handles, labels = axes[0,0].get_legend_handles_labels()
# fig.legend(handles, labels)
# fig.savefig(f"df_dx_1_1d_{n_segments[0]}_segments_no_penalty")
# plt.tight_layout()
# plt.show()