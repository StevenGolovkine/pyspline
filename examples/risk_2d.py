#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns
import pandas as pd
from functools import reduce
from scipy.optimize import minimize

from pyspline.basis import basis_bsplines
from pyspline.cv import cv, risk

cv_gamma = True

# Parameters 
param_1 = [0.001, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 5, 10, 50, 100]
param_2 = [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.025, 0.005, 0.001]
params = [(i, j) for j in param_2 for i in param_1]

param_1_deriv = [0.0001, 0.001,0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
param_2_deriv = [100, 50, 10, 5, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]
params_deriv = [(i, j) for j in param_2_deriv for i in param_1_deriv]

nb_simu = 100
order_derivative = 1
n_segments = (10,10)
ratios = [0.01, 0.05, 0.1]
nb_obs =[50, 100, 200]
domains = [(0,1), (0,1)]
degree = (3,3)
order_penalty=2

dims = [(0,), (1,), (0,1)]

# Expected values 
new_x_border = np.linspace(0,1,25)
new_y_border = np.linspace(0,1,25)
new_x_grid, new_y_grid = np.meshgrid(new_x_border, new_y_border)
z_grid = new_x_grid * np.sin(2 * np.pi * new_y_grid) + new_y_grid * np.sin(
        np.pi * new_x_grid)  
deriv_x = np.sin(2 * np.pi * new_y_grid) + np.pi * new_y_grid * np.cos(
    np.pi * new_x_grid) 
deriv_y =  new_x_grid * 2 * np.pi * np.cos(2 * np.pi * new_y_grid) + np.sin(
    np.pi * new_x_grid)
deriv_xy = 2 * np.pi * np.cos(2 * np.pi * new_y_grid) + np.pi * np.cos(
    np.pi * new_x_grid)

new_x_no_border = new_x_border[1:-1]
new_y_no_border = new_y_border[1:-1]
new_x_grid_no_border, new_y_grid_no_border= np.meshgrid(new_x_no_border, 
                                                        new_y_no_border)
z_grid_no_border = new_x_grid_no_border * np.sin(
    2 * np.pi * new_y_grid_no_border) + new_y_grid_no_border * np.sin(
        np.pi * new_x_grid_no_border)  
deriv_x_no_border = np.sin(2 * np.pi * new_y_grid_no_border
                           ) + np.pi * new_y_grid_no_border * np.cos(
    np.pi * new_x_grid_no_border) 
deriv_y_no_border =  new_x_grid_no_border * 2 * np.pi * np.cos(
    2 * np.pi * new_y_grid_no_border) + np.sin(np.pi * new_x_grid_no_border)
deriv_xy_no_border = 2 * np.pi * np.cos(2 * np.pi * new_y_grid_no_border
                    ) + np.pi * np.cos( np.pi * new_x_grid_no_border)

# EQM
error_z = []
error_deriv_x = []
error_deriv_y = []
error_deriv_xy = []
error_derivs = [error_deriv_x, error_deriv_y, error_deriv_xy]
for idx_nb_obs, n in enumerate(nb_obs): 
    for idx_ratio, ratio in enumerate(ratios): 
        print(f"nb_obs: {n}, ratio: {ratio}")
        rng = np.random.default_rng(42)

        # Simulate data
        x = rng.uniform(0,1,n)
        y = rng.uniform(0,1,n)
        z_true = x * np.sin(2 * np.pi * y) + y * np.sin(np.pi * x)
        variance=ratio * np.var(z_true)
        noise = rng.normal(loc=0, scale=np.sqrt(variance), size=n)
        z = z_true + noise 

        loocv = cv(
            X=np.stack((x, y), axis=1),
            y=z,
            params=params,
            n_segments=n_segments,
            degree=degree,
            order_penalty= 2,
            domains=domains)
        best_penalty = params[np.argmin(loocv)]
        print(f"best lambda: {best_penalty}")

        
        # Estimate risk
        best_penalty_deriv = {}
        for dim in dims: 
            # best_penalty_deriv[dim] = risk(
            #     X=np.stack((x, y), axis=1),
            #     y=z,
            #     params=params_deriv,
            #     n_segments=n_segments,
            #     degree=degree,
            #     domains=domains, 
            #     order_penalty=2,
            #     variance=variance,
            #     order_derivative=1, 
            #     dim=dim)
            cv_deriv = minimize(
                fun=lambda p: risk(
                    p, 
                    X=np.stack((x, y), axis=1), 
                    y = z, 
                    n_segments=n_segments,
                    degree=degree,
                    domains=domains, 
                    order_penalty =3, 
                    variance=variance,
                    order_derivative=1, 
                    dim=dim ), 
                x0= best_penalty)
            best_penalty_deriv[dim] = cv_deriv.x   
            print(f"best gamma: {best_penalty_deriv[dim]}")

        for idx_simu in range(nb_simu): 
            rng = np.random.default_rng(2*idx_simu)
            x = rng.uniform(0,1,n)
            y = rng.uniform(0,1,n)
            z_true = x * np.sin(2 * np.pi * y) + y * np.sin(np.pi * x)
            noise = rng.normal(loc=0, scale=np.sqrt(ratio * np.var(z_true)), 
                               size=n)
            z = z_true + noise 
                
            # EQM 
            for border, derivs in (
                (1, [deriv_x, deriv_y, deriv_xy]),
                (0, [deriv_x_no_border, deriv_y_no_border, deriv_xy_no_border]),
            ):
                for deriv, error_deriv, dim in zip(derivs, error_derivs, dims):
                    if border: 
                        new_x=new_x_border 
                        new_y=new_y_border
                    else: 
                        new_x=new_x_no_border 
                        new_y=new_y_no_border

                    # Basis
                    basis = [ basis_bsplines(
                            argvals=argvals,
                            n_functions=n_segments + degree,
                            degree=degree,
                            domain_min=float(domain[0]),
                            domain_max=float(domain[1]),
                            ).T
                            for argvals, n_segments, degree, domain in zip(
                                (np.stack((x, y), axis=1)).T, 
                                n_segments, degree, domains
                            )
                            ]

                    # Penalty matrix with best parameter 
                    diff_mat = [np.diff(np.eye(b.shape[1]), order_penalty).T 
                                for b in basis]
                    pen_mat = []
                    for i in range(len(basis)): 
                        matrices = []
                        for j, b in enumerate(basis): 
                            if j == i: 
                                matrices.append(diff_mat[j])
                            else: 
                                matrices.append(np.eye(b.shape[1]))
                        temp = reduce(np.kron, matrices)
                        pen_mat.append(temp.T @ temp)
                    total_penalty = 0 
                    for i in range(len(basis)): 
                        total_penalty += best_penalty_deriv[dim][i] * pen_mat[i]
        
                    small_penalty = 0 
                    for i in range(len(basis)): 
                        small_penalty += (1e-04 * pen_mat[i])

                    # Kronecker product
                    shape_b_kron = 1
                    for b in basis: 
                        shape_b_kron *= b.shape[1]
                    b_kron = np.zeros((n, shape_b_kron))
                    for i in range(n): 
                        temp = basis[0][i,:]
                        for idx_b in range(1, len(basis)): 
                            temp = np.kron(basis[idx_b][i,:], temp)
                        b_kron[i,:] = temp    
                    btb = b_kron.T @ b_kron
                    alpha_lambda =  np.linalg.solve(btb + total_penalty, 
                                                    b_kron.T) @ z
                    
                    # B kron derivatives
                    basis_deriv: list[npt.NDArray[np.float64]] = []
                    for i, (argvals, n_seg, deg, domain) in enumerate(zip((
                            np.stack((new_x, new_y), axis=1)).T, 
                            n_segments, degree, domains)):
                        if i in dim: 
                            deg = deg - order_derivative
                        b = basis_bsplines(
                            argvals=argvals,
                            n_functions=n_seg + deg,
                            degree=deg,
                            domain_min=domain[0],
                            domain_max=domain[1],
                        ).T
                        basis_deriv.append(b)
        
                    shape_b_kron_deriv = 1
                    for b in basis_deriv: 
                        shape_b_kron_deriv *= b.shape[1]
                    b_kron_deriv = np.zeros((len(new_x), shape_b_kron_deriv))
                    for i in range(len(new_x)): 
                        temp = basis_deriv[0][i,:]
                        for idx_b in range(1, len(basis_deriv)): 
                            temp = np.kron(basis_deriv[idx_b][i,:], temp)
                        b_kron_deriv[i,:] = temp

                    # distance
                    h = 1.0
                    for d in dim: 
                        h *= ((domains[d][1] - domains[d][0]) / 
                            n_segments[d])** int(order_derivative)

                    # Diff derivative
                    diff_list = []
                    for i, b in enumerate(basis): 
                        if i in dim: 
                            diff_list.append(np.diff(np.eye(b.shape[1]), 
                                                     n=order_derivative).T)
                        else: 
                            diff_list.append(np.eye(b.shape[1]))
                    diff_deriv = reduce(np.kron, diff_list)        

                    D_r = (b_kron_deriv @ diff_deriv/h)
                    estim_deriv = D_r @ alpha_lambda
                    
                    error = integrate.trapezoid(integrate.trapezoid((
                                deriv - estim_deriv.reshape(deriv.shape[0], 
                                -1) )**2, new_y, axis=0), new_x) 
                    error_deriv.append({"ratio": idx_ratio, "nb_obs": n, 
                                    "error": error, "borders": border})
            
# Figures EQM 
title_deriv_x = f"EQM_df_dx_2d_risk_{n_segments[0]}_segments"
title_deriv_y = f"EQM_df_dy_2d_risk_{n_segments[0]}_segments"
title_deriv_xy = f"EQM_df_dxy_2d_risk_{n_segments[0]}_segments"

for error, title in zip(
    [error_deriv_x, error_deriv_y, error_deriv_xy], 
    [title_deriv_x, title_deriv_y, title_deriv_xy]):
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

