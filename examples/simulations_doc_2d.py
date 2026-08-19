#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns
from pyspline.psplines import PSplines
import pandas as pd
from scipy.optimize import minimize
from functools import reduce

from pyspline.basis import basis_bsplines
from pyspline.cv import (cv, cv_derivative, gcv, risk, 
                         kronecker_product, penalties)

ratio=0.05

# Parameters 
param_1 = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
param_2 = [100, 50, 10, 5, 1, 0.5, 0.1, 0.01,  0.05, 0.001]
params = [(i, j) for j in param_2 for i in param_1]

param_1_deriv = [0.00001, 0.0001, 0.001,0.01, 0.1]
param_2_deriv = [0.1, 0.01,.001,0.0001, 0.00001]
params_deriv = [(i, j) for j in param_2_deriv for i in param_1_deriv]

nb_simu = 100
order_derivative = 1
n_segments = (10,10)
# ratios = [0.01, 0.05, 0.1]
nb_obs =[50, 100, 200]
domains = [(0,1), (0,1)]
degree = (3,3)
dims = [(0,), (1,), (0,1)]
order_penalty=2
METHODS = {"cv1": {"type": "cv", "penalty": 0}, 
           "cv2": {"type": "cv", "penalty": 1}, 
           "gcv1": {"type": "gcv", "penalty": 0}, 
           "gcv2": {"type": "gcv", "penalty": 1}, 
           "risk": {"type": "risk", "penalty": 2}}

# Expected values 
new_x_border = np.linspace(0,1,25)
new_y_border = np.linspace(0,1,25)
new_x_grid_border, new_y_grid_border = np.meshgrid(new_x_border, new_y_border)
z_grid = new_x_grid_border * np.sin(2 * np.pi * new_y_grid_border
        ) + new_y_grid_border * np.sin(np.pi * new_x_grid_border)  
deriv_x = (np.sin(2 * np.pi * new_y_grid_border) 
           + np.pi * new_y_grid_border * np.cos(np.pi * new_x_grid_border))
deriv_y =  (new_x_grid_border * 2 * np.pi * np.cos(
    2 * np.pi * new_y_grid_border) 
    + np.sin(np.pi * new_x_grid_border))
deriv_xy = 2 * np.pi * np.cos(2 * np.pi * new_y_grid_border) + np.pi * np.cos(
    np.pi * new_x_grid_border)

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
GRIDS = {
    1: (
        new_x_border,
        new_y_border,
        new_x_grid_border,
        new_y_grid_border
        ),
    0: (
        new_x_no_border,
        new_y_no_border,
        new_x_grid_no_border,
        new_y_grid_no_border
        ),
        }

# EQM
error_z = []
error_deriv_x = []
error_deriv_y = []
error_deriv_xy = []
for idx_nb_obs, n in enumerate(nb_obs): 
    for method, config in METHODS.items(): 
        print(f"nb_obs: {n}, method: {method}")
        rng = np.random.default_rng(42)

        # Simulate data
        x = rng.uniform(0,1,n)
        y = rng.uniform(0,1,n)
        z_true = x * np.sin(2 * np.pi * y) + y * np.sin(np.pi * x)
        variance = ratio * np.var(z_true)
        noise = rng.normal(loc=0, scale=np.sqrt(variance), size=n)
        z = z_true + noise 
        
        # CV
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

        # CV derivative 
        # First derivative estimation
        if config["penalty"] == 0 : 
            ps = PSplines(n_segments=n_segments, degree=degree, 
                                    penalty=(0,0), order_penalty=2)
            ps.fit(np.stack((x, y), axis=1), z, domains=domains)
        elif config["penalty"] == 1:
            ps = PSplines(n_segments=n_segments, degree=degree, 
                                    penalty=best_penalty, order_penalty=2)
            ps.fit(np.stack((x, y), axis=1), z, domains=domains)

        # Hyperparameter selection for derivative estimation    
        best_penalty_deriv = {}
        for dim in dims: 
            if config["type"] == "cv": 
                estim_deriv = ps.derivative(np.concatenate((
                        new_x_grid_border.reshape(-1,1), 
                        new_y_grid_border.reshape(-1,1)), 
                        axis=1), order_derivative=order_derivative, dim=dim)
                cv_deriv = cv_derivative(
                X=np.concatenate((new_x_grid_border.reshape(-1,1), 
                                new_y_grid_border.reshape(-1,1)), axis=1),
                deriv = estim_deriv,
                params=params_deriv,
                n_segments=n_segments,
                degree=degree,
                domains=domains)
                best_penalty_deriv[dim] = params_deriv[np.argmin(cv_deriv)]

            elif config["type"] == "gcv": 
                estim_deriv = ps.derivative(np.concatenate((
                        new_x_grid_border.reshape(-1,1), 
                        new_y_grid_border.reshape(-1,1)), 
                        axis=1), order_derivative=order_derivative, dim=dim)
                cv_deriv = minimize(
                    fun=lambda p: gcv(
                        p, 
                        X=np.concatenate((new_x_grid_border.reshape(-1,1), 
                                new_y_grid_border.reshape(-1,1)), axis=1), 
                        deriv = estim_deriv, 
                        n_segments=n_segments,
                        degree=degree,
                        domains=domains, 
                        order_penalty =order_penalty), x0= best_penalty)
                best_penalty_deriv[dim] = cv_deriv.x                    
    
            elif config["type"] == "risk": 
                cv_deriv = minimize(
                    fun=lambda p: risk(
                        p, 
                        X=np.stack((x, y), axis=1), 
                        y = z, 
                        n_segments=n_segments,
                        degree=degree,
                        domains=domains, 
                        order_penalty =order_penalty, 
                        variance=variance,
                        order_derivative=1, 
                        dim=dim), 
                    x0= best_penalty)
                best_penalty_deriv[dim] = cv_deriv.x   

            else: 
                print("ERROR")

        for idx_simu in range(nb_simu): 
            rng = np.random.default_rng(2*idx_simu)
            x = rng.uniform(0,1,n)
            y = rng.uniform(0,1,n)
            z_true = x * np.sin(2 * np.pi * y) + y * np.sin(np.pi * x)
            noise = rng.normal(loc=0, scale=np.sqrt(ratio * np.var(z_true)), 
                               size=n)
            z = z_true + noise 

            # Fit the model
            ps = PSplines(n_segments=n_segments, degree=degree, 
                          penalty=best_penalty, 
                        order_penalty=2)
            ps.fit(np.stack((x, y), axis=1), z, domains=domains)

            # EQM derivatives
            models = {}
            for dim in dims: 
                ps_deriv = PSplines(n_segments=n_segments, degree=degree, 
                            penalty=best_penalty_deriv[dim], order_penalty=2)
                if config["penalty"] == 0: 
                    ps_deriv.fit(np.stack((x, y), axis=1), z, domains=domains)
                elif config["penalty"] == 1: 
                    estim_deriv = ps.derivative(np.concatenate((
                        new_x_grid_border.reshape(-1,1), 
                        new_y_grid_border.reshape(-1,1)), 
                        axis=1), order_derivative=order_derivative, dim=dim)
                    ps_deriv.fit(np.concatenate((
                        new_x_grid_border.reshape(-1,1), 
                        new_y_grid_border.reshape(-1,1)), axis=1),  
                        estim_deriv, domains=domains)     
                models[dim] = ps_deriv

            for border, derivs in (
                (1, [deriv_x, deriv_y, deriv_xy]),
                (0, [deriv_x_no_border, deriv_y_no_border, deriv_xy_no_border]),
                ):
                new_x, new_y, new_x_grid, new_y_grid = GRIDS[border] 

                for deriv, error_deriv, dim in zip(
                        derivs, 
                        [error_deriv_x, error_deriv_y, error_deriv_xy], 
                        dims):

                    if config["penalty"] == 0: 
                        ps_deriv = models[dim]
                        estim_deriv = ps_deriv.derivative(np.concatenate((
                            new_x_grid.reshape(-1,1), 
                            new_y_grid.reshape(-1,1)), axis=1), 
                            order_derivative=order_derivative, dim=dim)
                            
                    elif config["penalty"] == 1: 
                        ps_deriv = models[dim]
                        estim_deriv = ps_deriv.predict(np.concatenate((
                            new_x_grid.reshape(-1,1), 
                            new_y_grid.reshape(-1,1)), axis=1))

                    elif config["penalty"] == 2: 
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
                        total_penalty = penalties(basis, order_penalty, 
                                                  best_penalty_deriv[dim])
                        small_penalty = penalties(basis, order_penalty, 
                                            tuple(np.repeat(1e-03, len(basis))))

                        # Kronecker product
                        b_kron = kronecker_product(basis)
                        btb = b_kron.T @ b_kron
                        alpha_lambda =  np.linalg.solve(btb + total_penalty, 
                                                        b_kron.T) @ z
                        
                        # B kron derivatives
                        basis_deriv = []
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
            
                        b_kron_deriv = kronecker_product(basis_deriv)

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
                        deriv - estim_deriv.reshape(deriv.shape[0], -1) )**2, 
                        new_y, axis=0), new_x) 
                    error_deriv.append({"method": method, "nb_obs": n, 
                                    "error": error, "borders": border})
                
# Figures EQM 
new_ratio = str(ratio).replace(".", "_")
title_deriv_x = f"EQM_df_dx_2d_all_ratio_{new_ratio}_{n_segments[0]}_segments"
title_deriv_y = f"EQM_df_dy_2d_all_ratio_{new_ratio}_{n_segments[0]}_segments"
title_deriv_xy = f"EQM_df_dxy_2d_all_ratio_{new_ratio}_{n_segments[0]}_segments"

for error, title in zip(
    [error_deriv_x, error_deriv_y, error_deriv_xy], 
    [title_deriv_x, title_deriv_y, title_deriv_xy]):
    error = pd.DataFrame(error)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 13))

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
    ax2.set_xticks([0, 1, 2, 3, 4], 
                ['cv 1', 'cv 2', 'gcv 1', 'gcv 2', 'risque']) 
    handles, _ = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=new_labels)
    ax2.set_xlabel("Méthode")
    fig.savefig(title)
    plt.tight_layout()
    plt.show()
