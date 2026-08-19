#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns
from pyspline.psplines import PSplines
import pandas as pd
from scipy.optimize import minimize

from pyspline.cv import cv, cv_derivative, gcv

cv_gamma = "gcv"

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
ratios = [0.01, 0.05, 0.1]
nb_obs =[50, 100, 200]
domains = [(0,1), (0,1)]
degree = (3,3)

# Expected values 
new_x = np.linspace(0,1,25)
new_y = np.linspace(0,1,25)
new_x_grid, new_y_grid= np.meshgrid(new_x, new_y)
z_grid = new_x_grid * np.sin(2 * np.pi * new_y_grid) + new_y_grid * np.sin(
        np.pi * new_x_grid)  
new_x_no_border = new_x[1:-1]
new_y_no_border = new_y[1:-1]
new_x_grid_no_border, new_y_grid_no_border= np.meshgrid(new_x_no_border, 
                                                        new_y_no_border)
z_grid_no_border = new_x_grid_no_border * np.sin(
    2 * np.pi * new_y_grid_no_border) + new_y_grid_no_border * np.sin(
        np.pi * new_x_grid_no_border)  

deriv_x = np.sin(2 * np.pi * new_y_grid) + np.pi * new_y_grid * np.cos(
    np.pi * new_x_grid) 
deriv_y =  new_x_grid * 2 * np.pi * np.cos(2 * np.pi * new_y_grid) + np.sin(
    np.pi * new_x_grid)
deriv_xy = 2 * np.pi * np.cos(2 * np.pi * new_y_grid) + np.pi * np.cos(
    np.pi * new_x_grid)

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
for idx_nb_obs, n in enumerate(nb_obs): 
    for idx_ratio, ratio in enumerate(ratios): 
        print(f"nb_obs: {n}, ratio: {ratio}")
        rng = np.random.default_rng(42)

        # Simulate data
        x = rng.uniform(0,1,n)
        y = rng.uniform(0,1,n)
        z_true = x * np.sin(2 * np.pi * y) + y * np.sin(np.pi * x)
        noise = rng.normal(loc=0, scale=np.sqrt(ratio * np.var(z_true)), 
                           size=n)
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
        ps = PSplines(n_segments=n_segments, degree=degree, 
                      penalty=best_penalty, order_penalty=2)
        ps.fit(np.stack((x, y), axis=1), z, domains=domains)

        estim_deriv_x = ps.derivative(np.concatenate((
                        new_x_grid.reshape(-1,1), new_y_grid.reshape(-1,1)), 
                        axis=1), order_derivative=order_derivative, dim=(0,))
        estim_deriv_y = ps.derivative(np.concatenate((
                new_x_grid.reshape(-1,1), new_y_grid.reshape(-1,1)), 
                axis=1), order_derivative=order_derivative, dim=(1,))
        estim_deriv_xy = ps.derivative(np.concatenate((
                        new_x_grid.reshape(-1,1), new_y_grid.reshape(-1,1)), 
                        axis=1), order_derivative=order_derivative, dim=(0,1))
        
        best_penalty_deriv = {}
        match cv_gamma: 
            case "cv": 
                for estim_deriv, dim in zip(
                    [estim_deriv_x, estim_deriv_y, estim_deriv_xy], 
                    [(0,), (1,), (0,1)]
                    ): 
                    cv_deriv = cv_derivative(
                    X=np.concatenate((new_x_grid.reshape(-1,1), 
                                    new_y_grid.reshape(-1,1)), axis=1),
                    deriv = estim_deriv,
                    params=params_deriv,
                    n_segments=n_segments,
                    degree=degree,
                    domains=domains)
                    best_penalty_deriv[dim] = params_deriv[np.argmin(cv_deriv)]
                    print(f"best gamma: {best_penalty_deriv[dim]}")

            case "gcv": 
                for estim_deriv, dim in zip(
                    [estim_deriv_x, estim_deriv_y, estim_deriv_xy], 
                    [(0,), (1,), (0,1)]
                    ): 
                    cv_deriv = minimize(
                        fun=lambda p: gcv(
                            p, 
                            X=np.concatenate((new_x_grid.reshape(-1,1), 
                                    new_y_grid.reshape(-1,1)), axis=1), 
                            deriv = estim_deriv, 
                            n_segments=n_segments,
                            degree=degree,
                            domains=domains, 
                            order_penalty =3), x0= best_penalty)
                    best_penalty_deriv[dim] = cv_deriv.x                    
                    print(f"best gamma: {best_penalty_deriv[dim]}")
        
            case _: 
                best_penalty_deriv[(0,)] = best_penalty
                best_penalty_deriv[(1,)] = best_penalty
                best_penalty_deriv[(0,1)] = best_penalty

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

            # Predict + Validation
            new_z = ps.predict(np.concatenate((new_x_grid.reshape(-1,1), 
                                            new_y_grid.reshape(-1,1)), axis=1))
            new_z = new_z.reshape(new_x_grid.shape[0], new_x_grid.shape[1])
            error = integrate.trapezoid(integrate.trapezoid((
                z_grid - new_z)**2, new_y, axis=0), new_x) 
            error_z.append({"ratio": idx_ratio, "nb_obs": n, "error": error, 
                            "borders": 1})
            
            new_z_no_border = ps.predict(np.concatenate((
                new_x_grid_no_border.reshape(-1,1), 
                new_y_grid_no_border.reshape(-1,1)), axis=1))
            new_z_no_border = new_z_no_border.reshape(
                new_x_grid_no_border.shape[0], new_x_grid_no_border.shape[1])
            error = integrate.trapezoid(integrate.trapezoid((
                z_grid_no_border - new_z_no_border)**2, 
                new_y_no_border, axis=0), new_x_no_border) 
            error_z.append({"ratio": idx_ratio,  "nb_obs": n, "error": error, 
                            "borders": 0})

            # EQM with borders
            for deriv, error_deriv, dim in zip(
                    [deriv_x, deriv_y, deriv_xy], 
                    [error_deriv_x, error_deriv_y, error_deriv_xy], 
                    [(0,), (1,), (0,1)]):

                # Estimate the partial derivative 
                estim_deriv = ps.derivative(np.concatenate((
                    new_x_grid.reshape(-1,1), new_y_grid.reshape(-1,1)), 
                    axis=1), order_derivative=order_derivative, dim=dim)
                
                # Fit model
                ps_deriv = PSplines(n_segments=n_segments, degree=degree, 
                                    penalty=best_penalty_deriv[dim], 
                    order_penalty=2)
                ps_deriv.fit(np.concatenate((new_x_grid.reshape(-1,1), 
                    new_y_grid.reshape(-1,1)), axis=1),  
                    estim_deriv, domains=domains)     
                
                # Compute derivative
                new_estim_deriv = ps_deriv.predict(np.concatenate((
                    new_x_grid.reshape(-1,1), 
                    new_y_grid.reshape(-1,1)), axis=1))
                error = integrate.trapezoid(integrate.trapezoid((
                            deriv - new_estim_deriv.reshape(deriv.shape[0], 
                            -1) )**2, new_y, axis=0), new_x) 
                error_deriv.append({"ratio": idx_ratio, "nb_obs": n, 
                                  "error": error, "borders": 1})
            
            # EQM without borders
            for deriv, error_deriv, dim in zip(
                    [deriv_x_no_border, deriv_y_no_border, deriv_xy_no_border], 
                    [error_deriv_x, error_deriv_y, error_deriv_xy], 
                    [(0,), (1,), (0,1)]):
                
                # Estimate the partial derivative 
                estim_deriv = ps.derivative(np.concatenate((
                    new_x_grid_no_border.reshape(-1,1), 
                    new_y_grid_no_border.reshape(-1,1)), axis=1), 
                    order_derivative=order_derivative, dim=dim)
                
                # Fit model
                ps_deriv = PSplines(n_segments=n_segments, degree=degree, 
                            penalty=best_penalty_deriv[dim], order_penalty=2)
                ps_deriv.fit(np.concatenate((new_x_grid_no_border.reshape(-1,1), 
                    new_y_grid_no_border.reshape(-1,1)), axis=1),  
                    estim_deriv, domains=domains)     
    
                # Compute derivative
                new_estim_deriv = ps_deriv.predict(np.concatenate((
                    new_x_grid_no_border.reshape(-1,1), 
                    new_y_grid_no_border.reshape(-1,1)), axis=1))
                error = integrate.trapezoid(integrate.trapezoid((
                            deriv - new_estim_deriv.reshape(deriv.shape[0], 
                        -1) )**2, new_y_no_border, axis=0), new_x_no_border) 
                error_deriv.append({"ratio": idx_ratio, "nb_obs": n, 
                                  "error": error, "borders": 0})
            
# Figures EQM 
match cv_gamma: 
    case "cv": 
        title_deriv_x = f"EQM_df_dx_2_2d_{n_segments[0]}_segments"
        title_deriv_y = f"EQM_df_dy_2_2d_{n_segments[0]}_segments"
        title_deriv_xy = f"EQM_df_dxy_2_2d_{n_segments[0]}_segments"
    case "gcv": 
        title_deriv_x = f"EQM_df_dx_2_2d_gcv_{n_segments[0]}_segments"
        title_deriv_y = f"EQM_df_dy_2_2d_gcv_{n_segments[0]}_segments"
        title_deriv_xy = f"EQM_df_dxy_2_2d_gcv_{n_segments[0]}_segments"
    case _: 
        title_deriv_x = f"EQM_df_dx_2_2d_no_cv_{n_segments[0]}_segments"
        title_deriv_y = f"EQM_df_dy_2_2d_no_cv_{n_segments[0]}_segments"
        title_deriv_xy = f"EQM_df_dxy_2_2d_no_cv_{n_segments[0]}_segments"

for error, title in zip(
    [error_z, error_deriv_x, error_deriv_y, error_deriv_xy], 
    [f"EQM_f_2d_{n_segments[0]}_segments", title_deriv_x, title_deriv_y, 
     title_deriv_xy]):
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
