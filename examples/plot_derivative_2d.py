#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import integrate

from pyspline.psplines import PSplines
from pyspline.cv import cv 

# Set RNG
rng = np.random.default_rng(42)

# Simulate data
n = 1000
x = rng.uniform(0,1,n)
y = rng.uniform(0,1,n)
noise = rng.normal(loc=0, scale=0.05, size=n)
z = x * np.sin(y) + y * np.sin(x) + noise 

# CV
loocv = cv(
    X=np.stack((x, y), axis=1),
    y=z,
    params=[(0.1, 10),(1,10),(10,10),
             (0.1, 1),(1,1),(10,1), 
             (0.1, 0.1),(1,0.1),(10, 0.1)],
    n_segments=(10,10),
    degree=(3,3),
    order_penalty= 2,
    domains=[(0,1), (0,1)])

loocv = loocv.reshape(3,3)
print(loocv)
fig, ax = plt.subplots()
im = ax.imshow(loocv, cmap='viridis')

ax.set_xticks(range(loocv.shape[0]), labels=[-1,0,1])
ax.set_yticks(range(loocv.shape[1]),labels=[1, 0, -1])
ax.set_xlabel(r"$log_{10}(\lambda_1)$")
ax.set_ylabel(r"$log_{10}(\lambda_2)$")

for i in range(loocv.shape[0]):
    for j in range(loocv.shape[1]):
        text = ax.text(j, i, f"{loocv[i, j]:.3f}",
                       ha="center", va="center", color="w")
fig.tight_layout()
plt.show()

# Fit the model
ps = PSplines(n_segments=(50,50), degree=(3,3), penalty=(10,10), 
              order_penalty=2)
ps.fit(np.stack((x, y), axis=1), z, domains=[(0,1), (0,1)])
new_x = np.linspace(0,1,1000)
new_y = np.linspace(0,1,1000)
new_x_grid, new_y_grid= np.meshgrid(new_x, new_y)
new_z = ps.predict(np.concatenate((new_x_grid.reshape(-1,1), 
                                   new_y_grid.reshape(-1,1)), axis=1))

# Validation 
z_grid = new_x_grid * np.sin(new_y_grid) + new_y_grid * np.sin(new_x_grid)
new_z = new_z.reshape(new_x_grid.shape[0], new_x_grid.shape[1])
error_z = integrate.trapezoid(integrate.trapezoid((z_grid - new_z)**2, new_y, 
                                                  axis=0), new_x) 
print(f"Error on f(x,y): {error_z}")

# Partial derivatives 
deriv_x = np.sin(new_y_grid) + new_y_grid * np.cos(new_x_grid)
deriv_y = np.sin(new_x_grid) + new_x_grid * np.cos(new_y_grid)
deriv_xy = np.cos(new_y_grid) + np.cos(new_x_grid)

# Estimate the partial derivative df(x,y)/dx
order_derivative = 1

estim_deriv_x = ps.derivative(np.concatenate((new_x_grid.reshape(-1,1), 
                                new_y_grid.reshape(-1,1)), axis=1), 
                                order_derivative=order_derivative, dim=(0,))
error_deriv_x = integrate.trapezoid(integrate.trapezoid((
                deriv_x - estim_deriv_x.reshape(deriv_x.shape[0], -1) )**2, 
                new_y, axis=0), new_x) 
print(f"Error on df(x,y)/dx: {error_deriv_x}")

# Estimate the partial derivative df(x,y)/dy
estim_deriv_y = ps.derivative(np.concatenate((new_x_grid.reshape(-1,1), 
                                new_y_grid.reshape(-1,1)), axis=1),
                                order_derivative=order_derivative, dim=(1,))
error_deriv_y = integrate.trapezoid(integrate.trapezoid((
                deriv_y - estim_deriv_y.reshape(deriv_y.shape[0], -1) )**2, 
                new_y, axis=0), new_x) 
print(f"Error on df(x,y)/dy: {error_deriv_y}")

# Estimate the partial derivative df(x,y)/dxdy
estim_deriv_xy = ps.derivative(np.concatenate((new_x_grid.reshape(-1,1), 
                                new_y_grid.reshape(-1,1)), axis=1), 
                                order_derivative=order_derivative, dim=(0,1))
error_deriv_xy = integrate.trapezoid(integrate.trapezoid((
                deriv_xy - estim_deriv_xy.reshape(deriv_xy.shape[0], -1) )**2, 
                new_y, axis=0), new_x) 
print(f"Error on df(x,y)/dxdy: {error_deriv_xy}")

# Build the graph f(x,y)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(new_x, new_y, z_grid, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title('f(x,y)', fontsize=16)
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(new_x, new_y, new_z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(r'$\hat{f}$(x,y)', fontsize=16)
plt.show()

# Build the graph df(x,y)/dx
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(new_x_grid, new_y_grid, deriv_x, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(r"$\frac{\partial f(x,y)}{\partial x}$", fontsize=16)
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(new_x_grid, new_y_grid, 
            estim_deriv_x.reshape(new_x_grid.shape[0], new_x_grid.shape[1]), 
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(r"$\frac{\partial \hat{f}(x,y)}{\partial x}$", fontsize=16)
plt.show()

# Build the graph df(x,y)/dy
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(new_x_grid, new_y_grid, deriv_y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(r"$\frac{\partial f(x,y)}{\partial y}$", fontsize=16)
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(new_x_grid, new_y_grid, 
            estim_deriv_y.reshape(new_x_grid.shape[0], new_x_grid.shape[1]), 
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(r"$\frac{\partial \hat{f}(x,y)}{\partial y}$", fontsize=16)
plt.show()

# Build the graph df(x,y)/dxdy
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(new_x_grid, new_y_grid, deriv_xy, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(r"$\frac{\partial f(x,y)}{\partial x \partial y}$", fontsize=16)
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(new_x_grid, new_y_grid, 
            estim_deriv_xy.reshape(new_x_grid.shape[0], new_x_grid.shape[1]), 
            cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title(r"$\frac{\partial \hat{f}(x,y)}{\partial x \partial y}$")
plt.show()
