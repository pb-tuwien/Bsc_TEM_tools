#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.interpolate import CubicSpline

def _menger_curvature(p1, p2, p3):
    if any([len(p) != 2 for p in [p1, p2, p3]]):
        raise ValueError('Only implemented for 2D Points')
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    matrix = np.array([p2 - p1, p3 - p2])

    dist_12 = np.linalg.norm(p2 - p1)
    dist_23 = np.linalg.norm(p3 - p2)
    dist_31 = np.linalg.norm(p1 - p3)

    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    curvature = (2 * np.abs(det)) / (dist_12 * dist_23 * dist_31) ** 0.5
    return curvature

def inversion(A, b, lam):
    U, s, Vt = svd(A)
    V = Vt.T

    d = s / (s ** 2 + lam ** 2)
    x_reg = V[:, :len(d)] @ np.diag(d) @ (U.T[:len(d), :] @ b)

    # Compute norms
    solution_norm = np.linalg.norm(x_reg)
    residual_norm = np.linalg.norm(A @ x_reg - b)

    return np.array([solution_norm, residual_norm])

def golden_section_search(A, b, interval, tol=0.1, maxIter=50):
    phi = (1 + 5 ** 0.5) / 2

    lam1, lam4 = interval
    lam2 = 10**((np.log10(lam4) + phi*np.log10(lam1)) / (1 + phi))
    lam3 = 10**(np.log10(lam1) + np.log10(lam4) - np.log10(lam2))

    p1 = inversion(A, b, lam1)
    p2 = inversion(A, b, lam2)
    p3 = inversion(A, b, lam3)
    p4 = inversion(A, b, lam4)

    opt_lambda = None

    for i in range(maxIter):
        if (lam4 - lam1)/lam4 < tol:
            break
        else:
            curve2 = _menger_curvature(p1, p2, p3)
            curve3 = _menger_curvature(p2, p3, p4)

            if curve2 > curve3:
                opt_lambda = lam2
                lam4 = lam3
                lam3 = lam2
                p4 = p3
                p3 = p2
                lam2 = 10**((np.log10(lam4) + phi*np.log10(lam1)) / (1 + phi))
                p2 = inversion(A, b, lam2)
            else:
                opt_lambda = lam3
                lam1 = lam2
                lam2 = lam3
                p1 = p2
                p2 = p3
                lam3 = 10**(np.log10(lam1) + np.log10(lam4) - np.log10(lam2))
                p3 = inversion(A, b, lam3)

    return opt_lambda

# Example matrix A and vector b
A = np.array([[3, 4], [6, 3], [1, 1]])
b = np.array([1, 2, 3])

# Regularization parameters
lambdas = np.logspace(-4, 4, 100)
# lambdas = np.linspace(1e-4, 1e4, 1000)

# Storage for norms
points = []

optimal_lambda = golden_section_search(A, b, [1e-4, 1e4], tol=0.01)

# Loop over regularization parameters
for lam in lambdas:
    point = inversion(A, b, lam)
    points.append(point)


points = np.array(points)
roughness_values = points.T[0]
rms_values = points.T[1]
lambda_values = np.array(lambdas)

# first_derivative_grad = np.gradient(rms_values, roughness_values)
# second_derivative_grad = np.gradient(first_derivative_grad, roughness_values)
# curvature_values_grad = second_derivative_grad / (1 + first_derivative_grad ** 2) ** (3 / 2)
# max_curvature_index_grad = np.argmax(curvature_values_grad)
# opt_lambda_grad = lambda_values[max_curvature_index_grad]
#
# first_derivative_diff = np.diff(rms_values) / np.diff(roughness_values)
# second_derivative_diff = np.diff(first_derivative_diff) / np.diff(roughness_values[:-1])
# curvature_values_diff = second_derivative_diff / (1 + first_derivative_diff[:-1] ** 2) ** (3 / 2)
# max_curvature_index_diff = np.argmax(curvature_values_diff)
# opt_lambda_diff = lambda_values[max_curvature_index_diff]


sol_opt, res_opt = inversion(A, b, optimal_lambda)
# res_obt_grad, sol_opt_grad = inversion(A, b, opt_lambda_grad)
# res_obt_diff, sol_opt_diff = inversion(A, b, opt_lambda_diff)

# Plot L-curve
fig, ax = plt.subplots(1,2, figsize=(10, 5))

ax[0].plot(roughness_values, rms_values, '-o')
ax[0].plot(sol_opt, res_opt, 'd', label='Golden Section Search')
# ax[0].plot(sol_opt_grad, res_obt_grad, 's', label='Gradient Curvature')
# ax[0].plot(sol_opt_diff, res_obt_diff, 'x', label='Difference Curvature')
ax[0].set_ylabel('Residual Norm ||Ax - b||')
ax[0].set_xlabel('Solution Norm ||x||')
ax[0].set_title('L-curve')
ax[0].grid(True)
ax[0].legend()

# ax[1].plot(roughness_values[:-2], curvature_values_diff, '.-', label='diff')
# ax[1].plot(roughness_values, curvature_values_grad, '--', label='grad')
# ax[1].set_ylabel('Curvature')
# ax[1].set_xlabel('Solution Norm ||x||')
# ax[1].grid(True)
# ax[1].legend()

fig.show()