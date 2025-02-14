#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

def _menger_curvature(p1, p2, p3):
    if any([len(p) != 2 for p in [p1, p2, p3]]):
        raise ValueError('Only implemented for 2D Points')
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    matrix = np.array([p2 - p1, p3 - p1])

    dist_12 = np.linalg.norm(p2 - p1)
    dist_23 = np.linalg.norm(p3 - p2)
    dist_31 = np.linalg.norm(p1 - p3)

    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    curvature = (2 * det) / (dist_12 * dist_23 * dist_31) ** 0.5
    return curvature

def inversion(A, b, lam):
    U, s, Vt = svd(A)
    V = Vt.T

    d = s / (s ** 2 + lam ** 2)
    x_reg = V[:, :len(d)] @ np.diag(d) @ (U.T[:len(d), :] @ b)

    # Compute norms
    solution_norm = np.linalg.norm(x_reg)
    residual_norm = np.linalg.norm(A @ x_reg - b)

    return np.array([residual_norm, solution_norm])

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
A = np.array([[3, 2], [2, 3], [1, 1]])
b = np.array([1, 2, 3])

# Regularization parameters
lambdas = np.logspace(-4, 4, 100)

# Storage for norms
solution_norms = []
residual_norms = []

optimal_lambda = golden_section_search(A, b, [1e-4, 1e4], tol=0.01)

# Loop over regularization parameters
for lam in lambdas:
    res_norm, sol_norm = inversion(A, b, lam)

    solution_norms.append(sol_norm)
    residual_norms.append(res_norm)

res_opt, sol_opt = inversion(A, b, optimal_lambda)
# Plot L-curve
plt.figure()
plt.plot(solution_norms, residual_norms, '-o')
plt.plot(sol_opt, res_opt, '-d')
plt.ylabel('Residual Norm ||Ax - b||')
plt.xlabel('Solution Norm ||x||')
plt.title('L-curve')
plt.grid(True)
plt.show()