"""Alternating Direction Method of Multipliers (ADMM) for Lasso.
100% Python Standard Library.
"""

import math

class ADMMLasso:
    """Solves: min 1/2 ||Ax-b||_2^2 + lambda ||z||_1 subject to x - z = 0."""
    @staticmethod
    def soft_threshold(v, kappa):
        return [math.copysign(max(abs(x) - kappa, 0.0), x) for x in v]

    @staticmethod
    def solve(A, b, lam=0.1, rho=1.0, max_iter=100):
        m = len(A)
        n = len(A[0])
        
        x = [0.0] * n
        z = [0.0] * n
        u = [0.0] * n
        
        AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
        Atb = [sum(A[k][i] * b[k] for k in range(m)) for i in range(n)]
        
        M = [[AtA[i][j] + (rho if i == j else 0.0) for j in range(n)] for i in range(n)]
        inv_M = ADMMLasso._invert(M)
        
        for _ in range(max_iter):
            rhs = [Atb[i] + rho * z[i] - u[i] for i in range(n)]
            x = [sum(inv_M[i][j] * rhs[j] for j in range(n)) for i in range(n)]
            
            v = [x[i] + u[i] / rho for i in range(n)]
            z = ADMMLasso.soft_threshold(v, lam / rho)
            
            for i in range(n):
                u[i] += rho * (x[i] - z[i])
                
        return {"x": [round(xi, 4) for xi in x], "z": [round(zi, 4) for zi in z]}

    @staticmethod
    def _invert(matrix):
        n = len(matrix)
        augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
        for i in range(n):
            pivot = augmented[i][i]
            for j in range(2 * n):
                augmented[i][j] /= pivot
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] -= factor * augmented[i][j]
        return [row[n:] for row in augmented]
