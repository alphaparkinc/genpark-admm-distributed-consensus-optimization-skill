"""Example demonstrating ADMM Lasso sparse recovery."""
from client import ADMMLasso

def main():
    A = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    b = [2.0, 0.0, 2.0]
    res = ADMMLasso.solve(A, b, lam=0.5)
    print("ADMM Sparse Regression Results:")
    print("  Primal x:", res["x"])
    print("  Sparse z:", res["z"])

if __name__ == "__main__":
    main()
