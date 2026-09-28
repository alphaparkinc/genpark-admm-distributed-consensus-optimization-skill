# ADMM Distributed Consensus Optimization Skill

Alternating Direction Method of Multipliers (ADMM) solver decomposing composite convex optimization into distributed subproblems.

```mermaid
flowchart TD
    Init["Initialize x, z, u"] --> XUpdate["x-update: Argmin Proximal L2 Norm"]
    XUpdate --> ZUpdate["z-update: Proximal L1 Soft-Thresholding S_λ/ρ"]
    ZUpdate --> DualUpdate["u-update: Dual Multiplier Ascent u = u + ρ(x - z)"]
    DualUpdate --> Residual{"Primal & Dual Residuals Small?"}
    Residual -- No --> XUpdate
    Residual -- Yes --> Done["Sparse Optimal Solution z* Returned"]
```

## Features
- **100% Python Standard Library**: No external BLAS or solvers.
- **Proximal Soft-Thresholding**: Generates true exact zeros for feature selection.
- **Consensus Optimization Ready**: Direct scalability to federated agent learning.
