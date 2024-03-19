# Partial Differential Equations (PDEs)

Solving partial differential equations (PDEs) using https://github.com/SciML/MethodOfLines.jl

## Other PDE packages

- https://github.com/Ferrite-FEM/Ferrite.jl (Finite Element method)
- https://github.com/gridap/Gridap.jl and its [tutorials](https://github.com/gridap/Tutorials)
- https://github.com/trixi-framework/Trixi.jl
- https://github.com/weymouth/WaterLily.jl for fluid dynamics.

## PDE courses

- [Solving partial differential equations in parallel on GPUs](https://github.com/eth-vaw-glaciology/course-101-0250-00)

## Using neural networks to solve differential equations

- Universal Differential Equations (UDEs): https://github.com/SciML/DiffEqFlux.jl
- Physically-informed neural networks (PINNs): https://github.com/SciML/NeuralPDE.jl

`DiffEqFlux` is generally more efficient than `NeuralPDE` because `NeuralPDE` also tries to discover physical rules in the data, as mentioned in [this thread](https://discourse.julialang.org/t/comparisons-between-julia-neuralpde-jl-and-diffeqflux-jl-and-deepxde-python-package/52669).
