#===
# Solving PDEs with ModelingToolkit and NeuralPDE

Solving Poisson PDE Systems (https://docs.sciml.ai/NeuralPDE/stable/tutorials/pdesystem/)

$$
\partial^{2}_{x}u(x,y) + \partial^{2}_{y}u(x,y) = -\sin (\pi x) \sin (\pi y)
$$

with boundary conditions

$$
\begin{align}
u(0, y) &= 0 \\
u(1, y) &= 0 \\
u(x, 0) &= 0 \\
u(x, 1) &= 0 \\
\end{align}
$$

where

$x ∈ [0, 1], y ∈ [0, 1]$
===#
using NeuralPDE
using Lux
using Optimization
using OptimizationOptimJL
using ModelingToolkit
using DomainSets
using LineSearches
using Plots

# 2D PDE
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
eq  = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

# Boundary conditions
bcs = [
    u(0, y) ~ 0.0,
    u(1, y) ~ 0.0,
    u(x, 0) ~ 0.0,
    u(x, 1) ~ 0.0
]

# Space domains
domains = [
    x ∈ DomainSets.Interval(0.0, 1.0),
    y ∈ DomainSets.Interval(0.0, 1.0)
]

# Build a neural network for the PDE solver.
# Input: 2 dimensions.
# Hidden layers: 16 neurons * 2 layers.
# Output: single output `u(x, y)`
dim = 2
chain = Lux.Chain(Dense(dim, 16, Lux.σ), Dense(16, 16, Lux.σ), Dense(16, 1))

# Discretization method uses`PhysicsInformedNN()` (PINN).
dx = 0.05
discretization = PhysicsInformedNN(chain, QuadratureTraining(; batch = 200, abstol = 1e-6, reltol = 1e-6))

# Build the PDE system and discretize it.
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = discretize(pde_system, discretization)

# Callback function to record the loss
lossrecord = Float64[]
callback = function (p, l)
    push!(lossrecord, l)
    return false
end

# Solve the problem. It may take a long time.
opt = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking())
@time res = Optimization.solve(prob, opt, callback = callback, maxiters=1000)

#---
plot(lossrecord, xlabel="Iters", yscale=:log10, ylabel="Loss", lab=false)

# Plot the predicted solution of the PDE and compare it with the analytical solution to see the relative error.
xs, ys = [DomainSets.infimum(d.domain):dx/10:DomainSets.supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sinpi(x)*sinpi(y))/(2pi^2)

phi = discretization.phi
u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys], (length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
diff_u = abs.(u_predict .- u_real)

#---
p1 = plot(xs, ys, u_real, linetype=:contourf, title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf, title = "predicted");
p3 = plot(xs, ys, diff_u, linetype=:contourf, title = "error");
plot(p1, p2, p3)
