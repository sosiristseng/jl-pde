#===
# Solving ODEs with NeuralPDE.jl

Solving ODEs with Physics-Informed Neural Networks: https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode/
===#
using NeuralPDE
using Lux
using OptimizationOptimisers
using OrdinaryDiffEq
using LinearAlgebra
using Random
using Plots
rng = Random.default_rng()
Random.seed!(rng, 42)

# ## Solve ODEs
# The true function: $u^{\prime} = cos(2 \pi t)$
model(u, p, t) = cospi(2t)

# Prepare data
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(model, u0, tspan)

# Construct a neural network to solve the problem.
chain = Lux.Chain(Lux.Dense(1, 5, σ), Lux.Dense(5, 1))
ps, st = Lux.setup(rng, chain) |> Lux.f64

# Solve the ODE with `NeuralPDE.NNODE()`.
optimizer = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, optimizer, init_params = ps)
@time sol = solve(prob, alg, maxiters=2000, saveat = 0.01)

# Comparing to the regular solver
sol2 = solve(prob, Tsit5(), saveat=sol.t)

plot(sol2, label = "Tsit5")
plot!(sol.t, sol.u, label = "NNODE")

# ## Parameter estimation
using NeuralPDE
using OrdinaryDiffEq
using Lux
using Random
using OptimizationOptimJL
using LineSearches
using Plots
rng = Random.default_rng()
Random.seed!(rng, 0)

# NNODE only supports out-of-place functions `f(u, p ,t)`
function lv(u, p, t)
    u₁, u₂ = u
    α, β, γ, δ = p
    du₁ = α * u₁ - β * u₁ * u₂
    du₂ = δ * u₁ * u₂ - γ * u₂
    [du₁, du₂]
end

# Generate data
tspan = (0.0, 5.0)
u0 = [5.0, 5.0]
true_p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lv, u0, tspan, true_p)
sol_data = solve(prob, Tsit5(), saveat = 0.01)

t_ = sol_data.t
u_ = Array(sol_data)

# Define a neural network
n = 15
chain = Chain(Dense(1, n, σ), Dense(n, n, σ), Dense(n, n, σ), Dense(n, 2))
ps, st = Lux.setup(rng, chain) |> Lux.f64

# Loss function
additional_loss(phi, θ) = sum(abs2, phi(t_, θ) .- u_) / size(u_, 2)

# NNODE solver
opt = LBFGS(linesearch = BackTracking())
alg = NNODE(chain, opt, ps; strategy = WeightedIntervalTraining([0.7, 0.2, 0.1], 500), param_estim = true, additional_loss)

# Solve the problem
# `verbose=true` for the fitting process
@time sol = solve(prob, alg, verbose = true, abstol = 1e-8, maxiters = 5000, saveat = t_)

# See the fitted parameters
println(sol.k.u.p)

# Visualize the fit
plot(sol, labels = ["u1_pinn" "u2_pinn"])
plot!(sol_data, labels = ["u1_data" "u2_data"])

# ## Bayesian inference for PINNs
# https://docs.sciml.ai/NeuralPDE/stable/tutorials/Lotka_Volterra_BPINNs/
using NeuralPDE
using AdvancedHMC
using MCMCChains
using LogDensityProblems
using Lux
using Plots
using OrdinaryDiffEq
using Distributions
using Random

# NNODE only supports out-of-place functions `f(u, p ,t)`
function lotka_volterra(u, p, t)
    ## Model parameters.
    α, β, γ, δ = p
    ## Current state.
    x, y = u

    ## Evaluate differential equations.
    dx = (α - β * y) * x ## prey
    dy = (δ * x - γ) * y ## predator

    return [dx, dy]
end

#---
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 4.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)
dt = 0.01
solution = solve(prob, Tsit5(); saveat = dt)

# Dataset creation for parameter estimation (plus 30% noise)
time = solution.t
u = hcat(solution.u...)
x = u[1, :] + (u[1, :]) .* (0.3 .* randn(length(u[1, :])))
y = u[2, :] + (u[2, :]) .* (0.3 .* randn(length(u[2, :])))
dataset = [x, y, time]

## Plotting the data which will be used
plot(time, x, label = "noisy x")
plot!(time, y, label = "noisy y")
plot!(solution, labels = ["x" "y"])

# Define a PINN neural network. The input is time, and the output is the state of the system (x and y).
chain = Chain(Dense(1, 6, tanh), Dense(6, 6, tanh), Dense(6, 2))

# Use `BNNODE` for Bayesian inference. The parameters of the model are estimated with the dataset, and the uncertainty of the estimation is quantified with the posterior distribution.
alg = BNNODE(chain;
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.0, 3.0),
    param = [
        Normal(1, 2),
        Normal(2, 2),
        Normal(2, 2),
        Normal(0, 2)],
    progress = false)

# Solve the problem
@time sol_pestim = solve(prob, alg; saveat = dt)
sol_pestim.estimated_de_params

#---
plot(time, sol_pestim.ensemblesol[1], label = "estimated x")
plot!(time, sol_pestim.ensemblesol[2], label = "estimated y")
plot!(solution, labels = ["true x" "true y"])
