#===
# Solving ODEs with NeuralPDE.jl

From https://docs.sciml.ai/NeuralPDE/stable/tutorials/ode/
===#
using NeuralPDE
using Lux
using OptimizationOptimisers
using OrdinaryDiffEq
using LinearAlgebra
using Random
using Plots
rng = Random.Xoshiro(0)

# ## Solve ODEs
# The true function: $u^{\prime} = cos(2 \pi t)$
model(u, p, t) = cospi(2t)

# Prepare data
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(model, u0, tspan)

# Construct a neural network to solve the problem.
chain = Lux.Chain(Lux.Dense(1, 5, σ), Lux.Dense(5, 1))
ps, st = Lux.setup(rng, chain) |> f64

# Solve the ODE as in `DifferentialEquations.jl`, just change the solver algorithm to `NeuralPDE.NNODE()`.
optimizer = OptimizationOptimisers.Adam(0.1)
alg = NeuralPDE.NNODE(chain, optimizer, init_params = ps)
sol = solve(prob, alg, maxiters=2000, saveat = 0.01)

# Comparing to the regular solver
sol2 = solve(prob, Tsit5(), saveat=sol.t)

#---
plot(sol2, label = "Tsit5")
plot!(sol.t, sol.u, label = "NNODE")

# ## Parameter estimation
using NeuralPDE, OrdinaryDiffEq, Lux, Random, OptimizationOptimJL, LineSearches, Plots
rng = Random.Xoshiro(0)

# NNODE only supports out-of-place functions
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
ps, st = Lux.setup(rng, chain) |> f64

# Loss function
additional_loss(phi, θ) = sum(abs2, phi(t_, θ) .- u_) / size(u_, 2)

# NNODE solver
opt = LBFGS(linesearch = BackTracking())
alg = NNODE(chain, opt, ps; strategy = WeightedIntervalTraining([0.7, 0.2, 0.1], 500), param_estim = true, additional_loss)

# Solve the problem
# Use `verbose=true` to see the fitting process
sol = solve(prob, alg, verbose = false, abstol = 1e-8, maxiters = 5000, saveat = t_)

# See the fitted parameters
println(sol.k.u.p)

# Visualize the fit
plot(sol, labels = ["u1_pinn" "u2_pinn"])
plot!(sol_data, labels = ["u1_data" "u2_data"])
