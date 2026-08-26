# # Parameter estimation
#
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
