# # Bayesian inference for PINNs
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

rng = Random.default_rng()
Random.seed!(rng, 42)

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

# Reference solution for the Lotka-Volterra system
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
    progress = false
)

# Solve the problem
@time sol_pestim = solve(prob, alg; saveat = dt)
sol_pestim.estimated_de_params

# Visualize the fit
plot(time, sol_pestim.ensemblesol[1], label = "estimated x")
plot!(time, sol_pestim.ensemblesol[2], label = "estimated y")
plot!(solution, labels = ["true x" "true y"])
