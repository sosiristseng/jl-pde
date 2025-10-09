#===
# First Neural ODE example

A neural ODE is an ODE where a neural network defines its derivative function. $\dot{u} = NN(u)$

From: https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/
===#
using Lux, DiffEqFlux, OrdinaryDiffEq, ComponentArrays
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots
rng = Random.Xoshiro(0)

# True solution: $u^3$ and multiplied by a matrix
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

# Generate data from the true function
u0 = [2.0; 0.0]
datasize = 31
tspan = (0.0, 1.5)
tsteps = range(tspan[begin], tspan[end], length = datasize)
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Define a `NeuralODE` problem with a neural network from `Lux.jl`.
dudt2 = Lux.Chain(
    x -> x.^3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)

p, st = Lux.setup(rng, dudt2) |> f64
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

# Predicted output
predict_neuralode(p) = Array(prob_neuralode(u0, p, st)[1])

# Loss function
# Optimization.jl v4 only accept a scalar output
function loss_neuralode(p)
    pred = predict_neuralode(p)
    l2loss = sum(abs2, ode_data .- pred)
    return l2loss
end

# Callback function
anim = Animation()
lossrecord=Float64[]
callback = function (state, l; doplot = true)
    if doplot
        pred = predict_neuralode(state.u)
        plt = scatter(tsteps, ode_data[1,:], label = "data")
        scatter!(plt, tsteps, pred[1,:], label = "prediction")
        frame(anim)
        push!(lossrecord, l)
    else
        println(l)
    end
    return false
end

# Try the callback function to see if it works.
pinit = ComponentArray(p)
callback((; u = pinit), loss_neuralode(pinit); doplot=false)

# Use https://github.com/SciML/Optimization.jl to solve the problem and https://github.com/FluxML/Zygote.jl for automatic differentiation (AD).
adtype = Optimization.AutoZygote()

# Define a [function](https://docs.sciml.ai/Optimization/stable/API/optimization_function/) to optimize with AD.
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

# Define an `OptimizationProblem`
optprob = Optimization.OptimizationProblem(optf, pinit)

# Solve the `OptimizationProblem` using the ADAM optimizer first to get a rough estimate.
result_neuralode = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 300
)

println("Loss is: ", loss_neuralode(result_neuralode.u))

# Use another optimizer (BFGS) to refine the solution.
optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2,
    Optim.BFGS(; initial_stepnorm = 0.01),
    callback = callback,
    allow_f_increases = false
)

println("Loss is: ", loss_neuralode(result_neuralode2.u))

# Visualize the fitting process
mp4(anim, fps=15)

#---
lossrecord
plot(lossrecord[1:300], xlabel="Iters", ylabel="Loss", lab="Adam", yscale=:log10)
plot!(300:length(lossrecord), lossrecord[300:end], lab="BFGS")
