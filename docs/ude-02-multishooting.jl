#===
# Multiple Shooting

Docs: https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/

In Multiple Shooting, the training data is split into overlapping intervals. The solver (`OptimizationPolyalgorithms.PolyOpt()`) is trained on individual intervals. The results are stiched together.

This simple method assumes no noise in the data. A more robust version can be found at [JuliaSimModelOptimizer.jl](https://help.juliahub.com/jsmo/stable/), which is a  proprietary software.
===#
using ComponentArrays
using DiffEqFlux
using DiffEqFlux: group_ranges
using Lux
using Optimization
using OptimizationPolyalgorithms
using OrdinaryDiffEq
using Plots
using Random
rng = Random.Xoshiro(0)

# Define initial conditions and time steps
datasize = 51
u0 = Float32[2.0, 0.0]
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[begin], tspan[end], length = datasize)

# Generate data from the true function: $x^3 * A$
function trueODEfunc!(du, u, p, t; true_A = Float32[-0.1 2.0; -2.0 -0.1])
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc!, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Define the Neural Network using Lux.jl
nn = Lux.Chain(
    x -> x.^3,
    Lux.Dense(2, 16, tanh),
    Lux.Dense(16, 2)
)
p_init, st = Lux.setup(rng, nn)
ps = ComponentArray(p_init)
pd, pax = getdata(ps), getaxes(ps)

# Define the `NeuralODE` problem
neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->nn(u,p,st)[1], u0, tspan, ComponentArray(p_init))

# Parameters for Multiple Shooting
group_size = 3
continuity_term = 200  ## Penalty for discontinuity

function loss_function(data, pred)
    return sum(abs2, data .- pred)
end

l1, preds = multiple_shoot(ps, ode_data, tsteps, prob_node, loss_function, Tsit5(), group_size; continuity_term)

function loss_multiple_shooting(theta)
    ps = ComponentArray(theta, pax)
    loss, currpred = multiple_shoot(ps, ode_data, tsteps, prob_node, loss_function,
        Tsit5(), group_size; continuity_term)
    return loss
end

# Animate training process in the callback function
function plot_multiple_shoot(plt, preds, group_size)
	ranges = group_ranges(datasize, group_size)
	for (i, rg) in enumerate(ranges)
		plot!(plt, tsteps[rg], preds[i][1,:], markershape=:circle, label="Group $(i)")
	end
end

anim = Animation()
lossrecord=Float64[]
callback = function (state, l; doplot = true, prob_node = prob_node)
    if doplot
        l1, preds = multiple_shoot(
            ComponentArray(state.u, pax), ode_data, tsteps, prob_node, loss_function,
            Tsit5(), group_size; continuity_term)
        plt = scatter(tsteps, ode_data[1,:], label = "Data")
        plot_multiple_shoot(plt, preds, group_size)
        frame(anim)
        push!(lossrecord, l)
    end
    return false
end

# Solve the problem using `OptimizationPolyalgorithms.PolyOpt()`.
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pd)
@time res_ms = Optimization.solve(optprob, PolyOpt(), callback = callback, maxiters = 300)

println("Loss is ", loss_multiple_shooting(res_ms.u)[1])

# Loss over epochs
plot(lossrecord, yscale=:log10, label="Loss", xlabel="Iterations", ylabel="Loss (log10)", title="Loss over iterations")

# Visualize the fitting processes
mp4(anim, fps=15)
