#===
# Multiple Shooting

Docs: https://docs.sciml.ai/DiffEqFlux/dev/examples/multiple_shooting/

In Multiple Shooting, the training data is split into overlapping intervals. The solver (`OptimizationPolyalgorithms.PolyOpt()`) is trained on individual intervals. The results are stiched together.

This simple method assumes no noise in the data. A more robust version can be found at [JuliaSimModelOptimizer.jl](https://help.juliahub.com/jsmo/stable/), which is a  proprietary software.
===#
using Lux
using ComponentArrays
using DiffEqFlux
using DiffEqFlux: group_ranges
using Optimization
using OptimizationPolyalgorithms
using OrdinaryDiffEq
using Plots
using Random
rng = Random.Xoshiro(0)

# Define initial conditions and time steps
datasize = 51
u0 = [2.0, 0.0]
tspan = (0.0, 5.0)
tsteps = range(tspan[begin], tspan[end], length = datasize)

# True values
true_A = [-0.1 2.0; -2.0 -0.1]

# Generate data from the true function: $x^3 * A$
function trueODEfunc!(du, u, p, t)
    du .= ((u.^3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc!, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Define the Neural Network using Lux.jl
# Notice the network is smaller than the first example.
nn = Lux.Chain(
    x -> x.^3,
    Lux.Dense(2, 16, tanh),
    Lux.Dense(16, 2)
)
p_init, st = Lux.setup(rng, nn) |> f64
ps = ComponentArray(p_init)
pd, pax = getdata(ps), getaxes(ps)

# Define the `NeuralODE` problem
neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->nn(u,p,st)[1], u0, tspan, ComponentArray(p_init))

# Animate training process in the callback function
function plot_multiple_shoot(plt, preds, group_size)
	ranges = group_ranges(datasize, group_size)
	for (i, rg) in enumerate(ranges)
		plot!(plt, tsteps[rg], preds[i][1,:], markershape=:circle, label="Group $(i)")
	end
end

anim = Animation()
lossrecord=Float64[]
callback = function (state, l; doplot = true)
    if doplot
        plt = scatter(tsteps, ode_data[1,:], label = "Data")
        plot_multiple_shoot(plt, preds, group_size)
        frame(anim)
        push!(lossrecord, l)
    end
    return false
end

# Parameters for Multiple Shooting
group_size = 3
continuity_term = 200  ## Penalty for discontinuity

function loss_function(data, pred)
    return sum(abs2, data .- pred)
end

l1, preds = multiple_shoot(ps, ode_data, tsteps, prob_node, loss_function, Tsit5(), group_size; continuity_term)

function loss_multiple_shooting(p)
    ps = ComponentArray(p, pax)

    loss, currpred = multiple_shoot(ps, ode_data, tsteps, prob_node, loss_function,
        Tsit5(), group_size; continuity_term)
    global preds = currpred
    return loss
end

# Solve the problem using `OptimizationPolyalgorithms.PolyOpt()`.
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pd)
res_ms = Optimization.solve(optprob, PolyOpt(), callback = callback)

println("Loss is ", loss_multiple_shooting(res_ms.u)[1])

# Visualize the fitting processes
mp4(anim, fps=15)
