md"""
# 2D time-independent heat equation

From the tutorial
+ https://docs.sciml.ai/MethodOfLines/stable/tutorials/heatss/
+ https://docs.sciml.ai/MethodOfLines/stable/tutorials/heat/

Using `MethodOfLines.jl` (https://github.com/SciML/MethodOfLines.jl/) to symbolically define the PDE system and use the [finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method) (FDM) to solve the following PDE:

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
$$
"""
using ModelingToolkit
using MethodOfLines
using DomainSets
using OrdinaryDiffEq
using NonlinearSolve
using Plots

# Setup variables and differential operators
function heat_equation(;dx=0.1)
    @independent_variables x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    bcs = [
        u(0, y) ~ x * y,
        u(1, y) ~ x * y,
        u(x, 0) ~ x * y,
        u(x, 1) ~ x * y
    ]
    domains = [ x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    ## Discretize the PDE system into an Nonlinear system
    ## Pass `nothing` to the time parameter
    discretization = MOLFiniteDifference([x=>dx, y=>dx], nothing, approx_order=2, grid_align = MethodOfLines.EdgeAlignedGrid())
    prob = discretize(pdesys, discretization)
    return (; prob, x, y, u)
end

@time prob, x, y, u = heat_equation()

# Solve the PDE
@time sol = NonlinearSolve.solve(prob, NewtonRaphson())

# Extract data
discrete_x = sol[x]
discrete_y = sol[y]
u_sol = sol[u(x,y)]

# ## Visualize the solution
heatmap(
    discrete_x, discrete_y, u_sol,
    xlabel="x values", ylabel="y values", aspect_ratio=:equal,
    title="Steady State Heat Equation", xlims=(0, 1), ylims=(0, 1)
)
