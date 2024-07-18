md"""
# 2D heat equation

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
using Plots

# Setup variables and differential operators
@variables t x y
@variables u(..)

Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# PDE equation
eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

# Boundary conditions
bcs = [
    u(0, x, y) ~ 0,
    u(t, 0, y) ~ x * y,
    u(t, 1, y) ~ x * y,
    u(t, x, 0) ~ x * y,
    u(t, x, 1) ~ x * y,
]

# Space and time domains
domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)
]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

# Discretize the PDE system into an ODE system
N = 20
discretization = MOLFiniteDifference([x=>N, y=>N], t, approx_order=2, grid_align=MethodOfLines.EdgeAlignedGrid())
prob = discretize(pdesys, discretization)

# Solve the PDE
sol = solve(prob, KenCarp4(), saveat=0.01)

# Extract data
discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]
solu = sol[u(t, x, y)]

# Animate the solution
anim = @animate for k in eachindex(discrete_t)
    heatmap(solu[k, 2:end-1, 2:end-1], title="u @ t=$(discrete_t[k])", aspect_ratio=:equal)
end

mp4(anim, fps = 8)
