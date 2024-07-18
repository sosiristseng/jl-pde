md"""
# Symbolic Brusselator PDE

[Source](https://docs.sciml.ai/MethodOfLines/stable/tutorials/brusselator/)

The Brusselator PDE:

$$
\begin{align}
\frac{\partial u}{\partial t} &= 1 + u^2v - 4.4u + \alpha (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) + f(x, y, t) \\
\frac{\partial v}{\partial t} &= 3.4u - u^2 v + \alpha (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2})
\end{align}
$$

where

$$
f(x, y, t) =
\begin{cases}
5 \qquad \text{if} (x - 0.3)^2 + (y - 0.6)^2 \leq 0.1^2 \ \text{and} \  t \geq 1.1  \\
0 \qquad \text{otherwise}
\end{cases}
$$

and the initial conditions are

$$
\begin{align}
u(x, y, 0) &= 22(y(1-y))^{1.5} \\
v(x, y, 0) &= 27(x(1-x))^{1.5}
\end{align}
$$

with the periodic boundary condition

$$
\begin{align}
u(x+1, y, 0) &= u(x, y, t)  \\
u(x, y+1, 0) &= u(x, y, t)
\end{align}
$$

on a time span of $t \in [0, 11.5]$.
"""

using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using DomainSets
using Plots

# Setup parameters, variables, and differential operators
@parameters x y t
@variables u(..) v(..)

Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
∇²(u) = Dxx(u) + Dyy(u)

# Dynamics on each grid point
brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5
α = 10.0

u0(x, y, t) = 22(y * (1 - y))^(3 / 2)
v0(x, y, t) = 27(x * (1 - x))^(3 / 2)

# PDEs
eqs = [
    Dt(u(x, y, t)) ~ 1.0 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) + α * ∇²(u(x, y, t)) + brusselator_f(x, y, t),
    Dt(v(x, y, t)) ~ 3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 + α * ∇²(v(x, y, t))
]

# Space and time domains
domains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    t ∈ Interval(t_min, t_max)
]

# Periodic boundary conditions
bcs = [
    u(x, y, 0) ~ u0(x, y, 0),
    u(0, y, t) ~ u(1, y, t),
    u(x, 0, t) ~ u(x, 1, t),
    v(x, y, 0) ~ v0(x, y, 0),
    v(0, y, t) ~ v(1, y, t),
    v(x, 0, t) ~ v(x, 1, t)
]

# PDE system
@named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])

# Discretization to an ODE system
discretization = let N = 16, order = 2
    MOLFiniteDifference([x=>N, y=>N], t, approx_order=order)
end
prob = discretize(pdesys, discretization)

# Solvers: https://diffeq.sciml.ai/stable/solvers/ode_solve/
@time sol = solve(prob, KenCarp4(), saveat=0.1)

# Extract data
discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]

solu = sol[u(x, y, t)]
solv = sol[v(x, y, t)]

umax = maximum(maximum, solu)
vmax = maximum(maximum, solu)

# ## Visualization
# Interval == `2:end` since in periodic condition, end == 1

anim = @animate for k in eachindex(discrete_t)
    heatmap(solu[2:end, 2:end, k], title="u @ t=$(discrete_t[k])", clims = (0.0, 4.2))
end

mp4(anim, fps = 8)

#---
anim = @animate for k in eachindex(discrete_t)
    heatmap(solv[2:end, 2:end, k], title="v @ t=$(discrete_t[k])", clims = (0.0, 4.2))
end

mp4(anim, fps = 8)
