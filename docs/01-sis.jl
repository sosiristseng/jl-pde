md"""
# 1D PDE: SIS diffusion model

[Source](https://docs.sciml.ai/MethodOfLines/stable/tutorials/sispde/)

$$
\begin{align}
\frac{\partial S}{\partial t} &= d_S S_{xx} - \beta(x)\frac{SI}{S+I} + \gamma(x)I  \\
\frac{\partial I}{\partial t} &= d_I I_{xx} + \beta(x)\frac{SI}{S+I} - \gamma(x)I
\end{align}
$$

where $x \in (0, 1)$

Solve the steady-state problem $\frac{\partial S}{\partial t} = \frac{\partial I}{\partial t} = 0$

The boundary condition: $\frac{\partial S}{\partial x} = \frac{\partial I}{\partial x} = 0$ for x = 0, 1

The conservative relationship: $\int^{1}_{0} (S(x) + I(x) ) dx = 1$

Notations:

- $x$ : location
- $t$ : time
- $S(x, t)$ : the density of susceptible populations
- $I(x, t)$ : the density of infected populations
- $d_S$ / $d_I$ : the diffusion coefficients for susceptible and infected individuals
- $\beta(x)$ : transmission rates
- $\gamma(x)$ : recovery rates
"""
using OrdinaryDiffEq
using ModelingToolkit
using MethodOfLines
using DomainSets
using Plots

# Setup parameters, variables, and differential operators
@parameters t x
@parameters dS dI brn ϵ
@variables S(..) I(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Helper functions
γ(x) = x + 1
ratio(x, brn, ϵ) =  brn + ϵ * sinpi(2x)

# 1D PDE for disease spreading
eqs = [
    Dt(S(t, x)) ~ dS * Dxx(S(t, x)) - ratio(x, brn, ϵ) * γ(x) * S(t, x) * I(t, x) / (S(t, x) + I(t, x)) + γ(x) * I(t, x),
    Dt(I(t, x)) ~ dI * Dxx(I(t, x)) + ratio(x, brn, ϵ) * γ(x) * S(t, x) * I(t, x) / (S(t, x) + I(t, x)) - γ(x) * I(t, x)
]

# Boundary conditions
bcs = [
    S(0, x) ~ 0.9 + 0.1 * sinpi(2x),
    I(0, x) ~ 0.1 + 0.1 * cospi(2x),
    Dx(S(t, 0)) ~ 0.0,
    Dx(S(t, 1)) ~ 0.0,
    Dx(I(t, 0)) ~ 0.0,
    Dx(I(t, 1)) ~ 0.0
]

# Space and time domains
domains = [
    t ∈ Interval(0.0, 10.0),
    x ∈ Interval(0.0, 1.0)
]

# Build the PDE system
@named pdesys = PDESystem(eqs, bcs, domains,
    [t, x], ## Independent variables
    [S(t, x), I(t, x)],  ## Dependent variables
    [dS => 0.5, dI => 0.1, brn => 3, ϵ => 0.1] ## Initial conditions
)

# Finite difference method (FDM) converts the PDE system into an ODE problem
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x => dx], t, approx_order=order)
prob = discretize(pdesys, discretization)

# ## Solving time-dependent SIS epidemic model
# `KenCarp4` is good at solving semilinear problems (like this example).
sol = solve(prob, KenCarp4(), saveat=0.2)

# Grid points
discrete_x = sol[x]
discrete_t = sol[t]

# Results (Matrices)
S_solution = sol[S(t, x)]
I_solution = sol[I(t, x)]

# Visualize the solution
surface(discrete_x, discrete_t, S_solution, xlabel="Location", ylabel="Time", title="Susceptible")

#---
surface(discrete_x, discrete_t, I_solution, xlabel="Location", ylabel="Time", title="Infectious")
