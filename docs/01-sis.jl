#===

# SIS reaction diffusion model

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

===#

using DifferentialEquations
using ModelingToolkit
using MethodOfLines
using DomainSets
using Plots
using DisplayAs: PNG

# Setup parameters, variables, and differentuial operators
@parameters t x
@parameters dS dI brn ϵ
@variables S(..) I(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Define functions

γ(x) = x + 1
ratio(x, brn, ϵ) =  brn + ϵ * sinpi(2x)

# 1D PDE and boundary conditions

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

# Define the PDE system

@named pdesys = PDESystem(eqs, bcs, domains,
    [t, x], ## Independent variables
    [S(t, x), I(t, x)],  ## Dependent variables
    [dS => 0.5, dI => 0.1, brn => 3, ϵ => 0.1] ## Initial conditions
)

# Method of lines discretization
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x => dx], t, approx_order=order)

# Convert the PDE system into an ODE problem by discretization
prob = discretize(pdesys, discretization)

# ## Solving time-dependent SIS epidemic model
# The KenCarp4 solver is good for semilinear problem (link this one).
sol = solve(prob, KenCarp4(), saveat=0.2)

# Retrive the results

discrete_x = sol[x]
discrete_t = sol[t]
S_solution = sol[S(t, x)]
I_solution = sol[I(t, x)]

surface(discrete_x, discrete_t, S_solution, xlabel="Location", ylabel="Time") |> PNG

#---
surface(discrete_x, discrete_t, I_solution, xlabel="Location", ylabel="Time") |> PNG

# ## Solving the steady state problem
# `SteadyStateProblem(prob)`

steadystateprob = SteadyStateProblem(prob)
sssol = solve(steadystateprob, DynamicSS(TRBDF2()))
