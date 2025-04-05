#===
# Creating PDEs from ODEs

Modeling the Brusselator PDE:

$$
\begin{align}
\frac{\partial u}{\partial t} &= 1 + u^2v - 4.4u + \alpha (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) + f(x, y, t) \\
\frac{\partial v}{\partial t} &= 3.4u - u^2 v + \alpha (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2})
\end{align}
$$

===#
using ModelingToolkit
using OrdinaryDiffEq
using Plots

function build_brussilator(; N::Int=32, x_min=0, y_min=0, t_min=0, x_max=1, y_max=1, t_max=11.5, name=:sys, simplify=true)
    @assert (x_min < x_max) && (y_min < y_max) && (t_min < t_max) && (N > 1)
    @independent_variables t
    @parameters α = 10.0
    @variables u(t)[1:N, 1:N] v(t)[1:N, 1:N]
    Dt = Differential(t)
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    ## Dynamics on each grid point
    brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5
    _u0(x, y) = 22 * (y * (1 - y))
    _v0(x, y) = 27 * (x * (1 - x))
    defaults = Dict()
    eqs = []
    for i in 1:N, j in 1:N
        x = (j - 1) * dx
        y = (i - 1) * dy
        uterm = 1.0 + v[i, j] * u[i, j]^2 - 4.4 * u[i, j] + brusselator_f(x, y, t)
        vterm = 3.4 * u[i, j] - v[i, j] * u[i, j]^2
        ## No flux boundary conditions
        i_prev = ifelse(i == 1, i, i - 1)
        i_next = ifelse(i == N, i, i + 1)
        j_prev = ifelse(j == 1, j, j - 1)
        j_next = ifelse(j == N, j, j + 1)
        udiff = (u[i, j_prev] - 2u[i, j] + u[i, j_next]) / dx^2 + (u[i_prev, j] - 2u[i, j] + u[i_next, j]) / dy^2
        vdiff = (v[i, j_prev] - 2v[i, j] + v[i, j_next]) / dx^2 + (v[i_prev, j] - 2v[i, j] + v[i_next, j]) / dy^2
        append!(eqs, [Dt(u[i, j]) ~ α * udiff + uterm, Dt(v[i, j]) ~ α * vdiff + vterm])
        defaults[u[i, j]] = _u0(x, y)
        defaults[v[i, j]] = _v0(x, y)
    end
    sys = ODESystem(eqs, t; name, defaults)
    if simplify
        sys = structural_simplify(sys)
    end
    return sys
end

#---
@time "Build system" sys = build_brussilator(N=20, simplify=false) |> complete
@time "Build ODE problem" prob = ODEProblem(sys, [], 11.5; sparse=true, jac=true)
@time "Solve the problem" sol = solve(prob, KenCarp47(), saveat=0.1)
