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
import ComponentArrays as CA
using SimpleUnPack
using OrdinaryDiffEq
using LinearSolve
using Plots

function build_brussilator(; N::Int=32, x_min=0, y_min=0, t_min=0, x_max=1, y_max=1, t_max=11.5, α = 10.0)
    @assert (x_min < x_max) && (y_min < y_max) && (t_min < t_max) && (N > 1)
    u0 = CA.ComponentArray(u = zeros(N, N), v = zeros(N, N))
    xx = range(x_min, x_max, length=N)
    yy = range(y_min, y_max, length=N)
    dx = step(xx)
    dy = step(yy)
    ps = CA.ComponentArray(; α, xx, yy, N, dx, dy)

    for i in 1:N, j in 1:N
        x = xx[j]
        y = yy[i]
        u0.u[i, j] = 22 * (y * (1 - y))
        u0.v[i, j] = 27 * (x * (1 - x))
    end

    odef = (ds, s, p, t) -> begin
        @unpack α, xx, yy, dx, dy = p
        @unpack u, v = s
        brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5
        for (i, y) in enumerate(yy), (j, x) in enumerate(xx)
            uterm = 1.0 + v[i, j] * u[i, j]^2 - 4.4 * u[i, j] + brusselator_f(x, y, t)
            vterm = 3.4 * u[i, j] - v[i, j] * u[i, j]^2
            ## No flux boundary conditions
            i_prev = clamp(i - 1, 1, N)
            i_next = clamp(i + 1, 1, N)
            j_prev = clamp(j - 1, 1, N)
            j_next = clamp(j + 1, 1, N)
            unabla = (u[i, j_prev] - 2u[i, j] + u[i, j_next]) / dx^2 + (u[i_prev, j] - 2u[i, j] + u[i_next, j]) / dy^2
            vnabla = (v[i, j_prev] - 2v[i, j] + v[i, j_next]) / dx^2 + (v[i_prev, j] - 2v[i, j] + v[i_next, j]) / dy^2
            ds.u[i, j] = α * unabla + uterm
            ds.v[i, j] = α * vnabla + vterm
        end
    end
    return (; u0, odef, ps)
end

#---
@unpack u0, odef, ps = build_brussilator()
tspan = (0.0, 11.5)
prob = ODEProblem(odef, u0, tspan, ps)
@time sol = solve(prob, KenCarp47(linsolve = KrylovJL_GMRES()), saveat=0.1)
