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
using ComponentArrays: ComponentArray
using SimpleUnPack
using OrdinaryDiffEq
using Plots

# Model parameters and initial conditions
function build_u0_ps(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, α=10.0, N=26::Int)
    u0 = ComponentArray(u=zeros(N, N), v=zeros(N, N))
    xx = range(x_min, x_max, length=N)
    yy = range(y_min, y_max, length=N)
    dx = step(xx)
    dy = step(yy)
    for I in CartesianIndices((N, N))
        x = xx[I[1]]
        y = yy[I[2]]
        u0.u[I] = 22 * (y * (1 - y))
        u0.v[I] = 27 * (x * (1 - x))
    end
    ps = (; xx=xx, yy=yy, Dx=α/dx^2, Dy=α/dy^2, N=N)
    return u0, ps
end

# A discretized PDE problem is a coupled ODE problem under the hood.
function model!(ds, s, p, t)
    SimpleUnPack.@unpack xx, yy, Dx, Dy, N = p
    SimpleUnPack.@unpack u, v = s
    brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5
    ## Interating all grid points

    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x = xx[I[1]]
        y = yy[I[2]]
        uterm = 1.0 + v[i, j] * u[i, j]^2 - 4.4 * u[i, j] + brusselator_f(x, y, t)
        vterm = 3.4 * u[i, j] - v[i, j] * u[i, j]^2
        ## No flux boundary conditions
        i_prev = clamp(i - 1, 1, N)
        i_next = clamp(i + 1, 1, N)
        j_prev = clamp(j - 1, 1, N)
        j_next = clamp(j + 1, 1, N)
        unabla = Dx * (u[i, j_prev] - 2u[i, j] + u[i, j_next]) + Dy * (u[i_prev, j] - 2u[i, j] + u[i_next, j])
        vnabla = Dx * (v[i, j_prev] - 2v[i, j] + v[i, j_next]) + Dy * (v[i_prev, j] - 2v[i, j] + v[i_next, j])
        ds.u[i, j] = unabla + uterm
        ds.v[i, j] = vnabla + vterm
    end
    return nothing
end

#---
u0, ps = build_u0_ps(;N=26)
tspan = (0, 11.5)
prob = ODEProblem(model!, u0, tspan, ps)
@time sol = solve(prob, FBDF(), saveat=0.1)
