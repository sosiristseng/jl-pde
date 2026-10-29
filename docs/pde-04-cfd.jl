#===
# A simple Navier-Stokes solver

Source: https://www.juliabloggers.com/cfd-tutorial-in-julia/ by Emad Masroor

This is a solver for the two-dimensional unsteady viscous incompressible Navier-Stokes equations in $\omega-\psi$ formulation on a rectangular Cartesian grid. We discretize the domain using second-order centered finite differences, and march the governing equations forward in time implicitly. 

All linear systems are solved by using either a naive Gauss-Siedel relaxation scheme or the native Julia matrix-inversion operator `\` on a `SparseArray`.

We test the solver on the lid-driven cavity problem and compare results with Ghia & Ghia's solution.

## Governing equations

The governing equations are as follows:

$$
\frac{\partial \omega}{\partial t} + (\nabla^{\bot} \psi )\cdot \nabla \omega = \frac{1}{Re}\nabla^2 \omega \\ \nabla^2 \psi = -\omega
$$

The first equation is a parabolic-hyperbolic equation for $\omega$ where the advection velocity is given through the streamfunction, which is defined as follows:

$$
\boldsymbol{u} =\nabla^{\bot} \psi \equiv \left( \frac{\partial \psi}{\partial y}, -\frac{\partial \psi}{\partial x}\right) \equiv (u,v)
$$

The second equation is an elliptic equation for $\psi$, specifically Poisson's equation with the vorticity as the source term. This equation proceeds directly from the definition of $\omega$ and $\psi	$:

$$
\omega \equiv \nabla \times \boldsymbol{u} = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} = -\frac{\partial^2 \psi }{\partial x^2} - \frac{\partial^2 \psi }{\partial y^2} \equiv -\nabla^2 \psi
$$

## Boundary conditions

In a rectangular domain $\Omega$, we have eight boundary conditions on $\psi$:

Four Dirichlet boundary conditions: $\psi = 0 $ on the entire $\partial \Omega$ which ensures that the walls are a single streamline, i.e. that the wall-normal velocities are zero.

Four Neumann boundary coniditions: $\frac{\partial \psi}{\partial n} = u_t$, the wall-tangential velocity, on each subset of $\partial \Omega$

For $\omega$, no explicit boundary conditions are given. Indeed, the vorticity at the wall is actually a crucial unknown in the Navier-Stokes equations with boundaries, since all vorticity in a fluid must have been first generated at boundaries.

### Thom's Formula

To derive implicit boundary conditions on the vorticity, let us write a Taylor expansion for the stream function at a point adjacent to a wall, with the subscript 'a' representing the wall-adjacent point and the subscript b representing the point at the wall. 'n' is a coordinate representing the wall-normal direction, and $\Delta n$ is the spatial discretization in the direction normal to the wall.

$$
\psi_a \approx \psi_b + \underbrace{\frac{\partial \psi}{\partial n}|_{b}}_{u_t} \frac{\Delta n}{1!} + \underbrace{\frac{\partial^2 \psi}{\partial n^2}|_{b}}_{-\omega_b} \frac{\Delta n^2}{2!} + ... 
$$

The normal derivative of $\psi$ at a wall is simply the wall-tangential velocity. The second spatial derivative of $\psi$ at a wall is (what is left of) the Laplacian of $\psi$ at the wall, which by definition equals negative of the vorticity. Hence, we now have a relation between the wall vorticity $\omega_b$, the wall-tangential velocity $u_t$, and the value of the stream function:

$$
\psi_a \approx \psi_b + u_t \Delta n - \omega_b \frac{\Delta n^2}{2}
\\
\omega_b \approx 2 \left[ \frac{\psi_b - \psi_a}{\Delta n^2} + \frac{u_t}{\Delta n}\right]
$$

In practice, we will use Dirichlet boundary conditions on $\psi$ and Thom's boundary conditions on $\omega$.
===#

#===
## Packages
===#

#---
using LinearAlgebra
using IterativeSolvers
using SparseArrays
using Plots
using LaTeXStrings

#---
function VorticityBoundaryConditions!(ω, ψ, Δx, Δy, un, us, ve, vw)
    ω[:, end] .= 2*((ψ[:, end] - ψ[:, end-1])/(Δx^2) .- ve/Δx)
    ω[:, 1] .= 2*((ψ[:, 1] - ψ[:, 2])/(Δx^2) .- vw/Δx)
    ω[end, :] .= 2*((ψ[end, :] - ψ[end-1, :])/(Δy^2) .+ us/Δy)
    ω[1, :] .= 2*((ψ[1, :] - ψ[2, :])/(Δy^2) .+ un/Δy)
end

#===
## Linear solvers

In theory, of course, any matrix-inverting technique can be used with any equation of the form $Ax=b$. Here, we will use the native Julia matrix-inversion operator `\` (or the conjugate gradient algorithm `cg!` from [`IterativeSovlers.jl`](https://iterativesolvers.julialinearalgebra.org/stable/) ) for the Poisson equation for $\psi$ because the boundary conditions for that equation are easy to implement, and they don't change at each time step. For the advection-diffusion equation for $\omega$, however, we will use the Gauss-Siedel technique. This equation is by far the easier one to solve, so the computational penalty of a naive solver like Gauss-Siedel is not very high.

### Solving sparse $A x = b$ with Gauss-Siedel

Consider a system of linear equations of the form $Ax=b$. The vector x represents an unknown quantity on the entire grid, and is arranged in the following form:

$$
\begin{bmatrix}            x_{11} \\            x_{12} \\            \vdots \\            x_{1N} \\            x_{21} \\            x_{22} \\            x_{2N} \\            \vdots \\            x_{N1} \\            x_{N2} \\            x_{NN}          \end{bmatrix}
$$
A is a sparse, pentadiagonal matrix with (at most) five non-zero terms. These terms are the coefficients of the $x_{ij}$'th point as well as its neighbors to the north, south, east and west. Thus, using 'N,S,E,W' to represent the neighboring points and 'p' to represent current point, the general form of any row of this system of equations is as follows:

$$
a_p x_p + a_N x_N + a_S x_S + a_E x_E + a_W x_W = b_p \\ \implies a_p x_p + \sum_{NSEW}a_i x_i = b_p
$$
In the Gauss-Siedel method, we make a new guess for $x^{n+1}$ based on the current guess, $x^n$ using the following procedure:

$$
res = b_p - (a_p x_p + \sum_{NSEW}a_i x_i) \\ \Delta x = \frac{res}{a_p} \\ x_p^{n+1} = x^n_p + \Delta x 
$$
this is repeated until the residual falls below a small $\epsilon$.

### Over-relaxation

The Gauss-Siedel algorithm can be significantly accelerated by adding an *over-relaxation* parameter $\lambda$. It can be added on to the end of each Gauss-Siedel iteration in the following way:

$$
x^{n+1} = \lambda (x^n + \Delta x) + (1-\lambda)x^n 
$$
this essentially 'weights' the new value between the predicted value and the previous value. When $\lambda = 1$, this reverts back to the usual Gauss-Siedel algorithm. As $\lambda \rightarrow 2$, this weighs the new value more heavily toward the predicted value. One rule of thumb for the over-relaxation parameter is $\lambda = 2 - \frac{1}{N-1}$.

In practice, we have found that over-relaxation only makes sense for solving the elliptic Poisson equation for $\psi$.
===#

#---
function GaussSiedel!(ϕ, Ap, An, As, Ae, Aw, Rp, res; λ=1, maxiter=1000)
    normRes = 1
    k = 0
    Ny, Nx = size(ϕ)
    while normRes >= 1e-8 && k < maxiter
        k += 1
        for i in 2:(Ny-1)
            for j in 2:(Nx-1)
                ϕP = ϕ[i, j]
                ϕE = ϕ[i+0, j+1]
                ϕW = ϕ[i+0, j-1]
                ϕN = ϕ[i-1, j+0]
                ϕS = ϕ[i+1, j+0]
                res[i, j] = Rp[i, j] - (Ap*ϕP
                    + An*ϕN
                    + As*ϕS
                    + Ae*ϕE
                    + Aw*ϕW)
                Δϕ = res[i, j]/Ap
                ϕ[i, j] = λ*(ϕ[i, j] + Δϕ) + (1-λ)*ϕ[i, j]
            end
        end
        normRes = norm(res)
    end
    return k
end

#===
## Solving sparse $Ax=b$ with `\` or `cg!`

In principle, Julia provides very simple syntax for matrix-inversion: `A\b` should be all we need. However, because we will be storing all variables as 2-D arrays, we need to first unwrap `x` and the right-hand side into a 1-D array, apply the matrix-inversion, and then wrap the updated `x` back into a 2-D array.
===#

#---
function LinearSolve!(A, x, b)
    ## Solves the equation Ax = b assuming zero Dirichlet BCs everywhere
    Ny, Nx = size(b)
    Ny, Nx = Ny-2, Nx-2
    x_int = x[2:(end-1), 2:(end-1)]
    b_int = b[2:(end-1), 2:(end-1)]
    b_vec = reshape(b_int, Ny*Nx)
    ## x_int = A\b_vec
    x_vec = reshape(x_int, Ny*Nx)
    cg!(x_vec, A, b_vec, log=true)
    x[2:(end-1), 2:(end-1)] .= reshape(x_int, (Ny, Nx))
end

#===
## Discrete system of equations for $\omega$ and $\psi$

### Poisson equation for $\psi$

The equation for the stream function is already a Poisson equation, which is linear. It is straightforward to cast it in the form `Ax = b` using finite differences:

$$
\nabla^2 \psi = \frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = -\omega \\ D_{xx} \psi + D_{yy} \psi = -\omega \\ \implies \frac{\psi_{i,j+1} - 2 \psi_{i,j} + \psi_{i,j-1}}{\Delta x^2} + \frac{\psi_{i+1,j} - 2 \psi_{i,j} + \psi_{i-1,j}}{\Delta y^2} = -\omega_{i,j}
$$
This only needs to be done once. We write a function which returns the 2-dimensional Laplacian using Julia's `SparseArray` type: 
===#

#---
function BuildPoissonMatrix(Ny, Nx, Δx, Δy)
    ## This function returns a (Ny*Nx) × (Ny*Nx) matrix in the form of
    ## a sparse array, corresponding to the discrete 2D Laplacian operator.
    Ny = Ny-2
    Nx = Nx-2

    Isx = [1:Ny; 1:(Ny-1); 2:Ny]
    Jsx = [1:Ny; 2:Ny; 1:(Ny-1)]

    Isy = [1:Nx; 1:(Nx-1); 2:Nx]
    Jsy = [1:Nx; 2:Nx; 1:(Nx-1)]

    Vsx = [fill(-2, Ny); fill(1, 2Ny-2)]
    Vsy = [fill(-2, Nx); fill(1, 2Nx-2)]
    D²x = sparse(Isx, Jsx, Vsx)
    D²y = sparse(Isy, Jsy, Vsy)
    ## D_xx = 1/(Δx^2) .* kron(sparse(I,Nx,Nx), D²x)
    ## D_yy = 1/(Δy^2) .* kron(D²y, sparse(I,Ny,Ny))
    D_yy = 1/(Δy^2) .* kron(sparse(I, Nx, Nx), D²x)
    D_xx = 1/(Δx^2) .* kron(D²y, sparse(I, Ny, Ny))
    Lap = D_xx + D_yy
end

#===
## Evolution equation for $\omega$

$$
\frac{\partial \omega}{\partial t} + \boldsymbol u \cdot \nabla \omega = \frac{1}{Re}\nabla^2 \omega
$$
We treat the parabolic part (i.e. the diffusion term) of this equation implicitly, but the hyperbolic part (i.e. the advection term) explicitly. This is because if we were to treat the term term $\boldsymbol u \cdot \nabla \omega$ implicitly with a central-difference scheme, we would get a non-diagonally-dominant matrix, which is not guaranteed to converge using the iterative matrix-solving techniques. If an upwind scheme is used, we can then treat the advection term implicitly as well. 

We can write a discrete version of the evolution equation for $\omega$ as follows:

$$
\frac{\omega^{n+1}-\omega^n}{\Delta t} + (D_y \psi^n D_x \omega^n -D_x \psi^n D_y \omega^n) = Re^{-1}(D_{xx}\omega^{n+1}+D_{yy}\omega^{n+1})
$$
where the superscript n denotes the value at the current (known) timestep, and the superscript n+1 denotes the value at the future (unknown) timestep. The diffusion term is treated implicitly, hence the n+1 there, while the advection term is treated explicitly, hence the n there. The time-derivative term has been treated fully implicitly with a first-order backwards Euler scheme, i.e. $\dot{\omega}^{n+1} \approx \frac{\omega^{n+1}-\omega^{n}}{\Delta t}$. Collecting the unknown terms on the left-hand side and the known terms on the right-hand side, we get:

$$
\left[ \Delta t ^{-1} \boldsymbol 1  - Re^{-1}D_{xx} - Re^{-1}D_{yy} \right] \omega^{n+1} = - \left[ D_y \psi^n D_x -D_x \psi^n D_y + \Delta t^{-1}\boldsymbol 1 \right] \omega^n
$$
The above is also, of course, a system of linear equations of the form $Ax=b$ and its diagonal dominance is guaranteed. Hence, it too can be solved using iterative methods. We build the matrix (in practice, only a set of coefficients, since we will solve this particular equation using the Gauss-Siedel technique) once, at the beginning:
===#

#---
function BuildAdvectionDiffusionCoefficients(Re, Δt, Δx, Δy)
    ## Time-derivative
    ap = 1/Δt
    ## Diffusion
    ap += 2/(Re*Δx^2) + 2/(Re*Δy^2)
    an = -1/(Re*Δy^2)
    aw = -1/(Re*Δx^2)
    as = -1/(Re*Δy^2)
    ae = -1/(Re*Δx^2)
    return ap, an, as, ae, aw
end

#===
On the other hand, the right-hand side of this equation will evidently be different at each time step, since the explicit term $\boldsymbol u \cdot \nabla \omega$ changes at every step. The following function, therefore, will be called at each time step:
===#

#---
function BuildAdvectionDiffusionRHS!(Rp, ϕ, ψ, Δt, Δx, Δy, Ny, Nx, Re)
    ## Time derivative
    Rp .= ϕ/Δt

    ## Diffusion term (fully implicit)

    ## Convection term
    for i in 2:(Ny-1)
        for j in 2:(Nx-1)
            ϕE = ϕ[i+0, j+1]
            ϕW = ϕ[i+0, j-1]
            ϕN = ϕ[i-1, j+0]
            ϕS = ϕ[i+1, j+0]
            ψE = ψ[i+0, j+1]
            ψW = ψ[i+0, j-1]
            ψN = ψ[i-1, j+0]
            ψS = ψ[i+1, j+0]

            u = (ψN - ψS)/(2Δy)
            v = -(ψE - ψW)/(2Δx)
            ∂ϕ∂y = (ϕN - ϕS)/(2Δy)
            ∂ϕ∂x = (ϕE - ϕW)/(2Δx)

            Rp[i, j] += - (u*∂ϕ∂x + v*∂ϕ∂y)
            ## Rp[i,j] += (ψE - ψW)/(2Δx) * (ϕN - ϕS)/(2Δy) -
            ##            (ψN - ψS)/(2Δy) * (ϕE - ϕW)/(2Δx)
        end
    end
end

#===
It is straightforward to replace the first-order backwards Euler time-stepping scheme with a second-order backwards Euler scheme. The only difference is that an additional set of $\omega$'s needs to be stored, and the time-derivative terms in the matrix as well as the RHS need to be slightly modified. The second-order backward scheme looks like this:

$$
 \dot{\omega}^{n+1} \approx \frac{1.5 \omega^{n+1} - 2 \omega^n + 0.5 \omega^{n-1}}{\Delta t}
$$
thus, we would simply need to replace `Rp .= ϕ/Δt` with `Rp .= 2ϕ/Δt - ϕold/(2Δt)` inside the function `BuildAdvectionDiffusionRHS!`, and replace `ap = 1/Δt` with `3/(2Δt)` inside the function `BuildAdvectionDiffusionCoefficients`. 

## Code utilities

### Record changes

In Julia, functions can be broadcast to multiple arguments. Hence, we only need a generic recording function:
===#

#---
function RecordHistory!(ϕ, ϕ_old, ϕ_hist)
    Δϕ = norm(ϕ - ϕ_old)
    ϕ_old .= ϕ
    push!(ϕ_hist, Δϕ)
    return (Δϕ)
end

#===
### Solution struct and associated functions

We create a struct (essentially, a new type) representing a solution. The solver's output will be assigned to a new instance of this struct. We also create some methods associated with this object type:
===#

#---
struct Results
    ψ::Array
    ω::Array
    hist::Array
    x::Array
    y::Array
    tfinal
    steps
    Re
end

ShowStreamlines(sol::Results) = contour(sol.x, sol.y, reverse(reverse(sol.ψ, dims=1), dims=2),
    aspectratio=1, framestyle=:box,
    xlims=(sol.x[1], sol.x[end]),
    ylims=(sol.y[1], sol.y[end]),
    legend=:none, grid=:none)

#===
## Assemble code

### User-input parameters

The above functions will be assembled into a function called `LidDrivenCavity()`, which accepts a number of keyword arguments. These are all optional, since there are default values associated with them.

* `tfinal = Inf`, the final time of the simulation. If not set, it will run till steady-state.
* `Lx=1`, length of $\Omega	$ in the horizontal direction
* `Ly=1`, length of $\Omega$ in the vertical direction
* `CFL=0.5`, the Courant-Fredericks-Levy number
* `Nx = 65`, the number of discretization points in the horizontal direction
* `Ny = 65`, the number of discretization points in the horizontal direction
* `u_n,u_s,v_w,v_e = (1,0,0,0)`, the tangential velocities at each wall (north, south, east, west)
* `printfreq`, prints output every this number of steps
* `Re=100`, the Reynolds number

### Complete function
===#

#---
function LidDrivenCavity(;
    tfinal=Inf,
    Lx=1, Ly=1, CFL=0.5, Re=100,
    Nx=65, Ny=65,
    u_n=1, u_s=0, v_w=0, v_e=0,
    printfreq=10)
    t0 = time() # begin timing
    println("------------------Ny = $(Ny), Nx = $(Nx) ---------------")
    Δy = Ly/(Ny-1)
    Δx = Lx/(Nx-1)
    x = 0:Δx:Lx
    y = 0:Δy:Ly
    Δt = CFL * Δx

    ## Construct matrix for Poisson equation
    A_poisson = BuildPoissonMatrix(Ny, Nx, Δx, Δy) # for coNxgrad
    ## Construct matrix for advection-diffusion equation
    ap, an, as, ae, aw = BuildAdvectionDiffusionCoefficients(Re, Δt, Δx, Δy)
    ## allocate empty matrices for Gauss-Siedel solver
    Rp = zeros(Ny, Nx)
    res = zeros(Ny, Nx)

    ## initialize ω and ψ
    ω = zeros(Ny, Nx)
    ψ = zeros(Ny, Nx)

    ## keep track of changes
    ω_old = zeros(Ny, Nx)
    ψ_old = zeros(Ny, Nx)
    ω_hist = []
    ψ_hist = []
    residual = 1

    ######### Begin time-stepping #########
    k0, t = 0, 0
    while t < tfinal && maximum(residual) > 1e-8
        t += Δt
        k0 += 1

        ## Solve Poisson equation for ψ:
        LinearSolve!(A_poisson, ψ, -ω)

        ## Determine boundary conditions on ω using Thom's formula
        VorticityBoundaryConditions!(ω, ψ, Δx, Δy, u_n, u_s, v_e, v_w)

        ## Modify the explicit part of advection-diffusion equation
        BuildAdvectionDiffusionRHS!(Rp, ω, ψ, Δt, Δx, Δy, Ny, Nx, Re)

        ## Solve advection-diffusion equation for ω:
        GaussSiedel!(ω, ap, an, as, ae, aw, Rp, res)

        ## Record changes
        residual = RecordHistory!.([ω, ψ], [ω_old, ψ_old], [ω_hist, ψ_hist])

        ## Print to terminal
        if (k0 % printfreq == 0)
            println("Step: $k0 \t Time: $(round(t,digits=3))\t",
                "|Δω|: $(round((residual[1]),digits=8)) \t",
                "|Δψ|: $(round((residual[2]),digits=8)) \t")
        end
    end
    tt = round(time() - t0, digits=3) # end timing
    println("This took $tt seconds.")
    println("--------------------------------------------------------")
    ## Create a struct containing the results
    Results(ψ, ω, hcat(ω_hist, ψ_hist), x, y, t, k0, Re)
end

#===
## Solutions for the Lid-Driven Cavity

### Classic test cast at Re = 100
===#

#---
sol1 = LidDrivenCavity()

ShowStreamlines(sol1)

#===
This looks good. let's take a look at the convergence history:
===#

#---
plot(log10.(sol1.hist),labels=["|Δω|" "|Δψ|"])

#===
and also, compare it with Ghia and Ghia's results:
===#

#---
using DelimitedFiles, Plots, LaTeXStrings
Nx, Ny, Lx, Ly = 65, 65, 1, 1
ψ1 = sol1.ψ
uref_along_y = readdlm("data/pde-04-reference_u.txt", skipstart=1)[:, 2:3]
vref_along_x = readdlm("data/pde-04-reference_v.txt", skipstart=1)[:, 2:3]
Ny, Nx = size(ψ1)
Δy = Ly/(Ny-1)
Δx = Lx/(Nx-1)
u1 = diff(ψ1[:, Int((end-1)/2)]) ./ Δy
y1 = reverse(range(Δy/2, 1-Δy/2, step=Δy))
v1 = -diff(ψ1[Int((end-1)/2), :]) ./ Δx
x1 = reverse(range(Δx/2, 1-Δx/2, step=Δx))
plot(y1, u1, markershape=:circle, color=:blue,
    label=L"u(x=0.5,y)", legendfont=font(14),
    framestyle=:box)
plot!(uref_along_y[:, 1], uref_along_y[:, 2],
    markershape=:square, color=:blue,
    label="Ghia and Ghia")
plot!(v1 .+ 0.5, x1, markershape=:circle, color=:red,
    label=L"v(x,y=0.5)")
plot!(vref_along_x[:, 2] .+ 0.5, vref_along_x[:, 1],
    label="Ghia and Ghia",
    markershape=:square, color=:red, yticks=:none, xticks=:none, legend=:left)

#===
## Two symmetric gyres, Re = 250

* `Lx=2`
* `v_w=1`
* `v_e=-1`
* `Re=250`
* `tfinal=10`
===#

#---
sol2 = LidDrivenCavity(Lx=2, v_w=1, v_e=-1, u_n=0, Re=250, tfinal=10);

ShowStreamlines(sol2)

#===
## Orthogonal velocities

* `u_n=1`
* `v_e=-1`
* `Re=250`
* `Ly=1.4`
===#

#---
sol3 = LidDrivenCavity(u_n=1, v_e=-1, Re=250, Ly=1.4);

ShowStreamlines(sol3)