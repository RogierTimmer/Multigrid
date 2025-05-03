using Plots
using LaTeXStrings
using PlotlyJS

include("my_functions.jl")

Plots.plotlyjs();

function exact_solution(Nx, Ny, h)
    u_exact = zeros(Nx * Ny)
    for j in 1:Ny
        for i in 1:Nx
            k = (j - 1) * Nx + i
            x = (i - 1) * h
            y = (j - 1) * h
            u_exact[k] = sin(2π * x) * sin(2π * y)
        end
    end
    return u_exact
end

Niter = 10
error_jacobi = zeros(Niter)
error_gs = zeros(Niter)
GSiter = zeros(Niter)
Jiter = zeros(Niter)

# ---------------- PARAMETERS ----------------

for i in 1:Niter
    Nx = 2^i
    Ny = 2^i
    h = 1.0 / (Nx - 1)  # Grid spacing
    N = Nx * Ny

    A = Laplacian_2D_new(Nx, Ny, h)
    f = rhs_2D(Nx, Ny, h)

    u_jacobi_new, errorAprox_jacobi_new, Jiter[i] = jacobi_solver_new(A, f, Nx, Ny)
    u_gauss_seidel_new, errorAprox_gauss_seidel_new, GSiter[i] = gauss_seidel_solver_new(A, f, Nx)

    u_exact = exact_solution(Nx, Ny, h)
    error_jacobi[i] = norm(u_jacobi_new - u_exact, 2)
    error_gs[i]     = norm(u_gauss_seidel_new - u_exact, 2)

    println("Iteration $i:")
    println("  L2 error (Jacobi):       $(error_jacobi[i])")
    println("  L2 error (Gauss-Seidel): $(error_gs[i])")
    println(" ")

end

grid_sizes = [2^i for i in 1:Niter]

Plots.plot(grid_sizes, Jiter, label="Jacobi iterations", color=:blue, yscale=:log10, xscale=:log10, marker=:circle, markersize=4, linewidth=2)
Plots.plot!(grid_sizes, GSiter, label="Gauss-Seidel iterations", color=:red, yscale=:log10, xscale=:log10, marker=:square, markersize=4, linewidth=2)
Plots.xlabel!("Grid size (Nx * Ny)")
Plots.ylabel!("Number of iterations (log scale)")
Plots.title!("Convergence Comparison")
Plots.plot!(legend=:bottomright)
