using Plots
using LaTeXStrings
using PlotlyJS

include("my_functions1.jl")

Plots.plotlyjs()

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

# Multigrid parameters
i = 11  # Level parameter: 2^i + 1 grid points per direction
Nx = 2^i + 1
Ny = 2^i + 1
h = 1.0 / (Nx - 1)
N = Nx * Ny

# System setup
A = Laplacian_2D_new(Nx, Ny, h)
f = rhs_2D(Nx, Ny, h)

# Initial smoothing
uh, errorAprox, iter, residual = gauss_seidel_solver_new(A, f, Nx; tol=1e-5, max_iter=3)

# Course grid setup
A_coarse = Laplacian_2D_new(Nx ÷ 2 +1, Ny ÷ 2 +1, h * 2)

# Restrict the residual to the coarse grid
function restrict_injection(u_fine::Vector{Float64}, Nx_fine::Int, Ny_fine::Int)
    Nx_coarse = div(Nx_fine + 1, 2)
    Ny_coarse = div(Ny_fine + 1, 2)
    u_coarse = zeros(Float64, Nx_coarse * Ny_coarse)

    for J in 1:Ny_coarse
        for I in 1:Nx_coarse
            k_coarse = (J - 1) * Nx_coarse + I
            i_fine = 2 * (I - 1) + 1
            j_fine = 2 * (J - 1) + 1
            k_fine = (j_fine - 1) * Nx_fine + i_fine
            u_coarse[k_coarse] = u_fine[k_fine]
        end
    end
    return u_coarse
end

residual_coarse = restrict_injection(residual, Nx, Ny)

uH, errorAprox, iter, residual = gauss_seidel_solver_new(A_coarse, residual_coarse, ceil(Nx ÷ 2+1); tol=1e-5, max_iter=300)

U = reshape(uH, Nx ÷ 2 + 1, Ny ÷ 2 + 1)
Ufine = reshape(uh, Nx, Ny)

p1 = contourf(U, title="Coarse Grid Solution", xlabel="x", ylabel="y", color=:viridis, aspect_ratio=1)
p2 = contourf(Ufine, title="Fine Grid Solution", xlabel="x", ylabel="y", color=:viridis, aspect_ratio=1)

Plots.plot(p1, p2, layout=(1, 2), size=(1000, 500))

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

u_exactFine = exact_solution(Nx, Ny, h)
u_exactCoarse = exact_solution(Nx ÷ 2 + 1, Ny ÷ 2 + 1, h * 2)

error_fine = norm(uh - u_exactFine, 2)
error_coarse = norm(uH - u_exactCoarse, 2)

display("L2 error (Fine Grid):       $error_fine")
display("L2 error (Coarse Grid):     $error_coarse")
display(" ")