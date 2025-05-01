using Plots
using LaTeXStrings
using PlotlyJS

include("my_functions1.jl")

plotlyjs()

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
i = 4  # Level parameter: 2^i + 1 grid points per direction
Nx = 2^i + 1
Ny = 2^i + 1
h = 1.0 / (Nx - 1)
N = Nx * Ny

# System setup
A = Laplacian_2D_new(Nx, Ny, h)
f = rhs_2D(Nx, Ny, h)
u_initial = zeros(N)

