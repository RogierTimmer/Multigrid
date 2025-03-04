using LinearAlgebra
using Plots

# Parameters
Nx = 201  # Number of grid points in x
Ny = 201  # Number of grid points in y
h = 1.0 / (Nx - 1)  # Grid spacing
h = 1

# Grid size
N = Nx * Ny  # Total number of unknowns

# Dirichlet boundary conditions
Dirichlet_left = 0.0
Dirichlet_right = 1.0
Dirichlet_bottom = 0.0
Dirichlet_top = 6.0

# Initialize Laplacian matrix and right-hand side
A = zeros(Float32, N, N)
f = zeros(N)

# Map (i, j) indices to 1D index
index(i, j) = (j - 1) * Nx + i

# Construct Laplacian matrix
for j in 1:Ny
    for i in 1:Nx
        k = index(i, j)  # Current index
        
        if i == 1  # Left boundary (Dirichlet)
            A[k, k] = 1.0
            f[k] = Dirichlet_left
        elseif i == Nx  # Right boundary (Dirichlet)
            A[k, k] = 1.0
            f[k] = Dirichlet_right
        elseif j == 1  # Bottom boundary (Dirichlet)
            A[k, k] = 1.0
            f[k] = Dirichlet_bottom
        elseif j == Ny  # Top boundary (Dirichlet)
            A[k, k] = 1.0
            f[k] = Dirichlet_top
        else  # Interior points (Finite Difference Approximation)
            A[k, k] = -4.0 / h^2
            A[k, index(i-1, j)] = 1.0 / h^2  # Left neighbor
            A[k, index(i+1, j)] = 1.0 / h^2  # Right neighbor
            A[k, index(i, j-1)] = 1.0 / h^2  # Bottom neighbor
            A[k, index(i, j+1)] = 1.0 / h^2  # Top neighbor
        end
    end
end

# Solve system
u = A \ f

# Reshape solution back to 2D
U = reshape(u, Nx, Ny)

# Plot solution
x = range(0, 1, length=Nx)
y = range(0, 1, length=Ny)
heatmap(x, y, U', xlabel="x", ylabel="y", title="2D Poisson Equation Solution", color=:viridis)
contourf(x, y, U', xlabel="x", ylabel="y", title="2D Poisson Equation Solution", levels=20, color=:turbo)