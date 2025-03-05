using LinearAlgebra
using Plots
using CUDA

println("Started")

# Parameters
Nx = 101  # Number of grid points in x
Ny = 101  # Number of grid points in y
h = 1.0 # Grid spacing

# Grid size
N = Nx * Ny  # Total number of unknowns

# Dirichlet boundary conditions
Dirichlet_left = 0.0
Dirichlet_right = 1.0
Dirichlet_bottom = 0.0
Dirichlet_top = 6.0

# Initialize Laplacian matrix and right-hand side
A = zeros(Float64, N, N)
f = zeros(Float64, N)

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

# Move data to GPU
A_d = CuArray(A)
f_d = CuArray(f)


function jacobi_solver_gpu(A, b, tol=1e-6, max_iter=50000)
    N = length(b)
    x_d = CUDA.zeros(N)  # Initialize x on GPU
    x_new_d = similar(x_d)

    dA = diag(A)  # Extract diagonal elements

    for iter in 1:max_iter
        CUDA.@sync begin
            @cuda threads=512 blocks=ceil(Int, N / 512) jacobi_iteration!(x_new_d, A, b, x_d, dA, N)
        end

        if norm(x_new_d - x_d, Inf) < tol
            println("Converged in $iter iterations.")
            return Array(x_new_d)  # Transfer solution back to CPU
        end
        
        x_d .= x_new_d  # Update x for the next iteration
    end
    println("Reached max iterations.")
    return Array(x_new_d)  # Return the solution
end

# **CUDA Kernel for Jacobi Iteration**
function jacobi_iteration!(x_new, A, b, x, dA, N)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= N  # Ensure we don't go out of bounds
        sum_neighbors = zero(eltype(x))
        for j in 1:N
            if i != j
                sum_neighbors += A[i, j] * x[j]
            end
        end
        x_new[i] = (b[i] - sum_neighbors) / dA[i]
    end
    return nothing
end



# Solve system on GPU
solution = jacobi_solver_gpu(A_d, f_d)

# Compute error
error = norm(A * solution - f, 2)
println("Error:", error)

# Reshape solution back to 2D
U = reshape(solution, Nx, Ny)

# Plot solution
x = range(0, 1, length=Nx)
y = range(0, 1, length=Ny)
p = contourf(x, y, U', xlabel="x", ylabel="y", title="2D Poisson Equation Solution Jacobi (CUDA)", levels=20, color=:turbo)
savefig(p, "myfig_cuda.png")
