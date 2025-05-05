using LinearAlgebra
using Plots
using SparseArrays

# ---------------- PARAMETERS ----------------
Nx = 5
Ny = 5
h = 1.0 / (Nx - 1)  # Grid spacing
N = Nx * Ny
π2 = 2π  # Precompute for efficiency

# Grid index mapping
index(i, j) = (j - 1) * Nx + i

# ---------------- LAPLACIAN FUNCTION ----------------
function Laplacian_2D(Nx, Ny, h)
    N = Nx * Ny
    A = spzeros(N, N)  # Sparse storage
    
    for j in 1:Ny
        for i in 1:Nx
            k = index(i, j)
            
            if i == 1 || i == Nx || j == 1 || j == Ny  # Dirichlet boundary
                A[k, k] = 1.0
            else
                A[k, k] = -4.0 / h^2
                A[k, index(i-1, j)] = 1.0 / h^2
                A[k, index(i+1, j)] = 1.0 / h^2
                A[k, index(i, j-1)] = 1.0 / h^2
                A[k, index(i, j+1)] = 1.0 / h^2
            end
        end
    end
    return A
end

# ---------------- RIGHT-HAND SIDE FUNCTION ----------------
function rhs_2D(Nx, Ny, h)
    f = zeros(Nx * Ny)
    
    for j in 1:Ny
        for i in 1:Nx
            k = index(i, j)
            x = (i - 1) * h
            y = (j - 1) * h
            
            if i == 1 || i == Nx || j == 1 || j == Ny  # Dirichlet boundary
                f[k] = 0.0
            else
                f[k] = -8π^2 * sin(π2 * x) * sin(π2 * y)
            end
        end
    end
    return f
end

# ---------------- JACOBI SOLVER ----------------
function jacobi_solver(A, b; tol=1e-10, max_iter=1000)
    x = zeros(size(b))
    x_new = similar(x)
    D = diagm(0 => diag(A))
    R = A - D
    
    for iter in 1:max_iter
        x_new .= (b .- R * x) ./ diag(D)
        if norm(x_new - x, Inf) < tol
            println("Jacobi converged in $iter iterations.")
            return x_new
        end
        x .= x_new
    end
    println("Jacobi reached max iterations.")
    return x
end

# ---------------- SOLVE SYSTEM ----------------
A = Laplacian_2D(Nx, Ny, h)
f = rhs_2D(Nx, Ny, h)

# Jacobi solution
u_jacobi = jacobi_solver(A, f)

# Direct solution
u_direct = A \ f

# ---------------- PLOTTING ----------------
U_jacobi = reshape(u_jacobi, Nx, Ny)'
U_direct = reshape(u_direct, Nx, Ny)'

x = range(0, stop=1, length=Nx)
y = range(0, stop=1, length=Ny)

p1 = contourf(x, y, U_jacobi, xlabel="x", ylabel="y", title="Jacobi Solution", levels=20, color=:turbo)
p2 = contourf(x, y, U_direct, xlabel="x", ylabel="y", title="Direct Solution", levels=20, color=:turbo)
plot(p1, p2, layout=(1,2), size=(1000,400))
