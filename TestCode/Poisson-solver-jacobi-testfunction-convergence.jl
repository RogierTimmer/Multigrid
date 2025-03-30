using LinearAlgebra
using Plots
using SparseArrays

# ---------------- PARAMETERS ----------------
Nx = 50
Ny = 50
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
function jacobi_solver(A, b; tol=1e-10, max_iter=1000, u_direct)
    x = zeros(size(b))
    x_new = similar(x)
    D = diagm(0 => diag(A))
    R = A - D
    errorAprox = zeros(max_iter)  # Preallocate error storage

    for iter in 1:max_iter
        x_new .= (b .- R * x) ./ diag(D)
        errorAprox[iter] = norm(x_new - u_direct, 2)  # Compute L2 norm error
        
        if norm(x_new - x, Inf) < tol
            println("Jacobi converged in $iter iterations.")
            return x_new, errorAprox[1:iter]  # Return only filled part of errorAprox
        end
        x .= x_new
    end

    println("Jacobi reached max iterations.")
    return x, errorAprox
end

# ---------------- GAUSS-SEIDEL SOLVER ----------------

function gauss_seidel_solver(x, A, b; tol=1e-10, max_iter=1000, u_direct=nothing)
    N = length(b)
    errorAprox = zeros(max_iter)

    for iter in 1:max_iter
        for i in 1:N
            sum1 = 0.0
            sum2 = 0.0

            # Compute sum(A[i, j] * x[j]) manually
            for j in 1:i-1
                sum1 += A[i, j] * x[j]
            end
            for j in i+1:N
                sum2 += A[i, j] * x[j]
            end

            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        end

        # Compute L2 norm error only if reference solution is provided
        if u_direct !== nothing
            errorAprox[iter] = norm(x - u_direct, 2)
        end

        if norm(A * x - b, Inf) < tol
            println("Gauss-Seidel converged in $iter iterations.")
            return x, errorAprox[1:iter]  # Return only filled part
        end
    end

    println("Gauss-Seidel reached max iterations.")
    return x, errorAprox
end




# ---------------- SOLVE SYSTEM ----------------
A = Laplacian_2D(Nx, Ny, h)
f = rhs_2D(Nx, Ny, h)


# Direct solution
u_direct = A \ f

# Jacobi solution
u_jacobi, errorAprox = jacobi_solver(A, f; u_direct=u_direct)
display(errorAprox)

# Gauss-Seidel solution
x = zeros(N)  # Initial guess
u_gauss_seidel, errorAprox_gs = gauss_seidel_solver(x, A, f; u_direct=u_direct)
display(errorAprox_gs)

# ---------------- PLOTTING ----------------
U_jacobi = reshape(u_jacobi, Nx, Ny)'
U_direct = reshape(u_direct, Nx, Ny)'

x = range(0, stop=1, length=Nx)
y = range(0, stop=1, length=Ny)

p1 = contourf(x, y, U_jacobi, xlabel="x", ylabel="y", title="Jacobi Solution", levels=20, color=:turbo)
p2 = contourf(x, y, U_direct, xlabel="x", ylabel="y", title="Direct Solution", levels=20, color=:turbo)
plot(p1, p2, layout=(1,2), size=(1000,400))

plot(errorAprox, xlabel="Iteration", ylabel="L2 Error", title="Convergence error", label="Jacobi", color=:blue)
plot!(errorAprox_gs, label="Gauss-Seidel", color=:red)