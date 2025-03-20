using LinearAlgebra
using Plots
using SparseArrays

# ---------------- PARAMETERS ----------------
Nx = 40
Ny = 40
h = 1.0
N = Nx * Ny

# Grid index mapping
index(i, j) = (j - 1) * Nx + i

# ---------------- LAPLACIAN FUNCTION ----------------
function Laplacian_2D(Nx, Ny, h;
    Dirichlet_left=nothing,
    Dirichlet_right=nothing,
    Dirichlet_bottom=nothing,
    Dirichlet_top=nothing)

    N = Nx * Ny
    A = zeros(Float64, N, N)
    index(i, j) = (j - 1) * Nx + i

    for j in 1:Ny
        for i in 1:Nx
            k = index(i, j)

            # Dirichlet boundary points
            if (i == 1 && Dirichlet_left !== nothing) ||
               (i == Nx && Dirichlet_right !== nothing) ||
               (j == 1 && Dirichlet_bottom !== nothing) ||
               (j == Ny && Dirichlet_top !== nothing)
                A[k, k] = 1.0
            else
                # Interior stencil
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


# ---------------- BOUNDARY FUNCTION ----------------
function boundary_2D(Nx, Ny, h;
    Dirichlet_left=nothing,
    Dirichlet_right=nothing,
    Dirichlet_bottom=nothing,
    Dirichlet_top=nothing)

    N = Nx * Ny
    f = zeros(Float64, N)
    index(i, j) = (j - 1) * Nx + i

    for j in 1:Ny
        for i in 1:Nx
            k = index(i, j)

            if i == 1 && Dirichlet_left !== nothing
                f[k] = Dirichlet_left
            elseif i == Nx && Dirichlet_right !== nothing
                f[k] = Dirichlet_right
            elseif j == 1 && Dirichlet_bottom !== nothing
                f[k] = Dirichlet_bottom
            elseif j == Ny && Dirichlet_top !== nothing
                f[k] = Dirichlet_top
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

    # Check for zero diagonal entries
    if any(diag(D) .== 0)
        error("Jacobi solver: Diagonal of A contains zero, cannot proceed.")
    end

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
A = Laplacian_2D(Nx, Ny, h; Dirichlet_bottom=0.0, Dirichlet_top=1.0)
f = boundary_2D(Nx, Ny, h; Dirichlet_bottom=0.0, Dirichlet_top=1.0)

# Jacobi solution
u_jacobi = jacobi_solver(A, f)

# Direct solution
u_direct = A \ f

println("u_jacobi contains NaN: ", any(isnan, u_jacobi))
println("u_direct contains NaN: ", any(isnan, u_direct))

# ---------------- PLOTTING ----------------
U_jacobi = reshape(u_jacobi, Nx, Ny)'  # Transpose to match (y, x)
U_direct = reshape(u_direct, Nx, Ny)'

x = range(0, stop=h*(Nx-1), length=Nx)
y = range(0, stop=h*(Ny-1), length=Ny)


# Note: 'contourf(x, y, U)' expects U[y,x] = rows = y-axis, columns = x-axis
p1 = contourf(x, y, U_jacobi, xlabel="x", ylabel="y", title="Jacobi Solution", levels=20, color=:turbo)
p2 = contourf(x, y, U_direct, xlabel="x", ylabel="y", title="Direct Solution", levels=20, color=:turbo)
plot(p1, p2, layout=(1,2), size=(1000,400))

