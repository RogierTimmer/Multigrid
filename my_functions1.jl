using LinearAlgebra

# ---------------- LAPLACIAN FUNCTION ----------------
function Laplacian_2D_new(Nx, Ny, h)
    N = Nx * Ny
    A = zeros(Float64, N, 5)
    for j in 1:Ny
        for i in 1:Nx
            k = (j - 1) * Nx + i
            if i == 1 || i == Nx || j == 1 || j == Ny  # Dirichlet boundary
                A[k, 3] = 1.0  # Center point for boundary
            else
                A[k, 3] = -4.0 / h^2  # Center point
                if i > 1
                    A[k, 2] = 1.0 / h^2  # Left neighbor
                end
                if i < Nx
                    A[k, 4] = 1.0 / h^2  # Right neighbor
                end
                if j > 1
                    A[k, 1] = 1.0 / h^2  # Top neighbor
                end
                if j < Ny
                    A[k, 5] = 1.0 / h^2  # Bottom neighbor
                end
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
            k = (j - 1) * Nx + i
            x = (i - 1) * h
            y = (j - 1) * h
            if i == 1 || i == Nx || j == 1 || j == Ny
                f[k] = 0.0
            else
                f[k] = -8π^2 * sin(2π * x) * sin(2π * y)
            end
        end
    end
    return f
end

# ---------------- JACOBI SOLVER ----------------
function jacobi_solver_new(A_new, b, Nx, Ny; tol=1e-5, max_iter=100000)
    N = length(b)
    x = zeros(Float64, N)
    x_new = similar(x)
    errorAprox = zeros(max_iter)

    for iter in 1:max_iter
        for k in 1:N
            sum_neighbors = 0.0
            if k > Nx
                sum_neighbors += A_new[k, 1] * x[k - Nx]  # Top neighbor
            end
            if (k - 1) % Nx != 0
                sum_neighbors += A_new[k, 2] * x[k - 1]  # Left neighbor
            end
            sum_neighbors += A_new[k, 3] * x[k]  # Center point
            if k % Nx != 0
                sum_neighbors += A_new[k, 4] * x[k + 1]  # Right neighbor
            end
            if k <= N - Nx
                sum_neighbors += A_new[k, 5] * x[k + Nx]  # Bottom neighbor
            end
            x_new[k] = (b[k] - sum_neighbors + A_new[k, 3] * x[k]) / A_new[k, 3]
        end

        errorAprox[iter] = norm(x_new - x, 2)
        if errorAprox[iter] < tol
            println("Jacobi (new datatype) converged in $iter iterations.")
            return x_new, errorAprox[1:iter], iter
        end
        x .= x_new
    end

    println("Jacobi (new datatype) reached max iterations.")
    return x, errorAprox, max_iter  # Use max_iter here since the loop completed
end

function gauss_seidel_solver_new(A_new, b, Nx ; tol=1e-5, max_iter=100000)
    N = length(b)
    x = zeros(Float64, N)
    x_old = similar(x)
    errorAprox = zeros(max_iter)
    iter = 0
    residual = zeros(Float64, N)

    for iter in 1:max_iter
        x_old .= x
        for k in 1:N
            sum_neighbors = 0.0
            if k > Nx
                sum_neighbors += A_new[k, 1] * x[k - Nx]  # Top
            end
            if (k - 1) % Nx != 0
                sum_neighbors += A_new[k, 2] * x[k - 1]  # Left
            end
            if k % Nx != 0
                sum_neighbors += A_new[k, 4] * x[k + 1]  # Right
            end
            if k <= N - Nx
                sum_neighbors += A_new[k, 5] * x[k + Nx]  # Bottom
            end
            x[k] = (b[k] - sum_neighbors) / A_new[k, 3]
        end

        # Compute residual vector r = b - A*x
        for k in 1:N
            Axk = A_new[k, 3] * x[k]
            if k > Nx
                Axk += A_new[k, 1] * x[k - Nx]
            end
            if (k - 1) % Nx != 0
                Axk += A_new[k, 2] * x[k - 1]
            end
            if k % Nx != 0
                Axk += A_new[k, 4] * x[k + 1]
            end
            if k <= N - Nx
                Axk += A_new[k, 5] * x[k + Nx]
            end
            residual[k] = b[k] - Axk
        end

        errorAprox[iter] = norm(x - x_old, 2)
        if errorAprox[iter] < tol
            println("Gauss-Seidel (new datatype) converged in $iter iterations.")
            return x, errorAprox[1:iter], iter, residual
        end
    end

    println("Gauss-Seidel (new datatype) reached max iterations.")
    return x, errorAprox, iter, residual
end


# ---------------- PLOTTING FUNCTION ----------------
function plot_contourf(u, Nx, Ny, h, title_str)
    U = reshape(u, Nx, Ny)'  # Transpose for correct orientation
    x = 0:h:(Nx - 1)*h
    y = 0:h:(Ny - 1)*h
    contourf(x, y, U;
             xlabel="x", ylabel="y", title=title_str,
             color=:viridis, aspect_ratio=1)
end

