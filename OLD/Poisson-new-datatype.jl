using LinearAlgebra
using Plots
using LaTeXStrings

# ---------------- PARAMETERS ----------------
Nx = 300
Ny = 300
h = 1.0 / (Nx - 1)  # Grid spacing
N = Nx * Ny

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

# ---------------- CONVERT TO FULL MATRIX ----------------
function convert_to_full_matrix(A_new, Nx, Ny)
    N = Nx * Ny
    A_full = zeros(Float64, N, N)
    for j in 1:Ny
        for i in 1:Nx
            k = (j - 1) * Nx + i
            if j > 1
                A_full[k, k - Nx] = A_new[k, 1]  # Top neighbor
            end
            if i > 1
                A_full[k, k - 1] = A_new[k, 2]  # Left neighbor
            end
            A_full[k, k] = A_new[k, 3]  # Center point
            if i < Nx
                A_full[k, k + 1] = A_new[k, 4]  # Right neighbor
            end
            if j < Ny
                A_full[k, k + Nx] = A_new[k, 5]  # Bottom neighbor
            end
        end
    end
    return A_full
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
function jacobi_solver_new(A_new, b; tol=1e-5, max_iter=10000)
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
            return x_new, errorAprox[1:iter]
        end
        x .= x_new
    end

    println("Jacobi (new datatype) reached max iterations.")
    return x, errorAprox
end

# ---------------- GAUSS-SEIDEL SOLVER ----------------
function gauss_seidel_solver_new(A_new, b; tol=1e-5, max_iter=10000)
    N = length(b)
    x = zeros(Float64, N)
    x_old = similar(x)
    errorAprox = zeros(max_iter)

    for iter in 1:max_iter
        x_old .= x
        for k in 1:N
            sum_neighbors = 0.0
            if k > Nx
                sum_neighbors += A_new[k, 1] * x[k - Nx]  # Top neighbor
            end
            if (k - 1) % Nx != 0
                sum_neighbors += A_new[k, 2] * x[k - 1]  # Left neighbor
            end
            if k % Nx != 0
                sum_neighbors += A_new[k, 4] * x[k + 1]  # Right neighbor
            end
            if k <= N - Nx
                sum_neighbors += A_new[k, 5] * x[k + Nx]  # Bottom neighbor
            end
            x[k] = (b[k] - sum_neighbors) / A_new[k, 3]
        end

        errorAprox[iter] = norm(x - x_old, 2)
        if errorAprox[iter] < tol
            println("Gauss-Seidel (new datatype) converged in $iter iterations.")
            return x, errorAprox[1:iter]
        end
    end

    println("Gauss-Seidel (new datatype) reached max iterations.")
    return x, errorAprox
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

# ---------------- MAIN EXECUTION ----------------
A = Laplacian_2D_new(Nx, Ny, h)
f = rhs_2D(Nx, Ny, h)

# Solve using Jacobi method
u_jacobi_new, errorAprox_jacobi_new = jacobi_solver_new(A, f)
plot_contourf(u_jacobi_new, Nx, Ny, h, "Jacobi Solution (Contour)")

# Solve using Gauss-Seidel method
u_gauss_seidel_new, errorAprox_gauss_seidel_new = gauss_seidel_solver_new(A, f)
plot_contourf(u_gauss_seidel_new, Nx, Ny, h, "Gauss-Seidel Solution (Contour)")



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

u_exact = exact_solution(Nx, Ny, h)

error_jacobi = norm(u_jacobi_new - u_exact, 2)
error_gs     = norm(u_gauss_seidel_new - u_exact, 2)

println("L2 error (Jacobi):       $error_jacobi")
println("L2 error (Gauss-Seidel): $error_gs")

function plot_error_contourf(u_numeric, u_exact, Nx, Ny, h, title_str)
    error_grid = reshape(abs.(u_numeric .- u_exact)*1000, Nx, Ny)'
    x = 0:h:(Nx - 1)*h
    y = 0:h:(Ny - 1)*h
    contourf(x, y, error_grid;
             xlabel="x", ylabel="y", title=title_str,
             color=:plasma, aspect_ratio=1, levels=20)
end



# Ensure both plots use the same color scale
min_error = min(minimum(abs.(u_jacobi_new .- u_exact)), minimum(abs.(u_gauss_seidel_new .- u_exact))) * 1000
max_error = max(maximum(abs.(u_jacobi_new .- u_exact)), maximum(abs.(u_gauss_seidel_new .- u_exact))) * 1000


plot1 = contourf(0:h:(Nx - 1)*h, 0:h:(Ny - 1)*h, reshape(abs.(u_jacobi_new .- u_exact)*1000, Nx, Ny)',
                 xlabel="x", ylabel="y", title="Error (Jacobi vs Exact) [x1000] \n  L2: $(round(error_jacobi, sigdigits=3)), Iterations: $(length(errorAprox_jacobi_new)) \n L_inf: $(round(norm(u_jacobi_new - u_exact, Inf),sigdigits=3)) Grid: $(Nx)x$(Ny)",
                 color=:plasma, aspect_ratio=1, levels=20, clim=(min_error, max_error))


plot2 = contourf(0:h:(Nx - 1)*h, 0:h:(Ny - 1)*h, reshape(abs.(u_gauss_seidel_new .- u_exact)*1000, Nx, Ny)',
                 xlabel="x", ylabel="y", title="Error (Gauss-Seidel vs Exact) [x1000] \n L2: $(round(error_gs,sigdigits=3)), Iterations: $(length(errorAprox_gauss_seidel_new)) \n L_inf: $(round(norm(u_gauss_seidel_new - u_exact, Inf),sigdigits=3)) Grid: $(Nx)x$(Ny)",
                 color=:plasma, aspect_ratio=1, levels=20, clim=(min_error, max_error))

plot(plot1, plot2, layout=(1, 2), size=(1000, 500))
