using Plots
using LaTeXStrings

include("my_functions.jl")

# ---------------- PARAMETERS ----------------
Nx = 40
Ny = 40
h = 1.0 / (Nx - 1)  # Grid spacing
N = Nx * Ny


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