using LinearAlgebra
using Printf
using Plots
using Logging

#-----------------------------------------
function elasticity_operator(u, div, i, j, h, λ, μ)
    lap_u1 = (u[i+1,j,1] + u[i-1,j,1] + u[i,j+1,1] + u[i,j-1,1] - 4 * u[i,j,1]) / h^2
    lap_u2 = (u[i+1,j,2] + u[i-1,j,2] + u[i,j+1,2] + u[i,j-1,2] - 4 * u[i,j,2]) / h^2

    ddiv_dx = (div[i+1, j] - div[i-1, j]) / (2*h)
    ddiv_dy = (div[i, j+1] - div[i, j-1]) / (2*h)

    Lu1 = μ * lap_u1 + (λ + μ) * ddiv_dx
    Lu2 = μ * lap_u2 + (λ + μ) * ddiv_dy

    return Lu1, Lu2
end

function compute_divergence(u, h)
    N = size(u, 1)
    div = zeros(N, N)

    for j in 2:N-1
        for i in 2:N-1
            div[i, j] = (u[i+1,j,1] - u[i-1,j,1]) / (2*h) + (u[i,j+1,2] - u[i,j-1,2]) / (2*h)
        end
    end

    return div
end




function rhs(N, h)
    F = zeros(N, N, 2)

    gravity = 9.81
    F[:, :, 2] .= -gravity  # uniform downward force

    # Apply a smoother localized force around center
    i0, j0 = div(N, 2), div(N, 2)
    force_magnitude = 100.0

    # Distribute force over 3x3 patch around center
    for di in -1:1
        for dj in -1:1
            weight = 1.0 / 9.0  # equal weights
            ii = i0 + di
            jj = j0 + dj
            if ii >= 1 && ii <= N && jj >= 1 && jj <= N
                F[ii, jj, 1] += force_magnitude * weight
            end
        end
    end

    return F
end


#-----------------------------------------
function residual(U, F, h, λ, μ)
    N = size(U, 1)
    r = zeros(N, N, 2)

    # Compute divergence once for the whole grid
    div = compute_divergence(U, h)

    for j in 2:N-1
        for i in 2:N-1
            Lu1, Lu2 = elasticity_operator(U, div, i, j, h, λ, μ)
            r[i, j, 1] = F[i, j, 1] - Lu1
            r[i, j, 2] = F[i, j, 2] - Lu2
        end
    end

    return r
end


function smooth_block(u, f, h, λ, μ, it)
    N = size(u, 1)
    A = (1 / h^2) * [2μ + λ  λ;
                      λ      2μ + λ]
    A_inv = inv(A)

    for _ in 1:it
        # Make a copy to avoid updating in-place for Jacobi
        u_old = copy(u)

        for j in 2:N-1
            for i in 2:N-1
                # Compute residual vector r = f - L(u_old)
                r = zeros(2)

                # Compute discrete Laplacians
                lap_u1 = (u_old[i+1, j, 1] + u_old[i-1, j, 1] + u_old[i, j+1, 1] + u_old[i, j-1, 1] - 4 * u_old[i, j, 1]) / h^2
                lap_u2 = (u_old[i+1, j, 2] + u_old[i-1, j, 2] + u_old[i, j+1, 2] + u_old[i, j-1, 2] - 4 * u_old[i, j, 2]) / h^2

                # Compute divergence derivatives (central differences)
                div_x = (u_old[i+1, j, 1] - u_old[i-1, j, 1]) / (2h)
                div_y = (u_old[i, j+1, 2] - u_old[i, j-1, 2]) / (2h)

                # Fix: remove division by h in residual computation
                r[1] = f[i, j, 1] - (μ * lap_u1 + (λ + μ) * div_x)
                r[2] = f[i, j, 2] - (μ * lap_u2 + (λ + μ) * div_y)


                # Update u at (i,j)
                du = A_inv * r
                u[i, j, 1] = u_old[i, j, 1] + du[1]
                u[i, j, 2] = u_old[i, j, 2] + du[2]
            end
        end

        # Apply Dirichlet BCs here if needed (e.g., zero displacement on boundaries)
        u[1, :, :] .= 0.0
        u[end, :, :] .= 0.0
        u[:, 1, :] .= 0.0
        u[:, end, :] .= 0.0
    end

    return u
end



function vcycle(U, F, h, level, maxlevel, nu1, nu2, λ, μ)
    N = size(U, 1)
    @info "V-cycle Level: $level, Grid Size: $N"

    if level == maxlevel
        U = smooth_block(U, F, h, 50, λ, μ)  # coarse grid solve
        return U
    end

    # 1. Pre-smoothing
    U = smooth_block(U, F, h, nu1, λ, μ)

    # 2. Compute residual
    r = residual(U, F, h, λ, μ)

    # 3. Restrict residual to coarse grid
    r_c = restrict(r)  # restrict must handle 3D array
    Nc = size(r_c, 1)
    h_c = 2h

    # 4. Initialize error on coarse grid and solve recursively
    e_c = zeros(Nc, Nc, 2)
    e_c = vcycle(e_c, r_c, h_c, level + 1, maxlevel, nu1, nu2, λ, μ)

    # 5. Interpolate error to fine grid and correct
    e_f = interpolate(e_c)  # interpolate must handle 3D array

    # Zero boundary corrections (assuming Dirichlet)
    e_f[1, :, :] .= 0.0
    e_f[end, :, :] .= 0.0
    e_f[:, 1, :] .= 0.0
    e_f[:, end, :] .= 0.0

    U += e_f

    # 6. Post-smoothing
    U = smooth_block(U, F, h, nu2, λ, μ)

    # Log residual norm (L2 norm over both components)
    res_norm = norm(r)
    @info "V-cycle Level: $level, Residual Norm: $res_norm"
    return U
end

#function smooth(U, F, h, it, λ, μ; ω=0.8)
#    N = size(U, 1)
#    U_new = copy(U)
#
#    for _ in 1:it
#        for j in 2:N-2
#            for i in 2:N-2
#                L_u, L_v = elasticity_operator(U, i, j, h, λ, μ)
#                
#                # Compute residuals
#                r_u = F[i, j, 1] - L_u
#                r_v = F[i, j, 2] - L_v
#
#                # Relaxation update
#                U_new[i, j, 1] = U[i, j, 1] + ω * r_u
#                U_new[i, j, 2] = U[i, j, 2] + ω * r_v
#            end
#        end
#        U, U_new = U_new, U  # swap references for next iteration
#    end
#
#    return U
#end


function restrict(r_f)
    Nf = size(r_f, 1)
    Nc = div(Nf - 1, 2) + 1
    r_c = zeros(Nc, Nc, 2)

    for jc in 2:Nc-1
        for ic in 2:Nc-1
            i, j = 2ic - 1, 2jc - 1
            for comp in 1:2
                r_c[ic, jc, comp] = (
                    4 * r_f[i, j, comp] +
                    2 * (r_f[i+1, j, comp] + r_f[i-1, j, comp] + r_f[i, j+1, comp] + r_f[i, j-1, comp]) +
                    (r_f[i-1, j-1, comp] + r_f[i+1, j-1, comp] + r_f[i-1, j+1, comp] + r_f[i+1, j+1, comp])
                ) / 16
            end
        end
    end

    return r_c
end


function interpolate(e_c)
    Nc = size(e_c, 1)
    Nf = 2 * (Nc - 1) + 1
    e_f = zeros(Nf, Nf, 2)

    for comp in 1:2
        # Inject coarse points
        for jc in 1:Nc
            for ic in 1:Nc
                e_f[2ic - 1, 2jc - 1, comp] = e_c[ic, jc, comp]
            end
        end

        # Interpolate odd rows, even columns
        for j in 1:2:Nf
            for i in 2:2:Nf-1
                e_f[i, j, comp] = 0.5 * (e_f[i - 1, j, comp] + e_f[i + 1, j, comp])
            end
        end

        # Interpolate even rows
        for j in 2:2:Nf-1
            for i in 1:Nf
                e_f[i, j, comp] = 0.5 * (e_f[i, j - 1, comp] + e_f[i, j + 1, comp])
            end
        end
    end

    return e_f
end


function multigrid(F, N, h, vcycles, nu1, nu2, λ, μ)
    U = zeros(N, N, 2)  # displacement initialization
    maxlevel = Int(floor(log2(N - 1)))  # grid levels
    resvec = zeros(vcycles)

    for v = 1:vcycles
        U = vcycle(U, F, h, 1, maxlevel, nu1, nu2, λ, μ)
        r = residual(U, F, h, λ, μ)
        resvec[v] = norm(r)
        @info "Cycle $v: Residual = ", @sprintf("%.2e", resvec[v])
    end

    return U, resvec
end


function main()
    N = 17  # Must be 2^k + 1
    h = 1.0 / (N - 1)
    vcycles = 2
    nu1 = 2
    nu2 = 2

    # Material parameters (example)
    λ = 1.0
    μ = 1.0

    # RHS body forces (vector valued)
    F = rhs(N, h)

    # Exact solution for testing (if available)
    u_exact = zeros(N, N, 2)  # replace with exact(N, h) if defined

    # Run multigrid solver
    U, resvec = multigrid(F, N, h, vcycles, nu1, nu2, λ, μ)

    # Coordinate grids
    x = range(0, 1, length=N)
    y = range(0, 1, length=N)

    # Plot error magnitude
    error_mag = sqrt.(dropdims(sum(U.^2, dims=3), dims=3))
    #surface(x, y, error_mag, xlabel="x", ylabel="y", title="Multigrid Error Magnitude")
    
    # Plot residual convergence
    #plot(1:length(resvec), resvec, yscale=:log10,
       # xlabel="V-cycle", ylabel="Residual Norm",
       # title="Residual Convergence Over V-cycles", marker=:circle, grid=true)
end


main()