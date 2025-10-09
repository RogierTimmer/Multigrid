using LinearAlgebra
using Printf

# Helper: compute divergence field for all points
function compute_div(u, h)
    N = size(u, 1)
    div = zeros(N, N)
    for j in 2:N-1, i in 2:N-1
        div[i,j] = (u[i+1,j,1] - u[i-1,j,1])/(2h) + (u[i,j+1,2] - u[i,j-1,2])/(2h)
    end
    return div
end

# Block Jacobi smoother for elasticity
function block_jacobi_elasticity(U, F, h, λ, μ, iters; ω=0.1)
    N = size(U, 1)
    for _ in 1:iters
        Uold = copy(U)
        div = compute_div(Uold, h)
        for j in 2:N-1, i in 2:N-1
            # Laplacians
            lap_u1 = (Uold[i+1,j,1] + Uold[i-1,j,1] + Uold[i,j+1,1] + Uold[i,j-1,1] - 4 * Uold[i,j,1]) / h^2
            lap_u2 = (Uold[i+1,j,2] + Uold[i-1,j,2] + Uold[i,j+1,2] + Uold[i,j-1,2] - 4 * Uold[i,j,2]) / h^2
            # Grad(div u)
            ddiv_dx = (div[i+1,j] - div[i-1,j]) / (2h)
            ddiv_dy = (div[i,j+1] - div[i,j-1]) / (2h)
            # Diagonal coefficient
            a_diag = -4μ/h^2
            U[i,j,1] = (1-ω)*Uold[i,j,1] + ω * (F[i,j,1] - (μ * lap_u1 + (λ+μ)*ddiv_dx) + a_diag*Uold[i,j,1]) / a_diag
            U[i,j,2] = (1-ω)*Uold[i,j,2] + ω * (F[i,j,2] - (μ * lap_u2 + (λ+μ)*ddiv_dy) + a_diag*Uold[i,j,2]) / a_diag
        end
        # Dirichlet BCs
        U[1,:,:] .= 0.0; U[end,:,:] .= 0.0; U[:,1,:] .= 0.0; U[:,end,:] .= 0.0
    end
    return U
end

# Apply elasticity operator (for residual)
function apply_operator(u, h, λ, μ)
    N = size(u, 1)
    Lu = zeros(N, N, 2)
    div = compute_div(u, h)
    for j in 2:N-1, i in 2:N-1
        lap_u1 = (u[i+1,j,1] + u[i-1,j,1] + u[i,j+1,1] + u[i,j-1,1] - 4 * u[i,j,1]) / h^2
        lap_u2 = (u[i+1,j,2] + u[i-1,j,2] + u[i,j+1,2] + u[i,j-1,2] - 4 * u[i,j,2]) / h^2
        ddiv_dx = (div[i+1,j] - div[i-1,j]) / (2h)
        ddiv_dy = (div[i,j+1] - div[i,j-1]) / (2h)
        Lu[i,j,1] = μ * lap_u1 + (λ + μ) * ddiv_dx
        Lu[i,j,2] = μ * lap_u2 + (λ + μ) * ddiv_dy
    end
    return Lu
end

# Residual
function residual(U, F, h, λ, μ)
    Lu = apply_operator(U, h, λ, μ)
    return F - Lu
end

# Restriction (full weighting)
function restrict(r_f)
    Nf = size(r_f, 1)
    Nc = div(Nf - 1, 2) + 1
    r_c = zeros(Nc, Nc, 2)
    for jc in 2:Nc-1, ic in 2:Nc-1
        i, j = 2ic-1, 2jc-1
        for comp in 1:2
            r_c[ic, jc, comp] = (
                4 * r_f[i, j, comp] +
                2 * (r_f[i+1, j, comp] + r_f[i-1, j, comp] + r_f[i, j+1, comp] + r_f[i, j-1, comp]) +
                (r_f[i-1, j-1, comp] + r_f[i+1, j-1, comp] + r_f[i-1, j+1, comp] + r_f[i+1, j+1, comp])
            ) / 16
        end
    end
    return r_c
end

# Interpolation (bilinear)
function interpolate(e_c)
    Nc = size(e_c, 1)
    Nf = 2 * (Nc - 1) + 1
    e_f = zeros(Nf, Nf, 2)
    for comp in 1:2
        # Inject coarse points
        for jc in 1:Nc, ic in 1:Nc
            e_f[2ic-1, 2jc-1, comp] = e_c[ic, jc, comp]
        end
        # Interpolate odd rows, even columns
        for j in 1:2:Nf, i in 2:2:Nf-1
            e_f[i, j, comp] = 0.5 * (e_f[i-1, j, comp] + e_f[i+1, j, comp])
        end
        # Interpolate even rows
        for j in 2:2:Nf-1, i in 1:Nf
            e_f[i, j, comp] = 0.5 * (e_f[i, j-1, comp] + e_f[i, j+1, comp])
        end
    end
    return e_f
end

# V-cycle
function vcycle(U, F, h, level, maxlevel, nu1, nu2, λ, μ)
    N = size(U, 1)
    if level == maxlevel
        return block_jacobi_elasticity(U, F, h, λ, μ, 20; ω=0.2)
    end
    # Pre-smooth
    U = block_jacobi_elasticity(U, F, h, λ, μ, nu1; ω=0.1)
    # Residual
    r = residual(U, F, h, λ, μ)
    # Restrict
    r_c = restrict(r)
    Nc = size(r_c, 1)
    h_c = 2h
    # Coarse solve
    e_c = zeros(Nc, Nc, 2)
    e_c = vcycle(e_c, r_c, h_c, level+1, maxlevel, nu1, nu2, λ, μ)
    # Interpolate and correct
    e_f = interpolate(e_c)
    # Zero boundary corrections
    e_f[1, :, :] .= 0.0; e_f[end, :, :] .= 0.0; e_f[:, 1, :] .= 0.0; e_f[:, end, :] .= 0.0
    U += e_f
    # Post-smooth
    U = block_jacobi_elasticity(U, F, h, λ, μ, nu2; ω=0.1)
    return U
end

# Multigrid driver
function multigrid(F, N, h, vcycles, nu1, nu2, λ, μ)
    U = zeros(N, N, 2)
    maxlevel = Int(floor(log2(N - 1)))
    resvec = zeros(vcycles)
    for v = 1:vcycles
        U = vcycle(U, F, h, 1, maxlevel, nu1, nu2, λ, μ)
        res = residual(U, F, h, λ, μ)
        rnorm = norm(res)
        resvec[v] = rnorm
        @printf("Cycle %d: Residual = %.4e\n", v, rnorm)
    end
    return U, resvec
end

# RHS
function rhs(N, h)
    F = zeros(N, N, 2)
    i0, j0 = div(N,2), div(N,2)
    F[i0, j0, 1] += 1.0
    return F
end

function main()
    N = 17  # must be 2^k+1
    h = 1.0 / (N - 1)
    λ = 1.0
    μ = 1.0
    vcycles = 10
    nu1 = 3
    nu2 = 3
    F = rhs(N, h)
    U, resvec = multigrid(F, N, h, vcycles, nu1, nu2, λ, μ)
    @printf("Final residual: %.4e\n", resvec[end])
end

main()