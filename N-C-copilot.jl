using LinearAlgebra
using Printf
using Plots


# Helper: compute divergence field for all points
function compute_div(u, h)
    N = size(u, 1)
    div = zeros(N, N)
    for i in 1:N, j in 1:N
        # x-derivative
        if i == 1
            dudx = (u[i+1,j,1] - u[i,j,1]) / h  # forward
        elseif i == N
            dudx = (u[i,j,1] - u[i-1,j,1]) / h  # backward
        else
            dudx = (u[i+1,j,1] - u[i-1,j,1]) / (2h)
        end
        # y-derivative
        if j == 1
            dvdy = (u[i,j+1,2] - u[i,j,2]) / h  # forward
        elseif j == N
            dvdy = (u[i,j,2] - u[i,j-1,2]) / h  # backward
        else
            dvdy = (u[i,j+1,2] - u[i,j-1,2]) / (2h)
        end
        div[i,j] = dudx + dvdy
    end
    return div
end

# Block Jacobi smoother for elasticity (the crucial fix!)
function block_jacobi_elasticity(U, F, h, λ, μ, iters; ω=0.8)
    N = size(U, 1)
    for _ in 1:iters
        Uold = copy(U)
        div = compute_div(Uold, h)
        for j in 1:N, i in 1:N
            # Laplacians
            if (i == 1 || i == N || j == 1 || j == N)
                lap_u1 = 0.0
                lap_u2 = 0.0
            else
                lap_u1 = (Uold[i+1,j,1] + Uold[i-1,j,1] + Uold[i,j+1,1] + Uold[i,j-1,1] - 4 * Uold[i,j,1]) / h^2
                lap_u2 = (Uold[i+1,j,2] + Uold[i-1,j,2] + Uold[i,j+1,2] + Uold[i,j-1,2] - 4 * Uold[i,j,2]) / h^2
            end
            # Derivatives of divergence
            if i == 1
                ddiv_dx = (div[i+1,j] - div[i,j]) / h
            elseif i == N
                ddiv_dx = (div[i,j] - div[i-1,j]) / h
            else
                ddiv_dx = (div[i+1,j] - div[i-1,j]) / (2h)
            end
            if j == 1
                ddiv_dy = (div[i,j+1] - div[i,j]) / h
            elseif j == N
                ddiv_dy = (div[i,j] - div[i,j-1]) / h
            else
                ddiv_dy = (div[i,j+1] - div[i,j-1]) / (2h)
            end
            a_diag = -4μ/h^2
            if (i == 1 || i == N || j == 1 || j == N)
                # Dirichlet BCs: keep at zero (could skip update or explicitly set to zero)
                U[i,j,1] = 0.0
                U[i,j,2] = 0.0
            else
                U[i,j,1] = (1-ω)*Uold[i,j,1] + ω * (F[i,j,1] - (μ * lap_u1 + (λ+μ)*ddiv_dx) + a_diag*Uold[i,j,1]) / a_diag
                U[i,j,2] = (1-ω)*Uold[i,j,2] + ω * (F[i,j,2] - (μ * lap_u2 + (λ+μ)*ddiv_dy) + a_diag*Uold[i,j,2]) / a_diag
            end
        end
    end
    return U
end

function apply_operator(u, h, λ, μ)
    N = size(u, 1)
    Lu = zeros(N, N, 2)
    div = compute_div(u, h)
    for i in 1:N, j in 1:N
        # Laplacians (for boundaries, you may want one-sided, but often set to zero for Dirichlet BC)
        if (i == 1 || i == N || j == 1 || j == N)
            lap_u1 = 0.0
            lap_u2 = 0.0
        else
            lap_u1 = (u[i+1,j,1] + u[i-1,j,1] + u[i,j+1,1] + u[i,j-1,1] - 4 * u[i,j,1]) / h^2
            lap_u2 = (u[i+1,j,2] + u[i-1,j,2] + u[i,j+1,2] + u[i,j-1,2] - 4 * u[i,j,2]) / h^2
        end
        # Derivatives of divergence
        if i == 1
            ddiv_dx = (div[i+1,j] - div[i,j]) / h
        elseif i == N
            ddiv_dx = (div[i,j] - div[i-1,j]) / h
        else
            ddiv_dx = (div[i+1,j] - div[i-1,j]) / (2h)
        end
        if j == 1
            ddiv_dy = (div[i,j+1] - div[i,j]) / h
        elseif j == N
            ddiv_dy = (div[i,j] - div[i,j-1]) / h
        else
            ddiv_dy = (div[i,j+1] - div[i,j-1]) / (2h)
        end
        Lu[i,j,1] = μ * lap_u1 + (λ + μ) * ddiv_dx
        Lu[i,j,2] = μ * lap_u2 + (λ + μ) * ddiv_dy
    end
    return Lu
end

function residual(U, F, h, λ, μ)
    Lu = apply_operator(U, h, λ, μ)
    return F - Lu
end

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

function interpolate(e_c)
    Nc = size(e_c, 1)
    Nf = 2 * (Nc - 1) + 1
    e_f = zeros(Nf, Nf, 2)
    for comp in 1:2
        for jc in 1:Nc, ic in 1:Nc
            e_f[2ic-1, 2jc-1, comp] = e_c[ic, jc, comp]
        end
        for j in 1:2:Nf, i in 2:2:Nf-1
            e_f[i, j, comp] = 0.5 * (e_f[i-1, j, comp] + e_f[i+1, j, comp])
        end
        for j in 2:2:Nf-1, i in 1:Nf
            e_f[i, j, comp] = 0.5 * (e_f[i, j-1, comp] + e_f[i, j+1, comp])
        end
    end
    return e_f
end

function vcycle(U, F, h, level, maxlevel, nu1, nu2, λ, μ)
    N = size(U, 1)
    if level == maxlevel
        return block_jacobi_elasticity(U, F, h, λ, μ, 100; ω=0.8)
    end
    U = block_jacobi_elasticity(U, F, h, λ, μ, nu1; ω=1.0)
    r = residual(U, F, h, λ, μ)
    r_c = restrict(r)
    Nc = size(r_c, 1)
    h_c = 2h
    e_c = zeros(Nc, Nc, 2)
    e_c = vcycle(e_c, r_c, h_c, level+1, maxlevel, nu1, nu2, λ, μ)
    e_f = interpolate(e_c)
    e_f[1, :, :] .= 0.0; e_f[end, :, :] .= 0.0; e_f[:, 1, :] .= 0.0; e_f[:, end, :] .= 0.0
    U += e_f
    U = block_jacobi_elasticity(U, F, h, λ, μ, nu2; ω=0.1)
    return U
end

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

function rhs(N, h)
    F = zeros(N, N, 2)
    # Example: point force at center
    i0, j0 = div(N,2), div(N,2)
    F[i0, j0, 1] += 1.0
    return F
end

function main()
    N = 2049  # must be 2^k+1 for full multigrid
    h = 1.0 / (N - 1)
    λ = 1.0
    μ = 1.0
    vcycles = 20
    nu1 = 3
    nu2 = 3
    F = rhs_manufactured(N, h, λ, μ)
    U, resvec = multigrid(F, N, h, vcycles, nu1, nu2, λ, μ)
    @printf("Final residual: %.4e\n", resvec[end])
    # Optionally plot the displacement magnitude
    x = range(0, 1, length=N)
    y = range(0, 1, length=N)
    disp_mag = sqrt.(U[:,:,1].^2 + U[:,:,2].^2)
    disp_mag = (U[:,:,2])
    E = exact_manufactured(N, h)
    exact_mag = sqrt.(E[:,:,1].^2 + E[:,:,2].^2)
    exact_mag = E[:,:,1]
    contourf(x, y, disp_mag-exact_mag, xlabel="x", ylabel="y", title="Displacement Magnitude")
    #plot(1:vcycles, resvec, yscale=:log10, xlabel="V-cycle", ylabel="Residual", title="Residual Convergence", marker=:circle, grid=true)
end

function rhs_manufactured(N, h, λ, μ)
    F = zeros(N, N, 2)
    for j in 1:N
        y = (j - 1) * h
        for i in 1:N
            x = (i - 1) * h
            sinxy = sin(2π * x) * sin(2π * y)
            cosxy = cos(2π * x) * cos(2π * y)
            Fval = -8π^2 * μ * sinxy + 4π^2 * (λ + μ) * (cosxy - sinxy)
            F[i, j, 1] = Fval
            F[i, j, 2] = Fval
        end
    end
    return F
end

function exact_manufactured(N, h)
    uex = zeros(N, N, 2)
    for j in 1:N
        y = (j - 1) * h
        for i in 1:N
            x = (i - 1) * h
            val = sin(2π * x) * sin(2π * y)
            uex[i, j, 1] = val
            uex[i, j, 2] = val
        end
    end
    return uex
end

main()