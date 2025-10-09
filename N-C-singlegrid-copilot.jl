using LinearAlgebra
using Printf

function apply_operator(u, h, λ, μ)
    N = size(u, 1)
    Lu = zeros(N, N, 2)
    div = zeros(N, N)
    for j in 2:N-1, i in 2:N-1
        div[i,j] = (u[i+1,j,1] - u[i-1,j,1])/(2h) + (u[i,j+1,2] - u[i,j-1,2])/(2h)
    end
    for j in 2:N-1, i in 2:N-1
        lap_u1 = (u[i+1,j,1] + u[i-1,j,1] + u[i,j+1,1] + u[i,j-1,1] - 4 * u[i,j,1]) / h^2
        lap_u2 = (u[i+1,j,2] + u[i-1,j,2] + u[i,j+1,2] + u[i,j-1,2] - 4 * u[i,j,2]) / h^2
        # Gradient of divergence
        ddiv_dx = (div[i+1,j] - div[i-1,j]) / (2h)
        ddiv_dy = (div[i,j+1] - div[i,j-1]) / (2h)
        Lu[i,j,1] = μ * lap_u1 + (λ + μ) * ddiv_dx
        Lu[i,j,2] = μ * lap_u2 + (λ + μ) * ddiv_dy
    end
    return Lu
end

function jacobi_elasticity(U, F, h, λ, μ, iters; ω=0.01)
    N = size(U, 1)
    diag = (4μ + 2λ) / h^2  # approximate diagonal for both components
    for _ in 1:iters
        Uold = copy(U)
        div = zeros(N, N)
        for j in 2:N-1, i in 2:N-1
            div[i,j] = (Uold[i+1,j,1] - Uold[i-1,j,1])/(2h) + (Uold[i,j+1,2] - Uold[i,j-1,2])/(2h)
        end
        for j in 2:N-1, i in 2:N-1
            lap_u1 = (Uold[i+1,j,1] + Uold[i-1,j,1] + Uold[i,j+1,1] + Uold[i,j-1,1] - 4 * Uold[i,j,1]) / h^2
            lap_u2 = (Uold[i+1,j,2] + Uold[i-1,j,2] + Uold[i,j+1,2] + Uold[i,j-1,2] - 4 * Uold[i,j,2]) / h^2
            ddiv_dx = (div[i+1,j] - div[i-1,j]) / (2h)
            ddiv_dy = (div[i,j+1] - div[i,j-1]) / (2h)
            Lu1 = μ * lap_u1 + (λ + μ) * ddiv_dx
            Lu2 = μ * lap_u2 + (λ + μ) * ddiv_dy
            U[i,j,1] = Uold[i,j,1] + ω * (F[i,j,1] - Lu1) / diag
            U[i,j,2] = Uold[i,j,2] + ω * (F[i,j,2] - Lu2) / diag
        end
        # Dirichlet BC
        U[1,:,:] .= 0.0; U[end,:,:] .= 0.0; U[:,1,:] .= 0.0; U[:,end,:] .= 0.0
    end
    return U
end

function rhs(N, h)
    F = zeros(N, N, 2)
    i0, j0 = div(N,2), div(N,2)
    F[i0, j0, 1] += 1.0
    return F
end

function main()
    N = 17         # Must be >= 5 for stencils
    h = 1.0 / (N - 1)
    λ = 1.0
    μ = 1.0
    U = zeros(N, N, 2)
    F = rhs(N, h)
    niters = 20000
    for it = 1:niters
        U = jacobi_elasticity(U, F, h, λ, μ, 1; ω=0.01)
        if it % 1000 == 0
            Lu = apply_operator(U, h, λ, μ)
            res = F - Lu
            rnorm = norm(res)
            @printf("Iter %d: Residual = %.4e\n", it, rnorm)
        end
    end
end

main()