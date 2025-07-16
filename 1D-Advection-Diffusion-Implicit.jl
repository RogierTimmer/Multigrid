using LinearAlgebra
using Printf
using Plots

#-----------------------------------------
function initial_condition(N, h)
    u0 = zeros(N)
    for i in 1:N
        x = (i - 1) * h
        if 0.3 ≤ x ≤ 0.7
            u0[i] = 1.0
        end
    end
    return u0
end

#-----------------------------------------
function apply_operator(u, N, h, dt, a, D)
    A = zeros(N)
    for i in 2:N-1
        adv = -a * (u[i] - u[i-1]) / h
        diff = D * (u[i-1] - 2*u[i] + u[i+1]) / h^2
        A[i] = u[i] / dt + adv - diff
    end
    return A
end

#-----------------------------------------
function residual(u, rhs, N, h, dt, a, D)
    A_u = apply_operator(u, N, h, dt, a, D)
    r = rhs - A_u
    return r
end

#-----------------------------------------
function smooth(u, rhs, N, h, dt, a, D, iters)
    for _ in 1:iters
        for i in 2:N-1
            # Backward Euler Gauss-Seidel update
            num = rhs[i] + D/h^2 * (u[i-1] + u[i+1]) + a/h * u[i-1]
            denom = 1/dt + 2D/h^2 + a/h
            u[i] = num / denom
        end
    end
    return u
end

#-----------------------------------------
function restrict(r_f)
    Nf = length(r_f)
    Nc = div(Nf - 1, 2) + 1
    r_c = zeros(Nc)
    for i in 2:Nc-1
        r_c[i] = (r_f[2i-2] + 2*r_f[2i-1] + r_f[2i]) / 4
    end
    return r_c
end

#-----------------------------------------
function interpolate(e_c)
    Nc = length(e_c)
    Nf = 2 * (Nc - 1) + 1
    e_f = zeros(Nf)

    for i in 1:Nc
        e_f[2i - 1] = e_c[i]
    end
    for i in 2:2:Nf-1
        e_f[i] = 0.5 * (e_f[i-1] + e_f[i+1])
    end

    return e_f
end

#-----------------------------------------
function mgstep(u, rhs, N, h, dt, a, D, nu)
    u = smooth(u, rhs, N, h, dt, a, D, nu)
    r = residual(u, rhs, N, h, dt, a, D)
    r_c = restrict(r)

    Nc = length(r_c)
    h_c = 2 * h
    dt_c = dt  # keep time step fixed
    e_c = zeros(Nc)
    e_c = smooth(e_c, r_c, Nc, h_c, dt_c, a, D, 50)

    e_f = interpolate(e_c)
    e_f[1] = 0
    e_f[end] = 0

    for i in 1:N
        u[i] += e_f[i]
    end

    u = smooth(u, rhs, N, h, dt, a, D, nu)
    return u
end

#-----------------------------------------
function solve_linear(rhs, N, h, dt, a, D, vcycles, nu)
    u = zeros(N)
    for v in 1:vcycles
        u = mgstep(u, rhs, N, h, dt, a, D, nu)
        r = residual(u, rhs, N, h, dt, a, D)
        println("  V-cycle $v: Residual = ", @sprintf("%.2e", norm(r, 2)))
    end
    return u
end

#-----------------------------------------
function main()
    # Grid + time setup
    N = 513
    L = 1.0
    h = L / (N - 1)
    T = 4
    dt = 0.01
    steps = Int(T / dt)

    # PDE parameters
    a = 1.0
    D = 0.01

    # Initial condition
    u = initial_condition(N, h)
    x = range(0, L, length=N)

    anim = @animate for step in 1:steps
        rhs = u ./ dt  # Backward Euler RHS

        u = solve_linear(rhs, N, h, dt, a, D, 2, 3)  # vcycles=2, nu=3

        plot(x, u, ylim=(0, 1.1), label="t = $(round(step*dt, digits=2))",
             xlabel="x", ylabel="u(x,t)", title="Advection-Diffusion (Multigrid)")
    end

    gif(anim, "advection_diffusion_multigrid.gif", fps=15)
end

main()
