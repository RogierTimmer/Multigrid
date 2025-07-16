using LinearAlgebra
using Printf
using Plots

#-----------------------------------------
function startsol(N)
    return zeros(N)
end

#-----------------------------------------
function rhs(N, h)
    f = zeros(N)
    for i in 1:N
        x = (i - 1) * h
        if 0.3 ≤ x ≤ 0.7
            f[i] = 1.0  # Pulse height
        end
    end
    return f
end


#-----------------------------------------
function exact(N, h, a)
    uex = zeros(N)
    for i in 1:N
        x = (i - 1) * h
        if x < 0.3
            uex[i] = 0
        elseif x ≤ 0.7
            uex[i] = (x - 0.3)
        else
            uex[i] = 0.4  # max pulse width
        end
    end
    return uex
end


#-----------------------------------------
function residual(u, f, N, h, a)
    r = zeros(N)
    for i in 2:N
        r[i] = f[i] - a * (u[i] - u[i-1]) / h
    end
    return r
end

#-----------------------------------------
function smooth(u, f, N, h, it, a)
    for _ in 1:it
        for i in 2:N
            u[i] = u[i-1] + h * f[i] / a
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
function mgstep(u, f, N, h, it, a)
    u[1] = 0.0  # Dirichlet BC at inflow

    # 1. Pre-smoothing
    u = smooth(u, f, N, h, it, a)

    # 2. Compute residual
    r = residual(u, f, N, h, a)

    # 3. Restrict residual
    r_c = restrict(r)

    # 4. Coarse-grid correction
    Nc = length(r_c)
    h_c = 2 * h
    e_c = zeros(Nc)
    e_c = smooth(e_c, r_c, Nc, h_c, 50, a)

    # 5. Interpolate
    e_f = interpolate(e_c)
    e_f[1] = 0.0  # Boundary correction

    # 6. Correct
    for i = 1:N
        u[i] += e_f[i]
    end

    # 7. Post-smoothing
    u = smooth(u, f, N, h, it, a)

    return u
end

#-----------------------------------------
function multigrid(f, N, h, vcycles, it, a)
    u = startsol(N)
    resvec = zeros(vcycles)

    for v in 1:vcycles
        u = mgstep(u, f, N, h, it, a)
        r = residual(u, f, N, h, a)
        resvec[v] = norm(r, 2)
        println("Cycle $v: Residual = ", @sprintf("%.2e", resvec[v]))
    end

    return u, resvec
end

#-----------------------------------------
function main()
    N = 513
    h = 1.0 / (N - 1)
    it = 50
    vcycles = 10
    a = 1.0

    f = rhs(N, h)
    u_exact = exact(N, h, a)
    u_mg, resvec = multigrid(f, N, h, vcycles, it, a)

    x = range(0, 1, length=N)

    plot(x, u_mg, label="Multigrid", linewidth=2)
    plot!(x, u_exact, label="Exact", linestyle=:dash,
          title="1D Advection (Upwind) Solution", xlabel="x", ylabel="u(x)")

    # Optional: Residual convergence plot
    # plot(resvec, yscale=:log10, xlabel="V-cycle", ylabel="Residual Norm",
    #      title="Residual Convergence", label="Residual Norm")
end

main()
