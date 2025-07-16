using LinearAlgebra
using Plots
using Printf
plotlyjs()

#-----------------------------------------
function startsol(N)
    return zeros(N)
end

#-----------------------------------------
function rhs(N, h)
    f = zeros(N)
    for i in 1:N
        x = (i - 1) * h
        f[i] = -4π^2 * sin(2π * x)  # -u''(x) = -4π^2 sin(2πx)
    end
    return f
end

#-----------------------------------------
function exact(N, h)
    uex = zeros(N)
    for i in 1:N
        x = (i - 1) * h
        uex[i] = sin(2π * x)
    end
    return uex
end

#-----------------------------------------
function residual(u, f, N, h)
    r = zeros(N)
    for i in 2:N-1
        r[i] = f[i] - (u[i-1] - 2*u[i] + u[i+1]) / h^2
    end
    return r
end

#-----------------------------------------
function smooth(u, f, N, h, it)
    for _ in 1:it
        for i in 2:N-1
            u[i] = (u[i-1] + u[i+1] - h^2 * f[i]) / 2
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
function mgstep(u, f, N, h, it)
    # 1. Pre-smoothing
    u = smooth(u, f, N, h, it)

    # 2. Compute residual
    r = residual(u, f, N, h)

    # 3. Restrict residual
    r_c = restrict(r)

    # 4. Solve coarse-grid error equation (approximate)
    Nc = length(r_c)
    h_c = 2 * h
    e_c = zeros(Nc)
    e_c = smooth(e_c, r_c, Nc, h_c, 50)

    # 5. Interpolate error
    e_f = interpolate(e_c)
    e_f[1] = 0.0
    e_f[end] = 0.0

    # 6. Correct fine-grid solution
    for i in 1:N
        u[i] += e_f[i]
    end

    # 7. Post-smoothing
    u = smooth(u, f, N, h, it)

    return u
end

#-----------------------------------------
function multigrid(f, N, h, vcycles, it)
    u = startsol(N)
    resvec = zeros(vcycles)

    for v in 1:vcycles
        u = mgstep(u, f, N, h, it)
        r = residual(u, f, N, h)
        resvec[v] = norm(r, 2)
        println("Cycle $v: Residual = ", @sprintf("%.2e", resvec[v]))
    end

    return u, resvec
end

#-----------------------------------------
function main()
    N = 513           # Must be 2^k + 1
    h = 1.0 / (N - 1)
    it = 50           # Smoothing iterations
    vcycles = 20      # Number of V-cycles

    f = rhs(N, h)
    u_exact = exact(N, h)
    u_mg, resvec = multigrid(f, N, h, vcycles, it)

    x = range(0, 1, length=N)

    # Plot solution
    p1 = plot(x, u_mg, label="Multigrid", linewidth=2)
    plot!(p1, x, u_exact, label="Exact", linestyle=:dash,
          title="Solution Comparison", xlabel="x", ylabel="u(x)")

    # Plot residual convergence
    p2 = plot(resvec, yscale=:log10, xlabel="V-cycle", ylabel="Residual Norm",
              title="Residual Convergence", label="Residual Norm")

    display(p1)
    display(p2)
end

main()
