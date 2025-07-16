using LinearAlgebra
using Plots

#-----------------------------------------
function rhs(N, h)
    f = zeros(N, N)
    for j in 1:N
        y = (j - 1) * h
        for i in 1:N
            x = (i - 1) * h
            f[i, j] = 2π^2 * sin(2π * x) * sin(2π * y)
        end
    end
    return f
end

#-----------------------------------------
function exact(N, h)
    uex = zeros(N, N)
    for j in 1:N
        y = (j - 1) * h
        for i in 1:N
            x = (i - 1) * h
            uex[i, j] = sin(π * x) * sin(π * y)
        end
    end
    return uex
end

#-----------------------------------------
function residual(u, f, N, h)
    r = zeros(N, N)
    for j in 2:N-1
        for i in 2:N-1
            r[i, j] = f[i, j] - (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4u[i, j]) / h^2
        end
    end
    return r
end

#-----------------------------------------
function smooth(u, f, N, h, it)
    for _ in 1:it
        for j in 2:N-1
            for i in 2:N-1
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - h^2 * f[i, j])
            end
        end
    end
    return u
end

#-----------------------------------------
function restrict(r_f)
    Nf = size(r_f, 1)
    Nc = div(Nf - 1, 2) + 1
    r_c = zeros(Nc, Nc)
    for jc in 2:Nc-1
        for ic in 2:Nc-1
            i, j = 2ic - 1, 2jc - 1
            r_c[ic, jc] = (
                4r_f[i, j] +
                2(r_f[i+1, j] + r_f[i-1, j] + r_f[i, j+1] + r_f[i, j-1]) +
                (r_f[i-1, j-1] + r_f[i+1, j-1] + r_f[i-1, j+1] + r_f[i+1, j+1])
            ) / 16
        end
    end
    return r_c
end

#-----------------------------------------
function interpolate(e_c)
    Nc = size(e_c, 1)
    Nf = 2 * (Nc - 1) + 1
    e_f = zeros(Nf, Nf)

    for jc in 1:Nc
        for ic in 1:Nc
            e_f[2ic - 1, 2jc - 1] = e_c[ic, jc]
        end
    end

    for j in 1:2:Nf
        for i in 2:2:Nf-1
            e_f[i, j] = 0.5 * (e_f[i-1, j] + e_f[i+1, j])
        end
    end

    for j in 2:2:Nf-1
        for i in 1:Nf
            e_f[i, j] = 0.5 * (e_f[i, j-1] + e_f[i, j+1])
        end
    end

    return e_f
end

#-----------------------------------------
function mgstep(u, f, N, h, nu)
    u = smooth(u, f, N, h, nu)
    r = residual(u, f, N, h)
    r_c = restrict(r)

    Nc = size(r_c, 1)
    h_c = 2h
    e_c = zeros(Nc, Nc)
    e_c = smooth(e_c, r_c, Nc, h_c, 50)

    e_f = interpolate(e_c)
    e_f[1, :] .= 0.0
    e_f[end, :] .= 0.0
    e_f[:, 1] .= 0.0
    e_f[:, end] .= 0.0

    u += e_f
    u = smooth(u, f, N, h, nu)
    return u
end

#-----------------------------------------
function multigrid(f, N, h, vcycles, nu)
    u = zeros(N, N)
    resvec = zeros(vcycles)

    for v in 1:vcycles
        u = mgstep(u, f, N, h, nu)
        r = residual(u, f, N, h)
        resvec[v] = norm(r)
        println("Cycle $v: Residual = ", @sprintf("%.2e", resvec[v]))
    end

    return u, resvec
end

#-----------------------------------------
function main()
    N = 257  # Must be 2^k + 1
    h = 1.0 / (N - 1)
    nu = 300
    vcycles = 100

    f = rhs(N, h)
    u_exact = exact(N, h)
    u, resvec = multigrid(f, N, h, vcycles, nu)

    x = range(0, 1, length=N)
    y = range(0, 1, length=N)

    surface(x, y, u, xlabel="x", ylabel="y", title="Multigrid Solution")
    # Optional: plot residual
    # plot(resvec, yscale=:log10, xlabel="V-cycle", ylabel="Residual", title="Convergence")
end

main()
