using LinearAlgebra
using Printf
using Plots

#-----------------------------------------
function rhs(N, h)
    f = zeros(N, N)
    for j in 1:N
        y = (j - 1) * h
        for i in 1:N
            x = (i - 1) * h
            f[i, j] = -8π^2 * sin(2π * x) * sin(2π * y)
        end
    end
    return f
end

function exact(N, h)
    uex = zeros(N, N)
    for j in 1:N
        y = (j - 1) * h
        for i in 1:N
            x = (i - 1) * h
            uex[i, j] = sin(2π * x) * sin(2π * y)
        end
    end
    return uex
end

#-----------------------------------------
function residual(u, f, h)
    N = size(u, 1)
    r = zeros(N, N)
    for j in 2:N-1
        for i in 2:N-1
            r[i, j] = f[i, j] - (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4u[i,j]) / h^2
        end
    end
    return r
end

function smooth(u, f, h, it)
    N = size(u, 1)
    for _ in 1:it
        for j in 2:N-1
            for i in 2:N-1
                u[i, j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h^2 * f[i, j])
            end
        end
    end
    return u
end

function restrict(r_f)
    Nf = size(r_f, 1)
    Nc = div(Nf - 1, 2) + 1
    r_c = zeros(Nc, Nc)
    for jc in 2:Nc-1
        for ic in 2:Nc-1
            i, j = 2ic - 1, 2jc - 1
            r_c[ic, jc] = (
                4r_f[i, j] +
                2(r_f[i+1,j] + r_f[i-1,j] + r_f[i,j+1] + r_f[i,j-1]) +
                (r_f[i-1,j-1] + r_f[i+1,j-1] + r_f[i-1,j+1] + r_f[i+1,j+1])
            ) / 16
        end
    end
    return r_c
end

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
function vcycle(u, f, h, level, maxlevel, nu1, nu2)
    N = size(u, 1)
    println("V-cycle Level: $level, Grid Size: $N")

    if level == maxlevel
        u = smooth(u, f, h, 50)  # coarse grid solve
        return u
    end

    # 1. Pre-smoothing
    u = smooth(u, f, h, nu1)

    # 2. Compute residual
    r = residual(u, f, h)

    # 3. Restrict residual to coarse grid
    r_c = restrict(r)
    Nc = size(r_c, 1)
    h_c = 2h

    # 4. Solve error equation recursively
    e_c = zeros(Nc, Nc)
    e_c = vcycle(e_c, r_c, h_c, level + 1, maxlevel, nu1, nu2)

    # 5. Interpolate and correct
    e_f = interpolate(e_c)
    e_f[1, :] .= 0.0
    e_f[end, :] .= 0.0
    e_f[:, 1] .= 0.0
    e_f[:, end] .= 0.0
    u += e_f

    # 6. Post-smoothing
    u = smooth(u, f, h, nu2)
    println("V-cycle Level: $level, Residual Norm: ", norm(r))

    return u
end

function wcycle(u, f, h, level, maxlevel, nu1, nu2)
    N = size(u, 1)
    println("W-cycle Level: $level, Grid Size: $N")

    if level == maxlevel
        u = smooth(u, f, h, 50)  # coarse grid solve
        return u
    end

    # 1. Pre-smoothing
    u = smooth(u, f, h, nu1)

    # 2. Compute residual
    r = residual(u, f, h)

    # 3. Restrict residual to coarse grid
    r_c = restrict(r)
    Nc = size(r_c, 1)
    h_c = 2h

    # 4. Solve error equation recursively (W-cycle: two recursive calls)
    e_c = zeros(Nc, Nc)
    e_c = wcycle(e_c, r_c, h_c, level + 1, maxlevel, nu1, nu2)
    e_c = wcycle(e_c, r_c, h_c, level + 1, maxlevel, nu1, nu2)

    # 5. Interpolate and correct
    e_f = interpolate(e_c)
    e_f[1, :] .= 0.0
    e_f[end, :] .= 0.0
    e_f[:, 1] .= 0.0
    e_f[:, end] .= 0.0
    u += e_f

    # 6. Post-smoothing
    u = smooth(u, f, h, nu2)
    println("W-cycle Level: $level, Residual Norm: ", norm(r))

    return u
end

function fcycle(u, f, h, level, maxlevel, nu1, nu2, is_first=true)
    N = size(u, 1)
    println("F-cycle Level: $level, Grid Size: $N")

    if level == maxlevel
        u = smooth(u, f, h, 50)  # coarse grid solve
        return u
    end

    # 1. Pre-smoothing
    u = smooth(u, f, h, nu1)

    # 2. Compute residual
    r = residual(u, f, h)

    # 3. Restrict residual to coarse grid
    r_c = restrict(r)
    Nc = size(r_c, 1)
    h_c = 2h

    # 4. Solve error equation recursively
    e_c = zeros(Nc, Nc)

    if is_first
        # For first call: go down twice like W-cycle
        e_c = fcycle(e_c, r_c, h_c, level + 1, maxlevel, nu1, nu2, true)
        e_c = fcycle(e_c, r_c, h_c, level + 1, maxlevel, nu1, nu2, false)
    else
        # After first call: V-cycle behavior
        e_c = fcycle(e_c, r_c, h_c, level + 1, maxlevel, nu1, nu2, false)
    end

    # 5. Interpolate and correct
    e_f = interpolate(e_c)
    e_f[1, :] .= 0.0
    e_f[end, :] .= 0.0
    e_f[:, 1] .= 0.0
    e_f[:, end] .= 0.0
    u += e_f

    # 6. Post-smoothing
    u = smooth(u, f, h, nu2)
    println("F-cycle Level: $level, Residual Norm: ", norm(r))

    return u
end



#-----------------------------------------
function multigrid(f, N, h, vcycles, nu1, nu2)
    u = zeros(N, N)
    maxlevel = Int(floor(log2(N - 1)))  # finest level = 1, coarsest = maxlevel
    resvec = zeros(vcycles)

    for v = 1:vcycles
        u = fcycle(u, f, h, 1, maxlevel, nu1, nu2)
        r = residual(u, f, h)
        resvec[v] = norm(r)
        println("Cycle $v: Residual = ", @sprintf("%.2e", resvec[v]))
    end

    return u, resvec
end

#-----------------------------------------
function main()
    N = 257  # Must be 2^k + 1
    h = 1.0 / (N - 1)
    vcycles = 10
    nu1 = 5
    nu2 = 5

    f = rhs(N, h)
    u_exact = exact(N, h)
    u, resvec = multigrid(f, N, h, vcycles, nu1, nu2)

    x = range(0, 1, length=N)
    y = range(0, 1, length=N)

    surface(x, y, u-u_exact, xlabel="x", ylabel="y", title="Recursive Multigrid Solution")
    #plot(resvec, yscale=:log10, xlabel="V-cycle", ylabel="Residual", title="Residual Convergence")
    
end

main()
