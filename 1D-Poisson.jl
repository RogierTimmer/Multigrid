using Plots
using LinearAlgebra

N = 513 #Gridpoints needs to be 2^i +1
h = 1.0 / (N - 1)  # Grid spacing
iter = 10
GSNormR = zeros(iter)
JNormR = zeros(iter)

u = zeros(N)
r = zeros(N)

u[1] = 0
u[N] = 0

f = zeros(N)
for i in 2:N-1
    x = (i - 1) * h
    f[i] = -8π^2 * sin(2π * x)
end

f[1] = f[1] - u[1]
f[N] = f[N] - u[N]


function Jacobi1D(f,u,N,iter,h)
    NormR = zeros(iter)
    r = zeros(N)
    for j in 1:iter

        for i in 2:N-1
            r[i] = f[i] - (u[i-1] - 2*u[i] - u[i+1])/h^2
        end
    
        for i in 2:N-1
            u[i] = u[i] - (r[i]*h^2)/(2)
        end
    NormR[j] = norm(r,2)
    end
    return NormR, u, r
end



function GS1D(f,u,N,iter,h)
    NormR = zeros(iter)
    r = zeros(N)

for j in 1:iter
    # Compute the initial residual
    for i in 2:N-1
        r[i] = f[i] - (u[i-1] - 2*u[i] - u[i+1])/h^2
    end

    for i in 2:N-1
        u[i] = u[i] - h^2 * r[i] / 2
        r[i] = f[i] - (u[i-1] - 2*u[i] - u[i+1])/h^2
    end
    NormR[j] = norm(r,2)
end
    return NormR, u, r
end




# Full weighting (restriction)
function restrict(r_fine)
    N_fine = length(r_fine)
    N_coarse = div(N_fine - 1, 2) + 1
    r_coarse = zeros(N_coarse)
    for i in 2:N_coarse-1
        r_coarse[i] = (r_fine[2i-2] + 2*r_fine[2i-1] + r_fine[2i]) / 4
    end
    return r_coarse
end

# Linear interpolation
function interpolate(e_coarse)
    N_coarse = length(e_coarse)
    N_fine = 2*(N_coarse - 1) + 1
    e_fine = zeros(N_fine)

    for i in 1:N_coarse
        e_fine[2i-1] = e_coarse[i]
    end

    for i in 2:2:N_fine-1
        e_fine[i] = 0.5 * (e_fine[i-1] + e_fine[i+1])
    end

    return e_fine
end


# 1. Pre-smoothing
_, u_smooth, _ = GS1D(f, copy(u), N, iter, h)

# 2. Compute residual
r = zeros(N)
for i in 2:N-1
    r[i] = f[i] - (u_smooth[i-1] - 2*u_smooth[i] + u_smooth[i+1]) / h^2
end

# 3. Restrict residual
r_coarse = restrict(r)

# 4. Solve Ae = r on coarse grid (approximate)
e0 = zeros(length(r_coarse))
_, e_coarse, _ = GS1D(r_coarse, e0, length(r_coarse), iter, 2*h)

# 5. Interpolate error
e_fine = interpolate(e_coarse)

# 6. Correct solution
u_corrected = u_smooth + e_fine

# 7. Post-smoothing
_, u_final, _ = GS1D(f, u_corrected, N, iter, h)
