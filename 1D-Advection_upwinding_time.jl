using DifferentialEquations
using Plots

#-----------------------------------------
function initial_condition(N, h)
    u0 = zeros(N)
    for i in 1:N
        x = (i - 1) * h
        if 0.3 ≤ x ≤ 0.7
            u0[i] = 1.0  # pulse
        end
    end
    return u0
end

#-----------------------------------------
function advection_ode!(du, u, p, t)
    a, h = p
    N = length(u)
    
    du[1] = 0.0  # Enforce inflow BC: u(0, t) = 0

    for i in 2:N
        du[i] = -a * (u[i] - u[i-1]) / h
    end
end

#-----------------------------------------
function main()
    N = 201
    L = 1.0
    h = L / (N - 1)
    a = 1.0              # Advection speed
    tspan = (0.0, 1.0)   # Time interval
    u0 = initial_condition(N, h)
    p = (a, h)           # Parameters

    # Define problem and solver
    prob = ODEProblem(advection_ode!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=0.01)

    # Plot animation
    x = range(0, L, length=N)
    anim = @animate for u in sol.u
        plot(x, u, ylim=(0, 1.1), label="u(x,t)", xlabel="x", ylabel="u", title="Pulse Advection")
    end
    gif(anim, "pulse_advection.gif", fps=20)
end

main()
