using LinearAlgebra
using Plots

# Parameters
N = 101  # Number of grid points
Dirichlet0 = 0.0
Dirichlet1 = 1.0

h = 1.0 / (N - 1)  # Grid spacing
A = zeros(N, N)
f = zeros(N)

# Construct 1D Laplacian matrix with Dirichlet boundary conditions
for i in 2:N-1
    A[i, i-1] = 1.0 / h^2
    A[i, i] = -2.0 / h^2
    A[i, i+1] = 1.0 / h^2
end

# Apply Dirichlet boundary conditions
A[1,1] = 1.0
A[N,N] = 1.0
f[1] = Dirichlet0
f[N] = Dirichlet1

# Solve linear system
u = A \ f

# Plot solution
x = range(0, 1, length=N)
plot(x, u, label="Numerical Solution", xlabel="x", ylabel="u(x)", title="1D Poisson Equation")
