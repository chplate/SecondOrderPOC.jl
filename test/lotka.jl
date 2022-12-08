using SecondOrderPOC
using MathOptInterface
const MOI = MathOptInterface
using Ipopt
using LinearAlgebra

# Define test options
objtol = 1e-04
primaltol = 1e-03

# Define Solver options
maxiter = 50
maxtime = 100.0
verbose = 0
tolerance = 1e-06

#Define solver
solver = Ipopt.Optimizer()
MOI.set(solver, MOI.RawOptimizerAttribute("tol"), tolerance)
MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), verbose)
MOI.set(solver, MOI.RawOptimizerAttribute("max_cpu_time"), maxtime)
MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), maxiter)

### Define system parameters
# Time vector
t0 = 0.0; tf = 12.0

# Cost function matrix
C = [1.0 0.0 -1.0;
     0.0 1.0 -1.0]
Q = C'*C

# Define initial state
x0 = [0.5; 0.7; 1.0]

### Define system dynamics
function nldyn(x)
  f = zeros(3,2)
  f[1,1] = x[1] - x[1]*x[2]
  f[2,1] = -x[2] + x[1]*x[2]

  f[1,2] = - 0.4*x[1]
  f[2,2] = - 0.2*x[2]
  return f
end

function nldyn_deriv(x)
  df = zeros(3, 3, 2)
  df[:,:,1] = [1.0-x[2]       -x[1]   0;
               x[2]           -1+x[1] 0;
                0     0     0]

  df[:,:,2] = [-0.4       0     0;
                0       -0.2    0;
                0       0       0]
  return df
end

# Define control intervals and grid discretization
N = 200
nlin = 2
n_omega = 1
# Set up problem and solve!
m = pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, N=N, Q=Q, nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m)

@testset "Test status and optimal objective value" begin
  objval = getobjval(m)
  status = getstat(m)

  @test string(status) == "LOCALLY_SOLVED"
  @test isapprox(objval, 1.34430, atol=objtol)
end

@testset "Test Optimal Solution" begin

  wOpt = getomega(m)

  @test isapprox(wOpt[10], 1.2407066699825483e-6, atol=primaltol)
  @test isapprox(wOpt[30], 4.106651214118706e-6, atol=primaltol)
  @test isapprox(wOpt[50], 0.9999685937556297, atol=primaltol)
  @test isapprox(wOpt[70], 0.5461066538488855, atol=primaltol)
  @test isapprox(wOpt[90], 0.19691524784347186, atol=primaltol)
  @test isapprox(wOpt[110], 0.06390840088516818, atol=primaltol)
  @test isapprox(wOpt[130], 0.020338517517364287, atol=primaltol)
  @test isapprox(wOpt[150], 0.006551330892981961, atol=primaltol)
  @test isapprox(wOpt[170], 0.002744391076987465, atol=primaltol)
  @test isapprox(wOpt[190], 0.0031940768611377563, atol=primaltol)
end

@testset "Simulation and linearization" begin
    x, xpts, J, t = simulate(m)
    x_lin, xpts_lin, J_lin, t_lin = simulatelinearized(m)

    @test norm(J - J_lin)/J < 1e-04
    @test size(xpts) == size(xpts_lin)
    @test norm(xpts - xpts_lin)/norm(xpts) < 1e-03
end
