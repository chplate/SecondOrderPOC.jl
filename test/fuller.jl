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
tolerance = 1e-10

#Define solver
solver = Ipopt.Optimizer()
MOI.set(solver, MOI.RawOptimizerAttribute("tol"), tolerance)
MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), verbose)
MOI.set(solver, MOI.RawOptimizerAttribute("max_cpu_time"), maxtime)
MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), maxiter)

### Define system parameters
# Time vector
t0 = 0.0; tf = 1.0

# Cost function matrix
Q = [1.0 0.0;
     0.0 0.0]

# Define initial state
x0 = [0.01, 0.0]

### Define system dynamics
# https://mintoc.de/index.php/Fuller%27s_problem
function nldyn(x)
  f = zeros(eltype(x), 2,2)
  f[1,1] = x[2]
  f[2,1] = 1.0

  f[2,2] = -2.0
  return f
end

function nldyn_deriv(x)
  df = zeros(eltype(x), 2, 2, 2)

  df[:,:,1] = [0.0 1.0;
               0.0 0.0]

  return df
end

# Define control intervals and grid discretization
N = 250
nlin = 2
n_omega = 1
# Terminal state
xN = x0

# Set up problem and solve!
m = pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, N=N, Q=Q, x_terminal=xN, nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m)

@testset "Test status optimal objective value for terminally constrained problem" begin
    objval = getobjval(m)
    status = getstat(m)

    @test string(status) == "LOCALLY_SOLVED"
    @test isapprox(objval, 1.5283092524132405e-5, atol=objtol)
end

@testset "Test Optimal Solution for terminally constrained problem" begin

    wOpt = getomega(m)
    @test isapprox(wOpt[10],  0.999972569499487 ,atol=primaltol)
    @test isapprox(wOpt[30],  0.000461420288494 ,atol=primaltol)
    @test isapprox(wOpt[50],  0.002676573924091 ,atol=primaltol)
    @test isapprox(wOpt[70],  0.458689100798266 ,atol=primaltol)
    @test isapprox(wOpt[90],  0.499767866277453 ,atol=primaltol)
    @test isapprox(wOpt[110], 0.500011711037377 ,atol=primaltol)
    @test isapprox(wOpt[130], 0.499999941374889 ,atol=primaltol)
    @test isapprox(wOpt[150], 0.499981739786975 ,atol=primaltol)
    @test isapprox(wOpt[170], 0.504091137099738 ,atol=primaltol)
    @test isapprox(wOpt[190], 0.880692280756526 ,atol=primaltol)
    @test isapprox(wOpt[210], 0.000483946144084 ,atol=primaltol)
    @test isapprox(wOpt[230], 0.999828515894311 ,atol=primaltol)
    @test isapprox(wOpt[250], 0.999988159272465 ,atol=primaltol)
end

@testset "Simulation and linearization" begin
    x, xpts, J, t = simulate(m)
    x_lin, xpts_lin, J_lin, t_lin = simulatelinearized(m)

    @test norm(J - J_lin)/J < 1e-03
    @test size(xpts) == size(xpts_lin)
    @test norm(xpts - xpts_lin)/norm(xpts) < 1e-03
end



# Set up problem without derivative information and solve!
m_AD = pocproblem(x0, nldyn, n_omega, N=N, Q=Q, x_terminal=xN, nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m_AD)

@testset "Test status and objective for problem with Jacobian calculated by ForwardDiff" begin
    objval = getobjval(m)
    status = getstat(m)

    @test string(status) == "LOCALLY_SOLVED"
    @test isapprox(objval, 1.5283092524132405e-5, atol=objtol)
end

@testset "Test Optimal Solution for problem with jacobian calculated by ForwardDiff" begin

    wOpt = getomega(m)
    @test isapprox(wOpt[10],  0.999972569499487 ,atol=primaltol)
    @test isapprox(wOpt[30],  0.000461420288494 ,atol=primaltol)
    @test isapprox(wOpt[50],  0.002676573924091 ,atol=primaltol)
    @test isapprox(wOpt[70],  0.458689100798266 ,atol=primaltol)
    @test isapprox(wOpt[90],  0.499767866277453 ,atol=primaltol)
    @test isapprox(wOpt[110], 0.500011711037377 ,atol=primaltol)
    @test isapprox(wOpt[130], 0.499999941374889 ,atol=primaltol)
    @test isapprox(wOpt[150], 0.499981739786975 ,atol=primaltol)
    @test isapprox(wOpt[170], 0.504091137099738 ,atol=primaltol)
    @test isapprox(wOpt[190], 0.880692280756526 ,atol=primaltol)
    @test isapprox(wOpt[210], 0.000483946144084 ,atol=primaltol)
    @test isapprox(wOpt[230], 0.999828515894311 ,atol=primaltol)
    @test isapprox(wOpt[250], 0.999988159272465 ,atol=primaltol)
end
