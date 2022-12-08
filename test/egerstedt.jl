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
t0 = 0.0; tf = 1.0

# Cost function matrix
Q = [1.0 0.0;
     0.0 1.0]

# Define initial state
x0 = [0.5; 0.5]

### Define system dynamics
function nldyn(x)
  f = zeros(eltype(x), 2,4)

  f[1,2] = -x[1]
  f[2,2] = x[1] + 2*x[2]

  f[1,3] = x[1] + x[2]
  f[2,3] = x[1] - 2*x[2]

  f[1,4] = x[1] - x[2]
  f[2,4] = x[1] + x[2]

  return f
end

function nldyn_deriv(x)
  df = zeros(eltype(x), 2, 2, 4)

  df[:,:,2] = [-1.0  0;
               1.0  2.0]

  df[:,:,3] = [1.0  1.0 ;
               1.0  -2.0]

  df[:,:,4] = [1.0  -1.0;
               1.0   1.0]
  return df
end

# Define control intervals and grid discretization
N = 200
nlin = 2
n_omega = 3
# Set up problem and solve!
m = pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, N=N, Q=Q, nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m)

@testset "Test status optimal objective value for unconstrained problem" begin
    objval = getobjval(m)
    status = getstat(m)

    @test string(status) == "LOCALLY_SOLVED"
    @test isapprox(objval, 0.9891892335661516, atol=objtol)
end

@testset "Simulation and linearization" begin
    x, xpts, J, t = simulate(m)
    x_lin, xpts_lin, J_lin, t_lin = simulatelinearized(m)

    @test norm(J - J_lin)/J < 1e-04
    @test size(xpts) == size(xpts_lin)
    @test norm(xpts - xpts_lin)/norm(xpts) < 1e-03
end

@testset "Test Optimal Solution for unconstrained problem" begin

    wOpt = getomega(m)

    @test isapprox(wOpt[10],  0.9998963160194825  ,atol=primaltol)
    @test isapprox(wOpt[30],  0.0001542718079415263  ,atol=primaltol)
    @test isapprox(wOpt[50],  1.8018436181612885e-5  ,atol=primaltol)
    @test isapprox(wOpt[70],  0.0001588797362919856  ,atol=primaltol)
    @test isapprox(wOpt[90],  0.9998889018048681  ,atol=primaltol)
    @test isapprox(wOpt[110], 4.308983593536599e-5  ,atol=primaltol)
    @test isapprox(wOpt[130], 3.233398716512662e-5  ,atol=primaltol)
    @test isapprox(wOpt[150], 0.9998900478758438  ,atol=primaltol)
    @test isapprox(wOpt[170], 0.00012743067130715585  ,atol=primaltol)
    @test isapprox(wOpt[190], 1.672570452755176e-5  ,atol=primaltol)
    @test isapprox(wOpt[210], 0.999631948309072  ,atol=primaltol)
    @test isapprox(wOpt[230], 0.000863769953432803  ,atol=primaltol)
    @test isapprox(wOpt[250], 1.270627303814143e-5  ,atol=primaltol)
    @test isapprox(wOpt[270], 0.23644483487032328  ,atol=primaltol)
    @test isapprox(wOpt[290], 0.8512817226977722  ,atol=primaltol)
    @test isapprox(wOpt[310], 2.0285118449361976e-5  ,atol=primaltol)
    @test isapprox(wOpt[330], 0.00927477210312532  ,atol=primaltol)
    @test isapprox(wOpt[350], 0.9969445021770226  ,atol=primaltol)
    @test isapprox(wOpt[370], 5.741480956721934e-5  ,atol=primaltol)
    @test isapprox(wOpt[390], 0.00097763033387671  ,atol=primaltol)
    @test isapprox(wOpt[410], 0.9991838531478734  ,atol=primaltol)
    @test isapprox(wOpt[430], 0.0005496885849576588  ,atol=primaltol)
    @test isapprox(wOpt[450], 0.0002413442844427572  ,atol=primaltol)
    @test isapprox(wOpt[470], 0.9534298768010578  ,atol=primaltol)
    @test isapprox(wOpt[490], 0.36372082180431387  ,atol=primaltol)
    @test isapprox(wOpt[510], 0.00021170593137915752  ,atol=primaltol)
    @test isapprox(wOpt[530], 0.6575553069261331  ,atol=primaltol)
    @test isapprox(wOpt[550], 0.3360125766247786  ,atol=primaltol)
    @test isapprox(wOpt[570], 0.000501043805588074  ,atol=primaltol)
    @test isapprox(wOpt[590], 0.6733112419954782  ,atol=primaltol)
end

lb = -Inf * ones(length(x0))
lb[1] = 0.4
m_constrained = pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, N=N, Q=Q, lb=lb, nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m_constrained)

@testset "Test status optimal objective value for constrained problem" begin
    objval = getobjval(m_constrained)
    status = getstat(m_constrained)

    @test string(status) == "LOCALLY_SOLVED"
    @test isapprox(objval, 0.9956075613378692, atol=objtol)
end

@testset "Test Optimal Solution for constrained problem" begin

    wOpt = getomega(m_constrained)

    @test isapprox(wOpt[10],  0.9998773903539714 ,atol=primaltol)
    @test isapprox(wOpt[30],  0.00019703584696618688   ,atol=primaltol)
    @test isapprox(wOpt[50],  2.491975917020186e-5   ,atol=primaltol)
    @test isapprox(wOpt[70],  0.00013733614605827512   ,atol=primaltol)
    @test isapprox(wOpt[90],  0.9998744912345864 ,atol=primaltol)
    @test isapprox(wOpt[110], 8.822743147863508e-5   ,atol=primaltol)
    @test isapprox(wOpt[130], 2.8570404215256884e-5   ,atol=primaltol)
    @test isapprox(wOpt[150], 0.999679451018951 ,atol=primaltol)
    @test isapprox(wOpt[170], 0.002688779645279815   ,atol=primaltol)
    @test isapprox(wOpt[190], 1.800897985543621e-5   ,atol=primaltol)
    @test isapprox(wOpt[210], 0.7215499626330161, atol=primaltol)
    @test isapprox(wOpt[230], 0.2830281752943914   ,atol=primaltol)
    @test isapprox(wOpt[250], 1.840395125177045e-5   ,atol=primaltol)
    @test isapprox(wOpt[270], 0.7090115221382342  ,atol=primaltol)
    @test isapprox(wOpt[290], 0.29517428461274003 ,atol=primaltol)
    @test isapprox(wOpt[310], 2.024644123429827e-5   ,atol=primaltol)
    @test isapprox(wOpt[330], 0.00954440072728195  ,atol=primaltol)
    @test isapprox(wOpt[350], 0.9971083966520119 ,atol=primaltol)
    @test isapprox(wOpt[370], 5.5363446948850114e-5   ,atol=primaltol)
    @test isapprox(wOpt[390], 0.000933855041574296  ,atol=primaltol)
    @test isapprox(wOpt[410], 0.999221922444172 ,atol=primaltol)
    @test isapprox(wOpt[430], 0.0005331757827248685   ,atol=primaltol)
    @test isapprox(wOpt[450], 0.0002269351942527153   ,atol=primaltol)
    @test isapprox(wOpt[470], 0.9500737656948206 ,atol=primaltol)
    @test isapprox(wOpt[490], 0.3526309890202948  ,atol=primaltol)
    @test isapprox(wOpt[510], 0.00020124093546149604   ,atol=primaltol)
    @test isapprox(wOpt[530], 0.6575725991618376 ,atol=primaltol)
    @test isapprox(wOpt[550], 0.3360216223677394 ,atol=primaltol)
    @test isapprox(wOpt[570], 0.00047629984505799934   ,atol=primaltol)
    @test isapprox(wOpt[590], 0.673842182635318 ,atol=primaltol)
end

function c(x, t)
  return [x[1]-0.4]
end

function dc(x, t)
  dcdx = zeros(1,2)
  dcdx[1,1] = 1.0
  return dcdx
end

m_generalConstraint = pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, N=N, Q=Q, c=c, dc=dc, lb_c=[0.0], nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m_generalConstraint)

@testset "Test status optimal objective value for constrained problem in formulation with general constraint" begin
  objval = getobjval(m_generalConstraint)
  status = getstat(m_generalConstraint)

  @test string(status) == "LOCALLY_SOLVED"
  @test isapprox(objval, 0.9956075613378692, atol=objtol)
end

@testset "Test Optimal Solution for constrained problem in formulation with general constraint" begin

  wOpt = getomega(m_generalConstraint)

  @test isapprox(wOpt[10],  0.9998773903539714 ,atol=primaltol)
  @test isapprox(wOpt[30],  0.00019703584696618688   ,atol=primaltol)
  @test isapprox(wOpt[50],  2.491975917020186e-5   ,atol=primaltol)
  @test isapprox(wOpt[70],  0.00013733614605827512   ,atol=primaltol)
  @test isapprox(wOpt[90],  0.9998744912345864 ,atol=primaltol)
  @test isapprox(wOpt[110], 8.822743147863508e-5   ,atol=primaltol)
  @test isapprox(wOpt[130], 2.8570404215256884e-5   ,atol=primaltol)
  @test isapprox(wOpt[150], 0.999679451018951 ,atol=primaltol)
  @test isapprox(wOpt[170], 0.002688779645279815   ,atol=primaltol)
  @test isapprox(wOpt[190], 1.800897985543621e-5   ,atol=primaltol)
  @test isapprox(wOpt[210], 0.7215499626330161, atol=primaltol)
  @test isapprox(wOpt[230], 0.2830281752943914   ,atol=primaltol)
  @test isapprox(wOpt[250], 1.840395125177045e-5   ,atol=primaltol)
  @test isapprox(wOpt[270], 0.7090115221382342  ,atol=primaltol)
  @test isapprox(wOpt[290], 0.29517428461274003 ,atol=primaltol)
  @test isapprox(wOpt[310], 2.024644123429827e-5   ,atol=primaltol)
  @test isapprox(wOpt[330], 0.00954440072728195  ,atol=primaltol)
  @test isapprox(wOpt[350], 0.9971083966520119 ,atol=primaltol)
  @test isapprox(wOpt[370], 5.5363446948850114e-5   ,atol=primaltol)
  @test isapprox(wOpt[390], 0.000933855041574296  ,atol=primaltol)
  @test isapprox(wOpt[410], 0.999221922444172 ,atol=primaltol)
  @test isapprox(wOpt[430], 0.0005331757827248685   ,atol=primaltol)
  @test isapprox(wOpt[450], 0.0002269351942527153   ,atol=primaltol)
  @test isapprox(wOpt[470], 0.9500737656948206 ,atol=primaltol)
  @test isapprox(wOpt[490], 0.3526309890202948  ,atol=primaltol)
  @test isapprox(wOpt[510], 0.00020124093546149604   ,atol=primaltol)
  @test isapprox(wOpt[530], 0.6575725991618376 ,atol=primaltol)
  @test isapprox(wOpt[550], 0.3360216223677394 ,atol=primaltol)
  @test isapprox(wOpt[570], 0.00047629984505799934   ,atol=primaltol)
  @test isapprox(wOpt[590], 0.673842182635318 ,atol=primaltol)
end

# Set up problem without derivative information, has to be calculated by ForwardDiff
m_generalConstraint_AD = pocproblem(x0, nldyn, n_omega, N=N, Q=Q, c=c, lb_c=[0.0], nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m_generalConstraint_AD)


@testset "Status & objective for constrained problem with general constraint and AD derivatives" begin
    objval = getobjval(m_generalConstraint_AD)
    status = getstat(m_generalConstraint_AD)

    @test string(status) == "LOCALLY_SOLVED"
    @test isapprox(objval, 0.9956075613378692, atol=objtol)
end


@testset "Optimal Solution for constrained problem with general constraint and AD derivatives" begin

    wOpt = getomega(m_generalConstraint_AD)

    @test isapprox(wOpt[10],  0.9998773903539714 ,atol=primaltol)
    @test isapprox(wOpt[30],  0.00019703584696618688   ,atol=primaltol)
    @test isapprox(wOpt[50],  2.491975917020186e-5   ,atol=primaltol)
    @test isapprox(wOpt[70],  0.00013733614605827512   ,atol=primaltol)
    @test isapprox(wOpt[90],  0.9998744912345864 ,atol=primaltol)
    @test isapprox(wOpt[110], 8.822743147863508e-5   ,atol=primaltol)
    @test isapprox(wOpt[130], 2.8570404215256884e-5   ,atol=primaltol)
    @test isapprox(wOpt[150], 0.999679451018951 ,atol=primaltol)
    @test isapprox(wOpt[170], 0.002688779645279815   ,atol=primaltol)
    @test isapprox(wOpt[190], 1.800897985543621e-5   ,atol=primaltol)
    @test isapprox(wOpt[210], 0.7215499626330161, atol=primaltol)
    @test isapprox(wOpt[230], 0.2830281752943914   ,atol=primaltol)
    @test isapprox(wOpt[250], 1.840395125177045e-5   ,atol=primaltol)
    @test isapprox(wOpt[270], 0.7090115221382342  ,atol=primaltol)
    @test isapprox(wOpt[290], 0.29517428461274003 ,atol=primaltol)
    @test isapprox(wOpt[310], 2.024644123429827e-5   ,atol=primaltol)
    @test isapprox(wOpt[330], 0.00954440072728195  ,atol=primaltol)
    @test isapprox(wOpt[350], 0.9971083966520119 ,atol=primaltol)
    @test isapprox(wOpt[370], 5.5363446948850114e-5   ,atol=primaltol)
    @test isapprox(wOpt[390], 0.000933855041574296  ,atol=primaltol)
    @test isapprox(wOpt[410], 0.999221922444172 ,atol=primaltol)
    @test isapprox(wOpt[430], 0.0005331757827248685   ,atol=primaltol)
    @test isapprox(wOpt[450], 0.0002269351942527153   ,atol=primaltol)
    @test isapprox(wOpt[470], 0.9500737656948206 ,atol=primaltol)
    @test isapprox(wOpt[490], 0.3526309890202948  ,atol=primaltol)
    @test isapprox(wOpt[510], 0.00020124093546149604   ,atol=primaltol)
    @test isapprox(wOpt[530], 0.6575725991618376 ,atol=primaltol)
    @test isapprox(wOpt[550], 0.3360216223677394 ,atol=primaltol)
    @test isapprox(wOpt[570], 0.00047629984505799934   ,atol=primaltol)
    @test isapprox(wOpt[590], 0.673842182635318 ,atol=primaltol)
end
