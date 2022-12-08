using SecondOrderPOC
using MathOptInterface
const MOI = MathOptInterface

using Ipopt#, NLopt, KNITRO

# Define Solver options
maxiter = 300
maxtime = 1000.0
verbose = 5
tolerance = 1e-10

#Define solver
solver = Ipopt.Optimizer()
MOI.set(solver, MOI.RawOptimizerAttribute("tol"), tolerance)
MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), verbose)
MOI.set(solver, MOI.RawOptimizerAttribute("max_cpu_time"), maxtime)
MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), maxiter)
#MOI.set(solver, MOI.RawOptimizerAttribute("hessian_approximation"), "limited-memory")


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
# Set up problem and solve!
xN = x0 # terminal state

m = SecondOrderPOC.pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, N=N, Q=Q, x_terminal=xN, nlin=nlin, t0=t0, tf=tf, solver=solver)
SecondOrderPOC.solve!(m)

println(SecondOrderPOC.getstat(m))
objective = SecondOrderPOC.getobjval(m)
println("Objective: ", objective)

# Plot solution
xsim, ~, J, t = simulate(m)
p1 = plot()
for i =1:size(xsim, 1)
    plot!(t, xsim[i,:], label="x$i")
end
plot!(legendfontsize=10, legend=:topleft)
plot!(grid=true, ylim=[(1-0.2*sign(minimum(xsim)))*minimum(xsim), (1+0.3*sign(maximum(xsim)))*maximum(xsim)])
display(p1)

t_w = collect(range(t0, stop=tf, length=N+1))
omegaopt = reshape(getomega(m), (n_omega, N))
p2 = plot()
for i=1:size(omegaopt,1)
    plot!(t_w,[omegaopt[i,1]; omegaopt[i,:]], label="w$i", linetype=:steppre)
end
plot!(ylim =[-0.1,1.4], grid=true, legendfontsize=10, xguidefontsize=12, xlabel="Time")
display(p2)
