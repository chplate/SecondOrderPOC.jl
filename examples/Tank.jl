using SecondOrderPOC
using MathOptInterface
const MOI = MathOptInterface

using Ipopt#, NLopt, KNITRO

# Define Solver options
maxiter = 25
maxtime = 1000.0
verbose = 5
tolerance = 1e-06

#Define solver
solver = Ipopt.Optimizer()
MOI.set(solver, MOI.RawOptimizerAttribute("tol"), tolerance)
MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), verbose)
MOI.set(solver, MOI.RawOptimizerAttribute("max_cpu_time"), maxtime)
MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), maxiter)
#MOI.set(solver, MOI.RawOptimizerAttribute("hessian_approximation"), "limited-memory")

### Define system parameters
# Time vector
t0 = 0.0; tf = 10.0

# Cost function matrix
C = [0.0 1.0 -1.0]
Q = C'*C

# Define initial state
x0 = [2.0; 2.0; 3.0]

### Define system dynamics
function nldyn(x)
  f = zeros(3,3)

  f[1,1] = -sqrt(x[1])
  f[2,1] = sqrt(x[1]) - sqrt(x[2])
  f[3,1] = -0.05

  f[1,2] = 1.0

  f[1,3] = 2.0
  return f
end

function nldyn_deriv(x)
  df = zeros(3,3,3)

  df[:,:,1] = [-1/(2*sqrt(x[1]))        0               0;
                1/(2*sqrt(x[1]))   -1/(2*sqrt(x[2]))    0;
                0                0                      0]
  return df
end

# Define control intervals and grid discretization
N = 200
nlin = 2
n_omega = 2
# Set up problem and solve!
m = pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, N=N, Q=Q, nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m)

println(getstat(m))
objective = SecondOrderPOC.getobjval(m)
println("Objective: ", objective)

# Plot solution
xsim, ~, J, t = simulate(m)
p1 = plot()
for i =1:size(xsim, 1)
    plot!(t, xsim[i,:], label="x$i")
end
plot!(legendfontsize=10, legend=:topleft, xlabel="Time")
plot!(grid=true, ylim=[(1-0.2*sign(minimum(xsim)))*minimum(xsim), (1+0.2*sign(maximum(xsim)))*maximum(xsim)])
display(p1)

t_w = collect(range(t0, stop=tf, length=N+1))
omegaopt = reshape(getomega(m), (n_omega, N))
p2 = plot()
for i=1:size(omegaopt,1)
    plot!(t_w,[omegaopt[i,1]; omegaopt[i,:]], label="w$i", linetype=:steppre)
end
plot!(ylim =[-0.1,1.1], grid=true, legendfontsize=10, xguidefontsize=12, xlabel="Time")
display(p2)
