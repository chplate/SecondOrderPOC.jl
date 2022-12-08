using SecondOrderPOC
using MathOptInterface
const MOI = MathOptInterface
using Plots
using Ipopt

# Define Solver options
maxiter = 50
maxtime = 1000.0
verbose = 0
tolerance = 1e-06

#Define solver
solver = Ipopt.Optimizer()
MOI.set(solver, MOI.RawOptimizerAttribute("tol"), tolerance)
MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), 5)
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

  df[:,:,4] = [1.0 -1.0 ;
               1.0  1.0 ]
  return df
end

# Since ODE is linear, we can also just use the system matrix A
A = nldyn_deriv(x0)

function c(x::Vector, t::Real)
  c1 = x[1] - 0.4
  return [c1]
end

function dc(x::Vector, t::Real)
  dcdx = [1.0 0.0]
  return dcdx
end

# Define control intervals and grid discretization
N = 200
nlin = 3
n_omega = 3

# Set up bound for box constraint
lb = -Inf * ones(length(x0))
lb[1] = 0.4

# Set up problem with nldyn and nldyn_deriv
m = pocproblem(x0, nldyn, n_omega, nonlin_dyn_deriv=nldyn_deriv, lb=lb, N=N, Q=Q, nlin=nlin, t0=t0, tf=tf, solver=solver)
solve!(m)

# Plot solution
xsim, ~, J, t = simulate(m)
p1 = plot()
for i =1:size(xsim, 1)
    plot!(t, xsim[i,:], label="x$i")
end
Plots.abline!(0,0.4,label="constraint", linestyle=:dash)
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
