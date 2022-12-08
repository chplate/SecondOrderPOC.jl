using SecondOrderPOC
using MathOptInterface
const MOI = MathOptInterface

using Ipopt#, NLopt, KNITRO
using Plots

# Define Solver options
maxiter = 100
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
t0 = 0.0; tf = 12.0

# Cost function matrix
C = [1.0 0.0 -1.0;
     0.0 1.0 -1.0]
Q = C'*C

# Define initial state
x0 = [0.5; 0.7; 1.0]

### Define system dynamics
function nldyn(x)
  f = zeros(eltype(x), 3,2)
  f[1,1] = x[1] - x[1]*x[2]
  f[2,1] = -x[2] + x[1]*x[2]

  f[1,2] = - 0.4*x[1]
  f[2,2] = - 0.2*x[2]
  return f
end

function nldyn_deriv(x)
  df = zeros(eltype(x), 3, 3, 2)
  df[:,:,1] = [1.0-x[2]       -x[1]   0;
               x[2]           -1+x[1] 0;
                0     0     0]

  df[:,:,2] = [-0.4       0     0;
                0       -0.2    0;
                0       0       0]
  return df
end


# Define control intervals and grid discretization
N = 100
nlin = 3
n_omega = 1

# Set up problem and solve!
m = pocproblem(x0, nldyn, n_omega, N=N, Q=Q, nlin=nlin, t0=t0, tf=tf, solver=solver)
SecondOrderPOC.solve!(m)

println("Status: ", getstat(m))
objective = getobjval(m)
println("Objective POC: ", objective)
@info "Solved in $(getsoltime(m)) seconds"


# SUM UP ROUNDING FOR POC
tau, uvec = sum_up_rounding(m)  # yields switching times tau and associated integer controls uvec

# Simulate Sum-Up-Rounding solution
xSUR, ~, J_SUR, t= simulate(m, tau, uvec)
println("Objective SUR: ", J_SUR)

# Plot solution
xsim, ~, J, t = simulate(m)
p1 = plot()
for i =1:size(xsim, 1)
    plot!(t, xsim[i,:], label="x$i")
end
plot!(legendfontsize=10, legend=:topleft, xlabel="Time")
splot!(grid=true, ylim=[(1-0.2*sign(minimum(xsim)))*minimum(xsim), (1+0.2*sign(maximum(xsim)))*maximum(xsim)])
display(p1)

t_w = collect(range(t0, stop=tf, length=N+1))
omegaopt = reshape(getomega(m), (n_omega, N))
p2 = plot()
for i=1:size(omegaopt,1)
    plot!(t_w,[omegaopt[i,1]; omegaopt[i,:]], label="w$i", linetype=:steppre)
end
plot!(ylim =[-0.1,1.1], grid=true, legendfontsize=10, xguidefontsize=12, xlabel="Time")
display(p2)
