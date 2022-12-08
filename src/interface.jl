# Create  linear POC Problem
function pocproblem(
    x0::Array{Float64,1},             # Initial State
    A::Array{Float64,3};              # Linear Dynamics
    N::Int64=10,		              # Number of control intervals
    nlin::Int64=1,                   # Number of intermediate steps on each control interval for approximation of derivative of Lagrange term
    t0::Float64=0.0,                  # Initial Time
    tf::Float64=1.0,                  # Final Time
    solver::MathOptInterface.AbstractOptimizer=Ipopt.Optimizer(), #Default Solver
    Q::Array{Float64,2}=emptyfmat,    # Lagrange Cost Matrix
    E::Array{Float64, 2}=emptyfmat,   # Mayer Cost Matrix
    l::Array{Float64, 1}=emptyfvec,   # Lower Bound on controls
    u::Array{Float64, 1}=emptyfvec,   # Upper Bound on controls
    lb::Array{Float64, 1}=emptyfvec,  # Lower Bound on states at the N points on coarse grid
    ub::Array{Float64, 1}=emptyfvec,  # Upper Bound on states at the N points on coarse grid
    c::Function=emptyfunc,              # general constraint c(x)
    dc::Function=emptyfunc,             # derivative of c(x) w.r.t. x
    lb_c::Array{Float64, 1}=emptyfvec,  # Lower Bound for c(x) at the N points on coarse grid
    ub_c::Array{Float64, 1}=emptyfvec,  # Upper Bound for c(x) at the N points on coarse grid
    x_terminal::Array{Float64, 1}=emptyfvec,  # Terminal state for terminal constraint
    omega0ws::Array{Float64,1}=emptyfvec, # Warm Start omega
    )

    # Get Dimensions
    n_omega = size(A)[3] - 1            # Counting all right hand sides minus common f0
    nx = length(x0)                     # State Dimension
    numVar = N * n_omega                # Degrees of freedom of Optimization problem
    xi = (tf - t0)/(N)            # fixed step size for integration

    # Adjust variables which have not been initalized
    if isempty(E)
        E = zeros(nx,nx)
    end

    if isempty(Q)
        Q = Matrix{Float64}(I, nx, nx)
    end

    if isempty(l)
        l = zeros(N*n_omega)        # Controls larger than zero
    end

    if isempty(u)
        u = ones(N*n_omega)         # Controls smaller than one
    end

    if isempty(lb) && !isempty(ub)
        @assert minimum(x0.<=ub) == 1.0 "Initial value violates upper box constraint"
        lb = -Inf*ones(nx)
    end

    if isempty(ub) && !isempty(lb)
        @assert minimum(x0.>=lb) == 1.0 "Initial value violates lower box constraint"
        ub = Inf*ones(nx)
    end

    if isempty(omega0ws)
        # initialize omegas with same values
        omega0ws = n_omega > 1 ? (1/n_omega)*ones(n_omega*N) : 0.5*ones(n_omega*N)
    end

    numConstraintsC = isnothing(c(x0,t0)) ? 0 : length(c(x0,t0))
    jacobian_c(x,t) = isnothing(dc(x0,t0)) && numConstraintsC > 0 ? ForwardDiff.jacobian(x -> c(x,t), x) : dc(x,t)
    if !isnothing(jacobian_c(x0,t0))
        sizeDc = size(jacobian_c(x0,t0))
        @assert sizeDc == (numConstraintsC,nx) "Jacobian of constraint must be of size ($numConstraintsC,$nx), yet given dimensions are $sizeDc."
    end

    if !isempty(lb_c) && isempty(ub_c)
        ub_c = Inf * ones(length(lb_c))
    elseif isempty(lb_c) && !isempty(ub_c)
        lb_c = -Inf * ones(length(ub_c))
    end

    ### Initialize NLP Evaluator
    @assert size(Q) == (nx,nx) && size(E) == (nx,nx) "Cost function matrices don't have correct sizes."

    # Define Required Matrices
    C = Array{Float64}(undef, nx, nx, N, n_omega)
    D = Array{Float64}(undef, nx, nx, N, n_omega, n_omega)
    L = Array{Float64}(undef, nx, nx, N, n_omega)
    G = Array{Float64}(undef, nx, nx, N, n_omega, n_omega)

    M = Array{Float64}(undef, nx, nx, N)
    S = Array{Float64}(undef, nx, nx, N+1)
    expMat = Array{Float64}(undef, nx, nx, N)
    Phi = Array{Float64}(undef, nx, nx, N+1, N+1)
    Z = Array{Float64}(undef, nx, nx, N, n_omega)
    ZZ = Array{Float64}(undef, nx, nx, N, n_omega, n_omega)

    # Define Arrays for saving previous iterates when evaluating functions
    omega_fun_prev = Array{Float64}(undef, numVar)
    omega_grad_prev = Array{Float64}(undef, numVar)
    omega_hess_prev = Array{Float64}(undef, numVar)
    omega_jac_prev = Array{Float64}(undef, numVar)

    # Initialize State trajectory
    xpts = Array{Float64}(undef, nx, N+1)
    xpts[:, 1] = x0

    # Get sparsity structure of hessian and constraints
    IndTril, Itril, Jtril = get_sparsity_of_hessian(N, n_omega)

    boxConstrainedStates = (lb.!=-Inf) .| (ub .!= Inf)
    numBoxConstrainedStates = sum(boxConstrainedStates)

    constrainedTerminalStates = (x_terminal .!=-Inf) .& (x_terminal .!= Inf)
    numConstrainedTerminalStates = sum(constrainedTerminalStates)

    Ag, Ig, Jg = get_sparsity_of_constraints(N, n_omega, numBoxConstrainedStates, numConstraintsC, numConstrainedTerminalStates)

    # Initialize objective evaluator
    nobjeval = 0                                          # Number of objective function evaluations
    ngradeval = 0                                         # Number of gradient evaluations
    nhesseval = 0                                         # Number of hessian evaluations

    # Construct NLPEvaluator
    POCev = linPOCev(x0, nx, A, N, n_omega, t0, tf, Q, E, xi, nlin, c, jacobian_c, IndTril, Itril, Jtril,
                        Ag, Ig, Jg, lb, ub, x_terminal, omega_fun_prev, omega_grad_prev, omega_hess_prev,  omega_jac_prev,
                        xpts, expMat, Phi, M, S, C, D, L, G, Z, ZZ, nobjeval, ngradeval, nhesseval)

    # Propagate Dynamics to compute matrix exponentials and states
    propagate_dynamics!(POCev, omega0ws)

    model = solver
    #MathOptInterface.empty!(model)
    omega = MathOptInterface.add_variables(model, numVar)
    for w in omega # Add bounds for controls
        MathOptInterface.add_constraint(model, w, MathOptInterface.GreaterThan(0.0))
        MathOptInterface.add_constraint(model, w, MathOptInterface.LessThan(1.0))
    end

    # Get bounds for N OneHot constraints -> sum over w_ik = 1 for each interval i in [N]
    # If only one right side -> Omit one-hot condition (bounds -Inf and Inf)
    lbOneHot = n_omega == 1 ? -Inf*ones(N) : ones(N)
    ubOneHot = n_omega == 1 ?  Inf*ones(N) : ones(N)

    # Add constraints to model
    lbub = Array{MathOptInterface.NLPBoundsPair}(undef, 0)
    lbubOneHot = [MathOptInterface.NLPBoundsPair(lbOneHot[i], ubOneHot[i]) for i=1:N]     # One Hot constraints
    append!(lbub, lbubOneHot)
    lbubBoxConstr = similar(lbubOneHot)
    if !isempty(lb) || !isempty(ub)
        # Box constraints (inequality constraint with lower and upper bounds)
        lbubBoxConstr = [MathOptInterface.NLPBoundsPair(lb[i], ub[i]) for j=1:N for i in findall(>(0), boxConstrainedStates)]
        append!(lbub, lbubBoxConstr)
    end
    lbubGeneralConstr = similar(lbubOneHot)
    if numConstraintsC > 0
        # General constraints (inequality or equality constraint, depending on lower and upper bounds)
        lbubGeneralConstr = [MathOptInterface.NLPBoundsPair(lb_c[i], ub_c[i]) for j=1:N for i=1:numConstraintsC]
        append!(lbub, lbubGeneralConstr)
    end
    lbubTerminalConstraint = similar(lbubOneHot)
    if numConstrainedTerminalStates > 0
        # Terminal constraint (equality constraint)
        lbubTerminalConstraint = [MathOptInterface.NLPBoundsPair(x_terminal[i], x_terminal[i]) for i in findall(>(0), constrainedTerminalStates)]
        append!(lbub, lbubTerminalConstraint)
    end

    MathOptInterface.set(model, MathOptInterface.VariablePrimalStart(), omega, omega0ws)
    block_data = MathOptInterface.NLPBlockData(
        lbub,
        POCev,
        true
    )
    MathOptInterface.set(model, MathOptInterface.NLPBlock(), block_data)
    MathOptInterface.set(model, MathOptInterface.ObjectiveSense(), MathOptInterface.MIN_SENSE)

    # Create POC
    POCproblem = linPOC(model, POCev, omega0ws, omega)

    return POCproblem  # Return POC
end


# Create  Nonlinear POC Problem
function pocproblem(
    x0::Array{Float64,1},             # Initial State
    nonlin_dyn::Function,             # Nonlinear Dynamics
    n_omega::Int;                     # Number of modes to take into account
    nonlin_dyn_deriv::Function=emptyfunc,# Jacobian of nonlinear dynamics
    N::Int64=10,		                # Number of control intervals
    nlin::Int64=1,                   # Number of Linearization points on each interval in [N]
    t0::Float64=0.0,                  # Initial Time
    tf::Float64=1.0,                  # Final Time
    solver::MathOptInterface.AbstractOptimizer=Ipopt.Optimizer(), #Default Solver
    Q::Array{Float64,2}=emptyfmat,      # Lagrange Cost Matrix
    E::Array{Float64,2}=emptyfmat,      # Mayer Cost Matrix
    l::Array{Float64,1}=emptyfvec,      # Lower Bound on controls
    u::Array{Float64,1}=emptyfvec,      # Upper Bound on controls
    lb::Array{Float64,1}=emptyfvec,     # Lower Bound on states at the N points on coarse grid
    ub::Array{Float64,1}=emptyfvec,     # Upper Bound on states at the N points on coarse grid
    c::Function=emptyfunc,              # general constraint c(x)
    dc::Function=emptyfunc,             # derivative of c(x) w.r.t. x
    lb_c::Array{Float64, 1}=emptyfvec,  # Lower Bound for c(x) at the N points on coarse grid
    ub_c::Array{Float64, 1}=emptyfvec,  # Upper Bound for c(x) at the N points on coarse grid
    x_terminal::Array{Float64, 1}=emptyfvec,  # Terminal state for terminal constraint
    omega0ws::Array{Float64,1}=emptyfvec, # Warm Start omega
    )

    jacobian_given = !isnothing(nonlin_dyn_deriv(x0))

    # Get Dimensions
    nx = length(x0) + 1        # State Dimension (directly add 1 due to linearization)
    numVar = N * n_omega       # Degrees of freedom of Optimization problem
    xi = (tf - t0)/(N*nlin)   # fixed step size for integration

    jac(x) = jacobian_given ? nonlin_dyn_deriv(x) : permutedims(reshape(transpose(ForwardDiff.jacobian(nonlin_dyn, x)), (nx-1, nx-1, 1+n_omega)), (2,1,3))
        # Adjust variables which have not been initalized
    if isempty(E)
        E = zeros(nx,nx)
    end

    if isempty(Q)
        Q = Matrix{Float64}(I, nx, nx)
    end

    if isempty(l)
        l = zeros(N*n_omega)        # Controls larger than zero
    end

    if isempty(u)
        u = ones(N*n_omega)         # Controls smaller than one
    end

    if isempty(lb) && !isempty(ub)
        @assert minimum(x0.<=ub) == 1.0 "Initial value violates upper box constraint"
        lb = -Inf*ones(nx-1)
    end

    if isempty(ub) && !isempty(lb)
        @assert minimum(x0.>=lb) == 1.0 "Initial value violates lower box constraint"
        ub = Inf*ones(nx-1)
    end

    if isempty(omega0ws)
        # initialize omegas with same values
        omega0ws = n_omega > 1 ? (1/n_omega)*ones(n_omega*N) : 0.5*ones(n_omega*N)
    end

    numConstraintsC = isnothing(c(x0, t0)) ? 0 : length(c(x0,t0))
    jacobian_c(x,t) = isnothing(dc(x0,t0)) && numConstraintsC > 0 ? ForwardDiff.jacobian(x -> c(x,t), x) : dc(x,t)
    if !isnothing(jacobian_c(x0,t0))
        sizeDc = size(jacobian_c(x0,t0))
        @assert sizeDc == (numConstraintsC,nx-1) "Jacobian of constraint must be of size ($numConstraintsC,$(nx-1), yet given dimensions are $sizeDc."
    end

    if !isempty(lb_c) && isempty(ub_c)
        ub_c = Inf * ones(length(lb_c))
    elseif isempty(lb_c) && !isempty(ub_c)
        lb_c = -Inf * ones(length(ub_c))
    end

    ### Initialize NLP Evaluator

    # Extend Initial State and Cost Matrix
    x0 = [x0; 1]
    if size(E) != (nx,nx)
        E = [E zeros(size(E,1),1); zeros(1,size(E,2)+1)]
    end
    if size(Q) != (nx,nx)
        Q = [Q zeros(size(Q,1),1); zeros(1,size(Q,2)+1)]
    end
    @assert size(Q) == (nx,nx) && size(E) == (nx,nx) "Cost function matrices don't have correct size."

    # Define Required Matrices
    A = Array{Float64}(undef, nx, nx, N, nlin)
    C = Array{Float64}(undef, nx, nx, N, n_omega)
    D = Array{Float64}(undef, nx, nx, N, n_omega, n_omega)
    L = Array{Float64}(undef, nx, nx, N, n_omega)
    G = Array{Float64}(undef, nx, nx, N, n_omega, n_omega)

    M = Array{Float64}(undef, nx, nx, N, nlin)
    S = Array{Float64}(undef, nx, nx, N+1)
    expMat = Array{Float64}(undef, nx, nx, N, nlin)
    Phi = Array{Float64}(undef, nx, nx, N+1, N+1)
    Z = Array{Float64}(undef, nx, nx, N, nlin, n_omega)
    ZZ = Array{Float64}(undef, nx, nx, N, nlin, n_omega, n_omega)

    # Define Arrays for saving previous iterates when evaluating functions
    omega_fun_prev = Array{Float64}(undef, numVar)
    omega_grad_prev = Array{Float64}(undef, numVar)
    omega_hess_prev = Array{Float64}(undef, numVar)
    omega_jac_prev = Array{Float64}(undef, numVar)

    # Initialize State trajectory
    xpts = Array{Float64}(undef, nx, N*nlin+1)
    xpts[:, 1] = x0
    xpts[end,:] .= 1.0

    # Get sparsity structure of hessian and constraints
    IndTril, Itril, Jtril = get_sparsity_of_hessian(N, n_omega)

    boxConstrainedStates = (lb.!=-Inf) .| (ub .!= Inf)
    numBoxConstrainedStates = sum(boxConstrainedStates)

    constrainedTerminalStates = (x_terminal .!=-Inf) .& (x_terminal .!= Inf)
    numConstrainedTerminalStates = sum(constrainedTerminalStates)

    Ag, Ig, Jg = get_sparsity_of_constraints(N, n_omega, numBoxConstrainedStates, numConstraintsC, numConstrainedTerminalStates)

    # Initialize objective evaluator
    nobjeval = 0                                          # Number of objective function evaluations
    ngradeval = 0                                         # Number of gradient evaluations
    nhesseval = 0                                         # Number of hessian evaluations

    # Construct NLPEvaluator
    POCev = nlinPOCev(x0, nx, A, N, n_omega, t0, tf, Q, E, nlin, xi, nonlin_dyn, jac, jacobian_given, c, jacobian_c,
                        IndTril, Itril, Jtril, Ag, Ig, Jg, lb, ub, x_terminal, omega_fun_prev, omega_grad_prev, omega_hess_prev,
                        omega_jac_prev, xpts, expMat, Phi, M, S, C, D, L, G, Z, ZZ, nobjeval, ngradeval, nhesseval)

    # Propagate Dynamics to compute matrix exponentials and states
    propagate_dynamics!(POCev, omega0ws)

    model = solver
    #MathOptInterface.empty!(model)
    omega = MathOptInterface.add_variables(model, numVar)
    for w in omega # Add bounds for controls
        MathOptInterface.add_constraint(model, w, MathOptInterface.GreaterThan(0.0))
        MathOptInterface.add_constraint(model, w, MathOptInterface.LessThan(1.0))
    end

    # Get bounds for N OneHot constraints -> sum over w_ik = 1 for each interval i in [N]
    # If only one right side -> Omit one-hot condition (bounds -Inf and Inf)
    lbOneHot = n_omega == 1 ? -Inf*ones(N) : ones(N)
    ubOneHot = n_omega == 1 ?  Inf*ones(N) : ones(N)

    # Add constraints to model
    lbub = Array{MathOptInterface.NLPBoundsPair}(undef, 0)
    lbubOneHot = [MathOptInterface.NLPBoundsPair(lbOneHot[i], ubOneHot[i]) for i=1:N]     # One Hot constraints
    append!(lbub, lbubOneHot)
    lbubBoxConstr = similar(lbubOneHot)
    if !isempty(lb) || !isempty(ub)
        # Box constraints (inequality constraint with lower and upper bounds)
        lbubBoxConstr = [MathOptInterface.NLPBoundsPair(lb[i], ub[i]) for j=1:N for i in findall(>(0), boxConstrainedStates)]
        append!(lbub, lbubBoxConstr)
    end
    lbubGeneralConstr = similar(lbubOneHot)
    if numConstraintsC > 0
        # General constraints (inequality or equality constraint, depending on lower and upper bounds)
        lbubGeneralConstr = [MathOptInterface.NLPBoundsPair(lb_c[i], ub_c[i]) for j=1:N for i=1:numConstraintsC]
        append!(lbub, lbubGeneralConstr)
    end
    lbubTerminalConstraint = similar(lbubOneHot)
    if numConstrainedTerminalStates > 0
        # Terminal constraint (equality constraint)
        lbubTerminalConstraint = [MathOptInterface.NLPBoundsPair(x_terminal[i], x_terminal[i]) for i in findall(>(0), constrainedTerminalStates)]
        append!(lbub, lbubTerminalConstraint)
    end

    MathOptInterface.set(model, MathOptInterface.VariablePrimalStart(), omega, omega0ws)
    block_data = MathOptInterface.NLPBlockData(
        lbub,
        POCev,
        true
    )
    MathOptInterface.set(model, MathOptInterface.NLPBlock(), block_data)
    MathOptInterface.set(model, MathOptInterface.ObjectiveSense(), MathOptInterface.MIN_SENSE)

    # Create POC
    POCproblem = nlinPOC(model, POCev, omega0ws, omega)

    return POCproblem  # Return POC
end


# Solve Optimization for Linear System
function solve!(m::POC)

    # Perform Optimization
    m.soltime = @elapsed MathOptInterface.optimize!(m.model)
    m.stat = MathOptInterface.get(m.model, MathOptInterface.TerminationStatus())
    m.omega = MathOptInterface.get(m.model, MathOptInterface.VariablePrimal(), m.control)
    m.objval = MathOptInterface.get(m.model, MathOptInterface.ObjectiveValue())
end

"Set warm starting point"
function setwarmstart!(m::POC, omega0ws::Array{Float64,1})
    # Set Warm Starting Point for Nonlinear Solver
    MathOptInterface.set(m.model, MathOptInterface.VariablePrimalStart(), m.control, omega0ws)
    nothing
end

"Set initial state x0"
function setx0!(m::POC, x0::Array{Float64,1})
    # Define initial point
    m.POCev.x0 = x0
    nothing
end

# Return Variables from POC
getomega(m::POC) = m.omega
getobjval(m::POC) = m.objval
getstat(m::POC) = m.stat
getsoltime(m::POC) = m.soltime
getnobjeval(m::POC) = m.POCev.nobjeval
getngradeval(m::POC) = m.POCev.ngradeval
getnhesseval(m::POC) = m.POCev.nhesseval


"Constructs sparsity structure of the hessian of the lagrangian."
function get_sparsity_of_hessian(N::Integer, n_omega::Integer)

    # Construct Matrix of Indeces for upper triangular Matrix (Hessian)
    # Indtril contains LinearIndices of nonzero elements in upper triangular matrix, i.e
    # if Indtril contains index 1 -> later Hessian[1,1] is retrieved. Matrices are stored column wise in Julia
    IndTril = (LinearIndices(triu(ones(N*n_omega, N*n_omega))))[findall(!iszero,triu(ones(N*n_omega, N*n_omega)))]

    # Itril and Jtril contain indices of rows and columns of nonzero elements of upper triangular matrix respectively.
    Itril, Jtril = begin
      I_temp = findall(!iszero, triu(ones(N*n_omega, N*n_omega)))
      (getindex.(I_temp, 1), getindex.(I_temp, 2))
    end

    return IndTril, Itril, Jtril
end

"Get sparsity structure of Jacobian of constraints."
function get_sparsity_of_constraints(N, n_omega, numBoxConstrainedStates, numConstraintsC, numConstrainedTerminalStates)

    # Construct matrix for sparsity structure of Hessian of Lagrangian
    # Construct Constraints matrices for sparsity structure of Jacobian of Constraints
    Ag = zeros(N, N*n_omega)
    for i=1:N
      for j=1:n_omega
        Ag[i,(i-1)*n_omega + j] = 1   # Ag is constructed so that it sums the w_ik over each interval on each control interval i
      end
    end

    # Check for box constraints
    A_BoxConstr = zeros(N*numBoxConstrainedStates, N*n_omega)
    if numBoxConstrainedStates > 0
      for i=1:N
        for ii=i:N
          A_BoxConstr[(ii-1)*numBoxConstrainedStates+1:ii*numBoxConstrainedStates, (i-1)*n_omega+1:i*n_omega] .= 1.0
        end
      end
    end

    A_GeneralConstraint = zeros(N*numConstraintsC,N*n_omega)
    if numConstraintsC > 0
      for i=1:N
        for ii=1:i
          A_GeneralConstraint[(i-1)*numConstraintsC+1:i*numConstraintsC, (ii-1)*n_omega+1:ii*n_omega] .= 1.0
        end
      end
    end

    # Terminal constraint
    A_TerminalConstraint = ones(numConstrainedTerminalStates, N*n_omega)

    ConstraintMatrix = [Ag; A_BoxConstr; A_GeneralConstraint; A_TerminalConstraint]
    # Ig and Jg contain indices of of rows and columns of nonzero elements of Jacobian of the constraints
    # Ig and Jg contain indices for both constraints, OneHot as well as Box
    Ig, Jg = begin
      I_temp = findall(!iszero, ConstraintMatrix)
      (getindex.(I_temp, 1), getindex.(I_temp, 2))
    end

    return Ag, Ig, Jg
end
