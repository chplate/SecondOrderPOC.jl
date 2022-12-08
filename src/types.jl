# Empty matrix and vector for function evaluation
const emptyfvec = Array{Float64}(undef, 0)
const emptyfmat = Array{Float64}(undef, 0, 0)
function emptyfunc(args...)  end

# Define Abstract NLP Evaluator for POC problem
abstract type POCev <: MathOptInterface.AbstractNLPEvaluator end
abstract type POC end
mutable struct nlinPOCev <: POCev
    # Parameters
    x0::Array{Float64,1}                     # Initial State x0
    nx::Int64                                # State dimension
    A::Array{Float64,4}                      # Linearized dynamics matrices
    N::Int64                                 # Number of switching times
    n_omega::Int64                           # Number of right hand sides
    t0::Float64                              # Initial Time
    tf::Float64                              # Final Time (from setup)
    Q::Array{Float64,2}                      # State Cost matrix (Mayer)
    E::Array{Float64,2}                      # State Cost matrix (Mayer)

    # Grid Variables
    nlin::Int64                             # Number of grid points
    xi::Float64			                         # Fixed step size for linearization

    # Nonlinear Dynamics and Derivatives Functions
    nonlin_dyn::Function
    nonlin_dyn_deriv::Function
    jacobian_given::Bool       # whether jacobian of dynamics were given by the user
    c::Function                # general nonlinear constraint...
    jacobian_c::Function       # ... and its Jacobian.

    # Precomputed Values
    IndTril::Array{Int, 1}                  # Single Element Indeces of Upper Triangular Matrices (Hessian)
    Itril::Array{Int, 1}                    # Double Element Indeces of Upper Triangular Matrices (Hessian)
    Jtril::Array{Int, 1}                    # Double Element Indeces of Upper Triangular Matrices (Hessian)
    Ag::Array{Float64, 2}                   # Linear Constraint Matrix
    Ig::Array{Int, 1}                       # Index Linear Constraint
    Jg::Array{Int, 1}                       # Index Linear Constraint
    lb::Array{Float64, 1}                   # Constant Term Linear Constraints
    ub::Array{Float64, 1}                   # Constant Term Linear Constraints
    x_terminal::Array{Float64, 1}           # Terminal state for constraint

    # Shared Data Between Functions
    omega_fun_prev::Array{Float64, 1}        # Previous controls for function evaluation
    omega_grad_prev::Array{Float64, 1}       # Previous controls for gradient evaluation
    omega_hess_prev::Array{Float64, 1}       # Previous controls for hessian evaluation
    omega_jac_prev::Array{Float64, 1}
    xpts::Array{Float64, 2}                  # States on complete grid
    expMat::Array{Float64, 4}                # Matrix Exponentials
    Phi::Array{Float64, 4}                   # State Transition Matrices
    M::Array{Float64,4}                      # Lagrange integral Matrices
    S::Array{Float64, 3}                     # Cost-to-go-matrices (combined for Mayer+Lagrange)

    C::Array{Float64, 4}                     # C Matrices for each interval
    D::Array{Float64, 5}                     # D Matrices for each interval
    L::Array{Float64, 4}                     # Derivatives of Lagrange integral
    G::Array{Float64, 5}                     # Derivative of L
    Z::Array{Float64, 5}                     # Z Matrices for each interval
    ZZ::Array{Float64, 6}                    # ZZ Matrices for each interval

    # Store Evaluations
    nobjeval::Int                            # Number of objective function evaluations
    ngradeval::Int                           # Number of gradient evaluations
    nhesseval::Int                           # Number of hessian evaluations
end

mutable struct linPOCev <: POCev
    # Parameters
    x0::Array{Float64,1}                     # Initial State x0
    nx::Int64                                # State dimension
    A::Array{Float64,3}                      # Linear matrices
    N::Int64                                 # Number of switching times
    n_omega::Int64                           # Number of right hand sides
    t0::Float64                              # Initial Time
    tf::Float64                              # Final Time (from setup)
    Q::Array{Float64,2}                      # State Cost matrix (Mayer)
    E::Array{Float64,2}                      # State Cost matrix (Mayer)

    # Grid Variables
    xi::Float64			                     # Fixed step size for linearization
    nlin::Int                               # 1 for linear system
    # Constraint functions
    c::Function                              # general nonlinear constraint...
    jacobian_c::Function                     # ... and its Jacobian.

    # Precomputed Values
    IndTril::Array{Int, 1}                  # Single Element Indeces of Upper Triangular Matrices (Hessian)
    Itril::Array{Int, 1}                    # Double Element Indeces of Upper Triangular Matrices (Hessian)
    Jtril::Array{Int, 1}                    # Double Element Indeces of Upper Triangular Matrices (Hessian)
    Ag::Array{Float64, 2}                   # Linear Constraint Matrix
    Ig::Array{Int, 1}                       # Index Linear Constraint
    Jg::Array{Int, 1}                       # Index Linear Constraint
    lb::Array{Float64, 1}                   # Constant Term Linear Constraints
    ub::Array{Float64, 1}                   # Constant Term Linear Constraints
    x_terminal::Array{Float64, 1}           # Terminal state for constraint

    # Shared Data Between Functions
    omega_fun_prev::Array{Float64, 1}        # Previous controls for function evaluation
    omega_grad_prev::Array{Float64, 1}       # Previous controls for gradient evaluation
    omega_hess_prev::Array{Float64, 1}       # Previous controls for hessian evaluation
    omega_jac_prev::Array{Float64, 1}
    xpts::Array{Float64, 2}                  # States on complete grid
    expMat::Array{Float64, 3}                # Matrix Exponentials
    Phi::Array{Float64, 4}                   # State Transition Matrices
    M::Array{Float64,3}                      # Lagrange integral Matrices
    S::Array{Float64, 3}                     # Cost-to-go-matrices (combined for Mayer+Lagrange)
    C::Array{Float64, 4}                     # C Matrices for each interval
    D::Array{Float64, 5}                     # D Matrices for each interval
    L::Array{Float64, 4}                     # Derivatives of Lagrange integral
    G::Array{Float64, 5}                     # Derivative of L
    Z::Array{Float64, 4}                     # Z Matrices for each interval
    ZZ::Array{Float64, 5}                    # ZZ Matrices for each interval

    # Store Evaluations
    nobjeval::Int                            # Number of objective function evaluations
    ngradeval::Int                           # Number of gradient evaluations
    nhesseval::Int                           # Number of hessian evaluations
end
mutable struct nlinPOC <: POC
    model::MathOptInterface.AbstractOptimizer
    POCev::nlinPOCev                                # NLP Evaluator for nonlinear POC

    # Data Obtained After Optimization
    omega::Array{Float64,1}                     # Optimal combinations of right hand sides
    control::Array{MathOptInterface.VariableIndex,1}
    objval::Float64                             # Optimal Value of Cost Function
    stat::MathOptInterface.TerminationStatusCode# Status of Opt Problem
    soltime::Float64                            # Time Required to solve Opt

    # Inner Contructor for Incomplete Initialization
    nlinPOC(model, POCev, omega, control) = new(model, POCev, omega, control)
end

mutable struct linPOC <: POC
    model::MathOptInterface.AbstractOptimizer
    POCev::linPOCev                                # NLP Evaluator for linear POC

    # Data Obtained After Optimization
    omega::Array{Float64,1}                     # Optimal combinations of right hand sides
    control::Array{MathOptInterface.VariableIndex,1}
    objval::Float64                             # Optimal Value of Cost Function
    stat::MathOptInterface.TerminationStatusCode# Status of Opt Problem
    soltime::Float64                            # Time Required to solve Opt

    # Inner Contructor for Incomplete Initialization
    linPOC(model, POCev, omega, control) = new(model, POCev, omega, control)
end
