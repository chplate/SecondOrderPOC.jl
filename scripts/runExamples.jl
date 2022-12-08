using SecondOrderPOC
using SwitchTimeOpt
include("getExamples.jl")
using Ipopt
using CSV, Dates
using Printf
using MathOptInterface
using LinearAlgebra
const MOI = MathOptInterface
using ArgParse


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--poc"
            help = "Generate results for POC formulation"
            arg_type = Bool
            default = true
        "--sto"
            help = "Generate results for STO formulation"
            arg_type = Bool
            default = false
        "--poc_sto"
            help = "Warmstart STO with SUR solution (only order of modes)."
            arg_type = Bool
            default = false
        "--poc_stoWS"
            help = "Warmstart STO with SUR solution (order of modes & switching times)"
            arg_type = Bool
            default = false
        "--poc_stoFair"
            help = "Warmstart STO with SUR solution with adjusted number of switches.
                    Number of switches in this case are the number of switches needed to
                    replicate order of modes by SUR when naively alternating between the modes."
            arg_type = Bool
            default = false
        "--problems"
            nargs = '*'
            help = "List of problems to solve"
            default = [3,4,5]
        "--tolerance"
            help = "Tolerance of solver"
            arg_type = Float64
            default = 1e-6
        "--maxIter"
            help = "Maximum number of iterations for solver"
            arg_type = Int
            default = 50
        "--maxTime"
            help = "Maximum amount of time for solver"
            arg_type = Float64
            default = 1000.0
        "--bfgs"
            help = "Shall BFGS updates be used instead of exact Hessians?"
            arg_type = Bool
            default = false
        "--boxConstraints"
            help = "Shall box constraints be taken into account when solving problems in POC formulation?"
            arg_type = Bool
            default = false
        "--N"
            nargs = '*'
            help = "Vector of values for N"
            default = collect(5:5:15)
        "--ngrid"
            nargs = '*'
            help = "Number of linearization points on complete horizon"
            default = collect(100:100:500)
        "--nlin"
            nargs = '*'
            help = "Number of linearization points on each interval (for POC)."
            default = collect(1:2:9)
        "--use_ngrid"
            arg_type = Bool
            help = "If true, ngrid is used for computation of nlin instead of using nlin directly."
            default = true
        "--parallel_worker_ID"
            help = "Instance of worker in case of parallel computing"
            default = 0

    end
    return parse_args(s)
end

args = parse_commandline()

@info args

instanceID = args["parallel_worker_ID"]
verbose = 0

filepath    = dirname(@__FILE__)
today       = string(Dates.today())
currentTime = string(Dates.format(now(), "HH-MM-SS"))
savedir = joinpath(filepath, "results", today, currentTime)
mkpath(savedir)
results_file = joinpath(savedir, "results_$instanceID.csv")

# Save options to CSV
CSV.write(joinpath(savedir, "options_$instanceID.csv"), args)


iteration = 0
function my_callback(
    alg_mod::Cint,
    iter_count::Cint,
    obj_value::Float64,
    inf_pr::Float64,
    inf_du::Float64,
    mu::Float64,
    d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64,
    alpha_pr::Float64,
    ls_trials::Cint,
)
    global iteration = iter_count
    return true
end


solver = Ipopt.Optimizer()
MOI.set(solver, MOI.RawOptimizerAttribute("tol"), args["tolerance"])
MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), verbose)
MOI.set(solver, MOI.RawOptimizerAttribute("max_cpu_time"), args["maxTime"])
MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), args["maxIter"])
if args["bfgs"]
    MOI.set(solver, MOI.RawOptimizerAttribute("hessian_approximation"), "limited-memory")
end

header = "ID,problem,formulation,bfgs,N,nlin,status,objValue,time,iterations,discretization\n"
# Write header to results
open(results_file,"w") do f
    write(f, header)
end

function readd_callback!(model)
    global iteration = 0
    MOI.empty!(model)
    MOI.set(model, Ipopt.CallbackFunction(), my_callback)
end


counter = 0
for index  in args["problems"]
    index = isa(index, String) ? tryparse(Int, index) : index
    name, t0, tf, x0, Q, f, lb, ub, n_omega = getProblem(index)
    @info "Solving problem $name with $n_omega controls."
    if !args["boxConstraints"]
        lb, ub = SecondOrderPOC.emptyfvec, SecondOrderPOC.emptyfvec
    end
    readd_callback!(solver)
    # Initialize m with default values and solve for few iterations, so that first instance doesnt take longer than the others
    MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), 5)
    _m = SecondOrderPOC.pocproblem(x0, f, n_omega, solver=solver)
    SecondOrderPOC.solve!(_m)
    MOI.set(solver, MOI.RawOptimizerAttribute("max_iter"), args["maxIter"])

    num_linearization_points = args["use_ngrid"] ? args["ngrid"] : args["nlin"]

    for num_lin in num_linearization_points
        for N in args["N"]
            num_lin = isa(num_lin, String) ? tryparse(Int, num_lin) : num_lin
            N = isa(N, String) ? tryparse(Int, N) : N
            discretization = args["use_ngrid"] ? num_lin : N * num_lin

            #global iteration = 0
            global result_row_csv = ""
            uniqueID = string(instanceID)*"_"*string(counter)
            global counter+=1
            try
                # Solve problem in POC formulation and round solution via SUR
                if args["poc"]
                    _nlin = args["use_ngrid"] ? Int(round(num_lin/N)) : num_lin
                    _nlin = _nlin == 0 ? 1 : _nlin
                    discretization = args["use_ngrid"] ? N*_nlin : N*num_lin
                    readd_callback!(solver)

                    m = SecondOrderPOC.pocproblem(x0, f, n_omega, N=N, Q=Q, lb=lb, ub=ub, nlin=_nlin, t0=t0, tf=tf, solver=solver)
                    SecondOrderPOC.solve!(m)
                    status = SecondOrderPOC.getstat(m)
                    _objective = SecondOrderPOC.getobjval(m)
                    time = SecondOrderPOC.getsoltime(m)
                    objective = @sprintf("%0.5e", _objective)
                    result_row_csv = string(result_row_csv, uniqueID, ",", name, ",", "poc", ",$(args["bfgs"]),", N, ",",_nlin, ",", string(status), ",", objective, ",", time,  ",",iteration, ",", discretization, "\n")

                    # SumUpRounding
                    timeSUR = @elapsed tauSUR, uvec = SecondOrderPOC.sum_up_rounding(m)
                    ~, ~, _objectiveSUR, ~ = SecondOrderPOC.simulate(m, tauSUR, uvec)
                    objectiveSUR = @sprintf("%0.5e", _objectiveSUR)
                    N_SUR = size(uvec, 2)
                    result_row_csv = string(result_row_csv, uniqueID, ",", name, ",", "SUR", ",$(args["bfgs"]),", N_SUR, ",",N*_nlin, ",", string(status), ",", objectiveSUR, ",", timeSUR,  ",",0,",",discretization, "\n")

                    if args["poc_sto"]
                        # Initialize STO problem with order of modes by SUR
                        readd_callback!(solver)
                        f_, df_, uvec_, = SecondOrderPOC.prepare_warmstart(m)
                        m_STO = SwitchTimeOpt.stoproblem(x0, f_, df_, uvec_, ngrid=discretization, Q=Q, t0=t0, tf=tf, solver=solver)
                        SwitchTimeOpt.solve!(m_STO)
                        status = SwitchTimeOpt.getstat(m_STO)
                        _objective = SwitchTimeOpt.getobjval(m_STO)
                        time = SwitchTimeOpt.getsoltime(m_STO)
                        objective = @sprintf("%0.5e", _objective)
                        N_stoWS = size(uvec_, 2)
                        result_row_csv = string(result_row_csv, uniqueID, ",", name, ",", "poc_sto", ",$(args["bfgs"]),", N_stoWS, ",",discretization, ",", string(status), ",", objective, ",", time,  ",",iteration, ",",discretization, "\n")

                    end

                    if args["poc_stoWS"]
                        # Initialize STO problem with order of modes and switching times by SUR
                        readd_callback!(solver)
                        f_, df_, uvec_, = SecondOrderPOC.prepare_warmstart(m)
                        m_STO = SwitchTimeOpt.stoproblem(x0, f_, df_, uvec_, ngrid=discretization, Q=Q, t0=t0, tf=tf, solver=solver, tau0ws=tauSUR)
                        SwitchTimeOpt.solve!(m_STO)
                        status = SwitchTimeOpt.getstat(m_STO)
                        _objective = SwitchTimeOpt.getobjval(m_STO)
                        time = SwitchTimeOpt.getsoltime(m_STO)
                        objective = @sprintf("%0.5e", _objective)
                        N_stoWS = size(uvec_, 2)
                        result_row_csv = string(result_row_csv, uniqueID, ",", name, ",", "poc_stoWS", ",$(args["bfgs"]),", N_stoWS, ",",discretization, ",", string(status), ",", objective, ",", time,  ",",iteration,",",discretization, "\n")

                    end

                    if args["poc_stoFair"]
                        N_SUR_FAIR = SecondOrderPOC.count_necessary_switches(uvec)
                        f_, df_, uvec_, = SecondOrderPOC.prepare_warmstart(m, N=N_SUR_FAIR)
                        readd_callback!(solver)

                        m_STO = SwitchTimeOpt.stoproblem(x0, f_, df_, uvec_, ngrid=discretization, Q=Q, t0=t0, tf=tf, solver=solver)
                        SwitchTimeOpt.solve!(m_STO)
                        status = SwitchTimeOpt.getstat(m_STO)
                        _objective = SwitchTimeOpt.getobjval(m_STO)
                        time = SwitchTimeOpt.getsoltime(m_STO)
                        objective = @sprintf("%0.5e", _objective)
                        result_row_csv = string(result_row_csv, uniqueID, ",", name, ",", "poc_stoFair", ",$(args["bfgs"]),", N_SUR_FAIR, ",",discretization, ",", string(status), ",", objective, ",", time,  ",",iteration,",",discretization, "\n")
                    end
                end

                # Solve problem in STO formulation
                if args["sto"]
                    readd_callback!(solver)
                    f_, df_, uvec_, = SecondOrderPOC.prepare_warmstart(_m, N=N)
                    m_STO = SwitchTimeOpt.stoproblem(x0, f_, df_, uvec_, ngrid=num_lin, Q=Q, t0=t0, tf=tf, solver=solver)
                    SwitchTimeOpt.solve!(m_STO)
                    status = SwitchTimeOpt.getstat(m_STO)
                    _objective = SwitchTimeOpt.getobjval(m_STO)
                    time = SwitchTimeOpt.getsoltime(m_STO)
                    objective = @sprintf("%0.5e", _objective)
                    result_row_csv = string(result_row_csv, uniqueID, ",", name, ",", "sto", ",$(args["bfgs"]),", N, ",",num_lin, ",", string(status), ",", objective, ",", time,  ",",iteration, ",", num_lin, "\n")
                end


            catch e
                @warn e
                status = "EXCEPTION:"*string(typeof(e))
                result_row_csv = string(result_row_csv, status)
            finally
                open(results_file,"a") do fi
                    write(fi, result_row_csv)
                    write(fi, "\n")
                end
            end
        end
    end
end
