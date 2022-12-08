module SecondOrderPOC

# Import Necessary Modules
using MathOptInterface
using DiffEqBase
using OrdinaryDiffEq
using Ipopt
using LinearAlgebra
using ForwardDiff

export pocproblem, setwarmstart!, setx0!, solve!
export getomega, getobjval, getstat, getsoltime, getnobjeval, getngradeval, getnhesseval
export simulatelinearized, simulate, sum_up_rounding, sur

# Include Source Files
include("types.jl")
include("interface.jl")
include("nlpPOC.jl")
include("rounding.jl")
include("simulation.jl")

end
