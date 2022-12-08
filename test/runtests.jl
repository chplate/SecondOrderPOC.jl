using SecondOrderPOC
using Test
using SafeTestsets

@time @safetestset "Lotka" begin include("./lotka.jl") end
@time @safetestset "Fuller (terminal constraint)" begin include("./fuller.jl") end
@time @safetestset "Egerstedt (box constraint)" begin include("./egerstedt.jl") end
