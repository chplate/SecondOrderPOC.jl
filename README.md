# SecondOrderPOC.jl
SecondOrderPOC.jl - Formulating and solving optimal control problems in Julia.

`SecondOrderPOC.jl` is a [Julia](https://julialang.org) package that allows users to easily define and efficiently solve the **partial outer convexification (POC)** reformulation of mixed-integer optimal control problems for switched systems governed by ordinary differential equations. 
****

The problem addressed in this package is of the following form:

```math
\begin{aligned}
    &\underset{w,x}{\text{min}} && \int_{t_0}^{t_f} x(t)^\top Q x(t) \mathrm{d}t + x(t_f)^\top E x(t_f) \\
    &\text{subject to}
    &&\dot{x}(t) = f_0(x(t)) + \sum_{i=1}^{n_{\omega}} w_i(t) f_i (x(t)),\\
    &&& c(x(t),t) \leq 0,\\
    &&& w_i(t) \in [0,1], \ i \in [n_{\omega}],\\
    &&& \sum_{i=1}^{n_{\omega}} w_i(t) = 1,\\
    &&& x(t_0) = x_0.\\
\end{aligned}
```

The implementation uses a direct, first-discretize-then-optimize approach. This involves a discretization of the control functions $w_i(t)$ as piecewise constant functions on $N$ control intervals. Moreover, the dynamics of the ODE are linearized and integrated via matrix exponentials. In particular, this implementation is based on the approach used in the package [SwitchTimeOpt.jl](https://github.com/oxfordcontrol/SwitchTimeOpt.jl), a package for solving switching time optimization (STO) problems for linear and nonlinear systems. 

Through [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl), several NLP solvers can be interfaced to our code, for example [Ipopt](https://github.com/jump-dev/Ipopt.jl), [KNITRO](https://github.com/jump-dev/KNITRO.jl) and [NLopt](https://github.com/JuliaOpt/NLopt.jl). 

## Installation

To use `SecondOrderPOC.jl`, install it via:

```julia
]add https://github.com/chplate/SwitchTimeOpt.jl
]add https://github.com/chplate/SecondOrderPOC.jl
using SecondOrderPOC
```
