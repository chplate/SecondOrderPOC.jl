"Simulate system for obtained control."
function simulate(m::POC)
    omega = getomega(m)
    return simulate(m, omega)
end

"Simulate nonlinear system by linearizing and integrating via matrix exponentials for
 obtained control."
function simulatelinearized(m::nlinPOC)
    omega = getomega(m)
    return simulatelinearized(m, omega)
end

"Simulate nonlinear system for given controls omega."
function simulate(m::POC, omega)
    # Define Time
    t = collect(range(m.POCev.t0, stop=m.POCev.tf, length=10000))

    linearSystem = typeof(m.POCev) <: linPOCev
    nx = linearSystem ? m.POCev.nx : m.POCev.nx-1

    # Get original Q, x0
    Q = m.POCev.Q[1:nx, 1:nx]
    E = m.POCev.E[1:nx, 1:nx]
    x0 = m.POCev.x0[1:nx]

    # Perform Simulation
    x, xpts, J = simulatepoc(m, omega, m.POCev.n_omega, x0, Q, E, t, linearSystem)

    return x, xpts, J, t
end

"Simulate system for switching times tau and integer inputs uvec (possibly
 obtained via sum-up-rounding)."
function simulate(m::POC, tau::Array{<:Real, 1}, uvec::Array{<:Real,2})

    # Define Time
    t = collect(range(m.POCev.t0, stop=m.POCev.tf, length=10000))
    linearSystem = typeof(m.POCev) <: linPOCev
    nx = linearSystem ? m.POCev.nx : m.POCev.nx-1

    # Get original, Q, x0
    Q = m.POCev.Q[1:nx, 1:nx]
    E = m.POCev.E[1:nx, 1:nx]
    x0 = m.POCev.x0[1:nx]

    # Perform Simulation
    x, xpts, J = simulateintegerinput(m, tau, x0, linearSystem, Q, E,  uvec, t)

    return x, xpts, J, t
end

"Simulate nonlinear system by linearizing and integrating via matrix exponentials
 for given controls omega."
function simulatelinearized(m::nlinPOC, omega)
    # Define Time
    t = collect(range(m.POCev.t0, stop=m.POCev.tf, length=m.POCev.N*m.POCev.nlin+1))

    # Get original E, x0
    x0 = m.POCev.x0[1:end-1]

    # Perform Simulation
    x, xpts, J = simulatelinearizedpoc(m.POCev.nonlin_dyn, m.POCev.nonlin_dyn_deriv, omega,
                t, m.POCev.xi, x0, m.POCev.N, m.POCev.Q, m.POCev.E, m.POCev.nlin)

    return  x[1:end-1,:], xpts, J, t
end

"Helper function to define nonlinear dynamics for integration.
 Computes weighted sum of each mode for given control omega."
function eval_f(nldyn::Function, x, omega)
    f = nldyn(x)
    f[:,1] + sum([f[:,i+1]*omega[i] for i=1:length(omega)])
end

"Lower-level function to simulate nonlinear system by calling ODE integrator
 on each control interval."
function simulatepoc(m::POC, omega::Array{T,1}, nomega::Int, x0::Array{T, 1},
      Q::Array{T,2}, E::Array{T,2}, t::Array{T,1}, linear::Bool) where T
    # Get dimensions
    nx = length(x0)  # Number of States
    N = Int64(length(omega)/nomega)  # Number of switches

    # Define empty state trajectory
    x = zeros(nx, length(t))

    # Define indeces to determine current control interval
    tempInd1 = 1
    tempInd2 = 1
    xprevSwitch = x0
    xpts = x0

    for i = 1:N # Integrate over all the intervals
        # redefine Dynamic function
        currentControl = omega[(i-1)*nomega+1:i*nomega]

        nldyn(x, p, t) = begin
            if !linear
                return eval_f(m.POCev.nonlin_dyn, x, currentControl)
            else
                systemMatrix =  m.POCev.A[:,:,1] + sum([currentControl[i]*m.POCev.A[:,:,i+1] for i in eachindex(currentControl)])
                return  systemMatrix*x
            end
        end

        while t[tempInd2] < i*(t[end]-t[1])/N
            tempInd2 = tempInd2 + 1  # Increase time index
        end

        if tempInd2>tempInd1
        prob = DiffEqBase.ODEProblem(nldyn,xprevSwitch,(t[tempInd1],t[tempInd2]))
        sol = DiffEqBase.solve(prob,OrdinaryDiffEq.Tsit5();saveat=t[tempInd1:tempInd2])
        xtemp = hcat((sol[i] for i in eachindex(sol))...)

        x[:, tempInd1:tempInd2] = xtemp

        # Update indeces for next iteration
        xprevSwitch = x[:, tempInd2]
        xpts = hcat(xpts, xprevSwitch)
        tempInd1 = tempInd2
        end
    end
    # Numerically Integrate Cost Function
    Jtoint = diag(x'*Q*x)
    J = trapz(t, Jtoint) + (x[:, end]'*E*x[:, end])[1]
    return x, xpts, J
end

function simulateintegerinput(m::POC, tau::Array{T,1}, x0::Array{T,1}, linear::Bool,
        Q::Array{T,2}, E::Array{T,2}, uvec::Array{T,2}, t::Array{T,1}) where T
    # Get dimensions
    nx = length(x0)  # Number of States
    N = length(tau)  # Number of switches

    tau = [t[1]; tau; t[end]]  # Extend tau vector to simplify numbering

    # Define empty state trajectory
    x = zeros(nx, length(t))
    x[:,1] = x0

    # Define indeces to determine current switching mode
    tempInd1 = 1
    tempInd2 = 1
    xprevSwitch = x0

    # Create Vector of states at grid points on coarse grid
    xpts = zeros(nx, N+1)
    xpts[:, 1] = x0

    for i = 1:N+1 # Integrate over all the intervals
        # redefine Dynamic function

        nldyn(x, p, t) = begin
            if !linear
                return eval_f(m.POCev.nonlin_dyn, x, uvec[:,i])
            else
                currentControl = uvec[:,i]
                systemMatrix =  m.POCev.A[:,:,1] + sum([currentControl[i]*m.POCev.A[:,:,i+1] for i in eachindex(currentControl)])
                return  systemMatrix*x
            end
        end

        while t[tempInd2] < tau[i+1]
            tempInd2 = tempInd2 + 1  # Increase time index
        end

        if tempInd2>tempInd1  # There has been a progress and the switching times are not collapsed. So we integrate.
            prob = DiffEqBase.ODEProblem(nldyn,xprevSwitch,(t[tempInd1],t[tempInd2]))
            if tempInd2 == tempInd1 + 1  # Only one step progress. Handle as special case for the integrator
                sol = DiffEqBase.solve(prob,OrdinaryDiffEq.Tsit5();save_everystep=false)
                xtemp = [sol[1] sol[end]]
                #_, xmap = ode45(nldyn, xprevSwitch, [t[tempInd1]; (t[tempInd1]+t[tempInd2])/2; t[tempInd2]], points=:specified)  # Integrate
                #xmap = hcat(xmap...)
                #xtemp = [xmap[:,1] xmap[:,3]]          # Take only two points
            else
                sol = DiffEqBase.solve(prob,OrdinaryDiffEq.Tsit5();saveat=t[tempInd1:tempInd2])
                xtemp = hcat((sol[i] for i in eachindex(sol))...)
            end

            x[:, tempInd1:tempInd2] = xtemp

            # Update indeces for next iteration
            xprevSwitch = x[:, tempInd2]
            tempInd1 = tempInd2

            # Update vector of switching states
            if i < N+1
                xpts[:, i+1] = x[:, tempInd2]
            end
        end
    end

    # Numerically Integrate Cost Function
    Jtoint = diag(x'*Q*x)
    J = trapz(t, Jtoint) + (x[:,end]'*E*x[:,end])[1]
    return x, xpts, J
end

"Lower-level function to numerically integrate system by linearizing and computing
 matrix exponentials for given control omega."
function simulatelinearizedpoc(nonlin_dyn::Function, nonlin_dyn_deriv::Function,
                                omega::Array{T,1}, t::Array{T,1}, xi::T, x0::Array{T, 1},
                                N::Int, Q::Array{T,2}, E::Array{T,2}, nlin::Int) where T

    # Get dimensions
    nx = length(x0)
    nomega = convert(Integer, length(omega)/N)
    A = zeros(nx+1, nx+1)

    x = zeros(nx+1, N*nlin+1)  # Augmented State for Linearization
    x[:,1] = [x0; 1]
    xpts = x0                   # xpts will include only states on the N outer grid points

    for i=1:N
        currentControl = omega[(i-1)*nomega+1:i*nomega]
        for j=1:nlin
            linearizeDyn!(A, nonlin_dyn, nonlin_dyn_deriv, x[1:end-1,(i-1)*nlin+j], currentControl)
            x[:,(i-1)*nlin+j+1] = exp(A*xi) * x[:,(i-1)*nlin+j]
        end
        xpts = hcat(xpts, x[1:end-1,i*nlin+1])
    end

    # Numerically Integrate Cost Function
    Jtoint = diag(x'*Q*x)
    J = trapz(t, Jtoint) + (x[:, end]'*E*x[:, end])[1]
    return x, xpts, J
end

"Trapezoidal integration rule"
function trapz(x::Vector{Tx}, y::Vector{Ty}) where {Tx, Ty}
    n = length(x)
    @assert n == length(y) "Vectors must be of the same length."
    @assert n > 1 "Vectors must be of length > 1."
    r = sum([(x[i] - x[i-1]) * (y[i] + y[i-1]) for i=2:n])
    return r/2
end