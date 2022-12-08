# Sum up rounding
function sum_up_rounding(m::POC)
    SUR = sur(m)
    tau, uvec = compute_sto_controls(m,SUR)

    # returns switching times tau and integer controls uvec
    return tau, uvec
end

# Compute switching times and integer controls from SUR
function compute_sto_controls(m::POC, SUR::Array{Float64,2})
    times = []
    columns = []
    lengthOfControlInterval = (m.POCev.tf - m.POCev.t0)/m.POCev.N

    currentColumn = SUR[:,1]
    currentLength = 0.0
    for interval = 1:m.POCev.N
        if SUR[:,interval] == currentColumn
            currentLength += lengthOfControlInterval
        else
            append!(times,currentLength)
            currentLength += lengthOfControlInterval
            append!(columns,currentColumn)
            currentColumn = SUR[:,interval]
        end
    end
    #add last configuration
    append!(columns,currentColumn)

    columns = Array{Float64,1}(columns)
    columns = reshape(columns,m.POCev.n_omega,:)
    times = Array{Float64,1}(times)
    return times, columns
end

# Sum up rounding with maximum choice for control for each interval
function sur(m::POC)
    controls = reshape(m.omega, (m.POCev.n_omega,:))
    return sur(controls)
end


function sur(controls::AbstractArray)
    if length(size(controls)) == 1
        controls = reshape(controls, (1,len(controls)))
    end
    @assert length(size(controls)) == 2 "Controls must be 2D array with dimensions (nw, N)."
    n_omega, N = size(controls)

    binarycontrols          = zeros(size(controls))
    summedbinarycontrols    = zeros(size(controls[:,1]))
    for i=1:N
        summedbinarycontrols  = i > 1 ? sum([binarycontrols[:,j] for j=1:i-1]) : summedbinarycontrols
        summedrelaxedcontrols = i > 1 ? sum([controls[:,j] for j=1:i]) : controls[:,i]
        diff_controls = summedrelaxedcontrols-summedbinarycontrols
        if n_omega > 1
            controlidx = argmax(diff_controls)
            binarycontrols[controlidx,i] = 1.0
        else
            binarycontrols[1,i] = diff_controls[1] > 1.0 ? 1.0 : 0.0
        end
    end
    return binarycontrols
end


# From SUR solution: how many switches do you need for a fair comparison with STO when modes
# are traversed in their order, e.g., 1-2-3-1-2-3-1-2-3...
# but you want the chance to replicate the order from SUR, e.g., 1,3,2,3,2,1..
function count_necessary_switches(columns::Array{Float64,2})
    N = 0
    n_omega = length(columns[:,1])
    numIntervals= length(columns[1,:])
    currentInterval = 1
    reachedEnd = false
    if n_omega == 1
        if columns[1,1] == 1.0
            return numIntervals
        else
            return numIntervals + 1
        end
    end
    while !reachedEnd
        for control = 1:n_omega
            N += 1
            if argmax(columns[:, currentInterval]) == control
                currentInterval += 1
                if currentInterval > numIntervals
                    reachedEnd = true
                    break
                end
            end
        end
    end
    return N
end

function prepare_warmstart(m::nlinPOC; N=0, firstMode=1)
    n_omega = m.POCev.n_omega
    if N == 0
        ~, uvec = sum_up_rounding(m)
    else
        uvec = zeros(n_omega, N)
        if n_omega > 1
            for i=1:N
                idx = (i-2+firstMode)%n_omega+1
                uvec[idx,i] = 1.0
            end
        else
            for i=firstMode:2:N
                uvec[1,i] = 1.0
            end
        end
    end

    f, df = m.POCev.nonlin_dyn, m.POCev.nonlin_dyn_deriv
    f_STO = (x, u) -> begin
        _f = f(x)
        _f[:,1] + sum([_f[:,i+1]*u[i] for i in eachindex(u)])
    end
    df_STO = (x,u) -> begin
        _df = df(x)
        _df[:,:,1] + sum([_df[:,:,i+1]*u[i] for i in eachindex(u)])
    end
    return f_STO, df_STO, uvec
end

function prepare_warmstart(m::linPOC; N=0, firstMode=1)
    _A = m.POCev.A
    ~, uvec = sum_up_rounding(m)
    n_omega = size(_A, 3)
    if N == 0
        _N = size(uvec)[2]
        A = zeros(m.POCev.nx, m.POCev.nx, _N)
        for i=1:_N
            A[:,:,i] = _A[:,:,1] + _A[:,:,1+argmax(uvec[:,i])]
        end
    elseif N > 0
        A = zeros(m.POCev.nx, m.POCev.nx, N)
        for i=1:N
            mode = mod(i-2+firstMode, n_omega) + 1
            A[:,:,i] = _A[:,:,1] + _A[:,:,1+mode]
        end
    end
    return A, uvec
end
