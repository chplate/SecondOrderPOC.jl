# Define initialization Function
function MathOptInterface.initialize(d::POCev, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
  end

# List Available Features
MathOptInterface.features_available(d::POCev) = [:Grad, :Jac , :Hess]


" Linearize nonlinear dynamics inplace for oop functions, eg result is saved in A but fun and dfun are oop."
function linearizeDyn!(A::Array{T,2}, fun::Function, dfun::Function, x::Array{T,1}, omega::Array{T,1}) where T

    f  = fun(x)
    df = dfun(x)

    f0 = f[:,1] + sum([f[:,i+1] * wi for (i,wi) in enumerate(omega)])
    Jf0 = df[:,:,1] + sum([df[:,:,i+1] * wi for (i,wi) in enumerate(omega)])

    A .= [Jf0 f0-Jf0*x;
        zeros(1, length(x)+1)]
    return nothing
end

# Functions to precompute matrices for POC
"Precompute matrices for cost function"
function precomputations_evaluation!(d::POCev, x)
    propagate_dynamics!(d, x)   # Propagate dynamics
    computeS!(d)                # Compute S matrices
end

"Precompute matrices for gradient"
function precomputations_gradient!(d::POCev, x)
    # Check if control has changed since last precomputation
    if d.omega_fun_prev != x
        precomputations_evaluation!(d, x)      # precompute matrices for cost function
        d.omega_fun_prev[:] = x
    end
    computeZ!(d, x)           # Compute Matrices Z

    if norm(d.Q) != 0.0       # when no Lagrange term present -> computation of L not needed
        computeLAndC!(d,x)
    else
        computeC!(d,x)
        fill!(d.L, zero(Float64))
    end
end


"Precompute matrices for hessian"
function precomputations_hessian!(d::POCev, x)
    if d.omega_grad_prev != x     # Different point than the previous gradient evaluation
        precomputations_gradient!(d, x)      # precompute matrices for gradient function
        d.omega_grad_prev[:] = x
    end

    computePhi!(d)            # Compute Matrices Phi
    if norm(d.Q) != 0.0       # when no Lagrange term present -> computation of G not needed
        computeDAndG!(d,x)
    else
        computeD!(d,x)
        fill!(d.G, zero(Float64))
    end
end

function eval_objective(d::POCev, x)
    d.nobjeval += 1           # Increase counter of objective function evaluations
    if d.omega_fun_prev != x  # Different point than in previous function evaluation
      precomputations_evaluation!(d, x)
      d.omega_fun_prev[:] = x
    end

    J = d.x0' * d.S[:,:,1] * d.x0   # Evaluate cost function
    return J
end

# Evaluate Cost Function
function MathOptInterface.eval_objective(d::POCev, x)
    eval_objective(d,x)
end

# Evaluate Gradient
function MathOptInterface.eval_objective_gradient(d::POCev, grad_f, x)
    d.ngradeval += 1            # Increase counter of gradient evaluations
    linearSystem = typeof(d) <: linPOCev
    nlin = linearSystem ? 1 : d.nlin
    if d.omega_grad_prev != x   # Different point than in previous gradient evaluation
        precomputations_gradient!(d, x)
        d.omega_grad_prev[:] = x
    end

    for i = 1:d.N
        x_i_start = @view d.xpts[:,(i-1)*nlin+1]
        x_i_end   = @view d.xpts[:,i*nlin+1]
        S_i       = @view d.S[:,:,i+1]

        for j = 1:d.n_omega
            C_ij = @view d.C[:,:,i,j]
            grad_f[(i-1)*d.n_omega+j] = x_i_start' * d.L[:,:,i,j] * x_i_start +
                                        2 * x_i_end' * S_i * C_ij * x_i_start
        end
    end
end


function MathOptInterface.eval_hessian_lagrangian(d::POCev, H, x, sigma, mu)
    d.nhesseval += 1   # Increase counter of Hessian evaluations
    linearSystem = typeof(d) <: linPOCev
    nlin = linearSystem ? 1 : d.nlin
    if d.omega_hess_prev != x   # Different point than in previous Hessian evaluation
        precomputations_hessian!(d, x)
        d.omega_hess_prev[:] = x
    end

    H_temp = zeros(d.N*d.n_omega,d.N*d.n_omega)

    for i=1:d.N
        x_i_start = @view d.xpts[:,(i-1)*nlin+1]
        S_i       = @view d.S[:,:,i+1]
        Phi_i_begin_to_i_end = @view d.Phi[:,:,i,i+1]

        for k=1:d.n_omega
            C_ik = @view d.C[:,:,i,k]

            for p=k:d.n_omega
                C_ip = @view d.C[:,:,i,p]
                D_ikp = @view d.D[:,:,i,k,p]

                H_temp[(i-1)*d.n_omega+k,(i-1)*d.n_omega+p] = x_i_start' * (d.G[:,:,i,k,p] +
                    2 * C_ik' * S_i * C_ip +
                    2 * Phi_i_begin_to_i_end' * S_i * D_ikp ) * x_i_start
            end
        end

        for j=i+1:d.N
            S_j       = @view d.S[:,:,j+1]
            Phi_i_end_to_j_start = @view d.Phi[:,:,i+1,j]
            Phi_j_start_to_j_end = @view d.Phi[:,:,j,j+1]
            x_j_start   = @view d.xpts[:,(j-1)*nlin+1]

            for k=1:d.n_omega
                C_ik = @view d.C[:,:,i,k]

                for p=1:d.n_omega
                    C_jp = @view d.C[:,:,j,p]
                    L_jp =  @view d.L[:,:,j,p]

                    H_temp[(i-1)*d.n_omega+k,(j-1)*d.n_omega+p] = 2*x_j_start' * (L_jp +
                        C_jp' * S_j * Phi_j_start_to_j_end +
                        Phi_j_start_to_j_end'* S_j * C_jp) * Phi_i_end_to_j_start * C_ik * x_i_start
                end
            end
        end
    end

    H_temp *= sigma
    H[:] = H_temp[d.IndTril]
end

"Propagate dynamics for nonlinear POC"
function propagate_dynamics!(d::nlinPOCev, x)
    A = similar(d.E)
    tmp = zeros(2 .* size(A))
    tmp[1:d.nx, d.nx+1:end] = d.Q

    current_index = 0
    for i = 1:d.N
        currentControl = x[(i-1)*d.n_omega+1:i*d.n_omega]
        for j = 1:d.nlin
            current_index += 1
            # Linearize Dynamics
            linearizeDyn!(A, d.nonlin_dyn, d.nonlin_dyn_deriv, d.xpts[1:d.nx-1,current_index],
                            currentControl)

            # Compute matrix exponential
            d.A[:,:,i,j] = A
            tmp[1:d.nx, 1:d.nx] = -A'
            tmp[d.nx+1:end,d.nx+1:end] = A
            Z = exp(tmp*d.xi)
            d.expMat[:,:,i,j] = Z[d.nx+1:end, d.nx+1:end]
            d.M[:,:,i,j] = Z[d.nx+1:end, d.nx+1:end]' * Z[1:d.nx, d.nx+1:end]

            # Compute state at the next grid point - Last state doesn't need to be computed
            d.xpts[1:end-1,current_index+1] = d.expMat[1:d.nx-1,:,i,j]*d.xpts[:,current_index]
        end
    end
end

"Propagate dynamics for linear POC"
function propagate_dynamics!(d::linPOCev, x)
    tmp = zeros(2*d.nx, 2*d.nx)
    tmp[1:d.nx, d.nx+1:end] = d.Q
    for i = 1:d.N
        currentControl = x[(i-1)*d.n_omega+1:i*d.n_omega]
        A = d.A[:,:,1] + sum([w_i * d.A[:,:,j+1] for (j, w_i) in enumerate(currentControl)])
        tmp[1:d.nx, 1:d.nx] = -A'
        tmp[d.nx+1:end,d.nx+1:end] = A
        Z = exp(tmp*d.xi)
        d.M[:,:,i] = Z[d.nx+1:end, d.nx+1:end]' * Z[1:d.nx, d.nx+1:end]
        d.expMat[:,:,i] = Z[d.nx+1:end, d.nx+1:end]
        d.xpts[:,i+1] = d.expMat[:,:,i]*d.xpts[:,i]
    end
end

"Low level function for computing S matrices"
function computeS!(d::nlinPOCev)
    d.S[:,:,d.N+1] = d.E
    @views for i = d.N:-1:1
        d.S[:,:,i] = d.S[:,:,i+1]
        for j = d.nlin:-1:1
            d.S[:,:,i] = d.expMat[:,:,i,j]' * d.S[:,:,i] * d.expMat[:,:,i,j]
            d.S[:,:,i] += d.M[:,:,i,j]
        end
    end
end

"Low level function for computing S matrices"
function computeS!(d::linPOCev)
    d.S[:,:,d.N+1] = d.E
    @views for i = d.N:-1:1
        d.S[:,:,i] = d.S[:,:,i+1]
        d.S[:,:,i] = d.expMat[:,:,i]' * d.S[:,:,i] * d.expMat[:,:,i]
        d.S[:,:,i] += d.M[:,:,i]
    end
end

"Compute second derivative of matrix expoential exp(A*dt) with directions B and C.
 Uses complex step method ('i-trick') in combination with Mathias' block-triangular method."
function complex_step_second_derivative(A, B, C, dt; full=false, h=1e-12)
    dim = size(A,1)
    D = zeros(eltype(A), dim, dim)

    d2AdBdC = 1/h * imag(exp(dt*[A+h*im*C B;
                          D A+h*im*C]))
    full && return d2AdBdC
    return d2AdBdC[1:dim, dim+1:2dim]
end


"Low level function for computing Z matrices (as derivatives of exp(A_ij*xi))"
function computeZ!(d::nlinPOCev, control; h::Float64=1e-12)
    fill!(d.Z, zero(Float64))
    fill!(d.ZZ, zero(Float64))

    for i = 1:d.N
        for j = 1:d.nlin
            current_index = (i-1)*d.nlin+j
            x = d.xpts[1:d.nx-1,current_index]

            f = d.nonlin_dyn(x)
            df = d.nonlin_dyn_deriv(x)
            for k = 1:d.n_omega

                Ak = [df[:,:,k+1] f[:,k+1]-df[:,:,k+1]*x;
                    zeros(1,length(x)+1)]

                dEdAkdAk = complex_step_second_derivative(d.A[:,:,i,j], Ak, Ak, d.xi, full=true, h=h)
                d.Z[:,:,i,j,k] = dEdAkdAk[1:d.nx,1:d.nx]
                d.ZZ[:,:,i,j,k,k] = dEdAkdAk[1:d.nx,d.nx+1:2*d.nx]

                for l = k+1:d.n_omega
                    Al = [df[:,:,l+1] f[:,l+1]-df[:,:,l+1]*x;
                        zeros(1,length(x)+1)]
                    d.ZZ[:,:,i,j,k,l] = complex_step_second_derivative(d.A[:,:,i,j], Ak, Al, d.xi, h=h)
                end
            end
        end
    end
end

"Low level function for computing Z matrices (as derivatives of exp(A_ij*xi))"
function computeZ!(d::linPOCev, control; h::Float64=1e-12)
    fill!(d.Z, zero(Float64))
    fill!(d.ZZ, zero(Float64))
    for i = 1:d.N
        currentControl = control[(i-1)*d.n_omega+1:i*d.n_omega]
        A = d.A[:,:,1] + sum([w_i * d.A[:,:,j+1] for (j, w_i) in enumerate(currentControl)])
        for k = 1:d.n_omega

            Ak = d.A[:,:,k+1]

            dEdAkdAk = complex_step_second_derivative(A, Ak, Ak, d.xi, full=true, h=h)
            d.Z[:,:,i,j,k] = dEdAkdAk[1:d.nx,1:d.nx]
            d.ZZ[:,:,i,j,k,k] = dEdAkdAk[1:d.nx,d.nx+1:2*d.nx]

            for l = k+1:d.n_omega
                Al = [df[:,:,l+1] f[:,l+1]-df[:,:,l+1]*x;
                    zeros(1,length(x)+1)]
                d.ZZ[:,:,i,j,k,l] = complex_step_second_derivative(A, Ak, Al, d.xi, h=h)
            end
        end
    end
end


"Low level function for computing C matrices for linear dynamics.
 Computes matrix C_i^j in notation."
function computeC!(d::linPOCev, x)
    # Fill all C matrices with zeros
    fill!(d.C, zero(Float64))

    # Loop over all control intervals and controls
    @views for i = 1:d.N
        for j = 1:d.n_omega
            d.C[:,:,i,j] = d.Z[:,:,i,j]
        end
    end
end

"Low level function for computing L and C matrices combined using a Horner scheme."
function computeLAndC!(d::nlinPOCev, x)
    fill!(d.L, zero(Float64))
    fill!(d.C, zero(Float64))
    # Loop over all control intervals and controls
    @views for i = 1:d.N
        for j = 1:d.n_omega
            for k=1:d.nlin
                TempProd = d.expMat[:,:,i,1]
                TempSum = d.Z[:,:,i,1,j]
                for l = 2:k
                    TempSum = d.expMat[:,:,i,l] * TempSum  + d.Z[:,:,i,l,j] * TempProd
                    TempProd = d.expMat[:,:,i,l] * TempProd
                end
                d.L[:,:,i,j] += d.xi * (TempSum' * d.Q * TempProd + TempProd' * d.Q * TempSum)
                if k==d.nlin
                    d.C[:,:,i,j] = TempSum
                end
            end
        end
    end
end

"Low level function for computing L and C matrices combined using a Horner scheme."
function computeC!(d::nlinPOCev, x)
    fill!(d.C, zero(Float64))
    # Loop over all control intervals and controls
    @views for i = 1:d.N
        for j = 1:d.n_omega
            TempProd = d.expMat[:,:,i,1]
            TempSum = d.Z[:,:,i,1,j]
            for l = 2:d.nlin
              TempSum = d.expMat[:,:,i,l] * TempSum  + d.Z[:,:,i,l,j] * TempProd
              TempProd = d.expMat[:,:,i,l] * TempProd
         	end
            d.C[:,:,i,j] = TempSum
        end
    end
end

"Low level function for computing L and C matrices combined using a Horner scheme."
function computeLAndC!(d::linPOCev, x)
    fill!(d.L, zero(Float64))
    fill!(d.C, zero(Float64))
    # Loop over all control intervals and controls
    xi = d.xi / d.nlin
    @views for i = 1:d.N
        currentControl = x[(i-1)*d.n_omega+1:i*d.n_omega]
        A = d.A[:,:,1] + sum([currentControl[j]*d.A[:,:,j+1] for j in eachindex(currentControl)])
        expA = exp(A*xi)
        for j = 1:d.n_omega
            d.C[:,:,i,j] = d.Z[:,:,i,j]
            for k=1:d.nlin
                TempProd = expA
                TempSum = d.Z[:,:,i,j]
                for l = 2:k
                    TempSum = expA * TempSum  + d.Z[:,:,i,j] * TempProd
                    TempProd = expA * TempProd
                end
                d.L[:,:,i,j] += xi * (TempSum' * d.Q * TempProd + TempProd' * d.Q * TempSum)
            end
        end
    end
end


"Low level function for computing Phi matrices.
 Reminder: Phi matrix offer transition of states from first index
 to the second index (where the indices can be in range [1,N+1])."
function computePhi!(d::nlinPOCev)
    tmp = zeros(d.nx-1, d.nx)
    Identity = 1.0 * Matrix(I, d.nx, d.nx)
    @views for i = 1:d.N+1
        d.Phi[:,:,i,i] = Identity   # Identity Matrix to Start
        for l = i:d.N   # Iterate over all successive Phi matrices
            d.Phi[:,:,i,l+1] = d.Phi[:,:,i,l] # Initialize with previous matrix
            for j = 1:d.nlin # Iterate over all points on control interval
                mul!(tmp, d.expMat[1:end-1,:,l,j], d.Phi[:,:,i,l+1])
                d.Phi[1:d.nx-1,:,i,l+1] = tmp
            end
        end
    end
end

"Low level function for computing Phi matrices.
 Reminder: Phi matrix offer transition of states from first index
 to the second index (where the indices can be in range [1,N+1])."
function computePhi!(d::linPOCev)
    Identity = 1.0 * Matrix(I, d.nx, d.nx)
    @views for i = 1:d.N+1
        d.Phi[:,:,i,i] = Identity   # Identity Matrix to Start
        for l = i:d.N   # Iterate over all successive Phi matrices
            d.Phi[:,:,i,l+1] = d.Phi[:,:,i,l] # Initialize with previous matrix
            d.Phi[:,:,i,l+1] = d.expMat[:,:,l] * d.Phi[:,:,i,l+1]
        end
    end
end

"Low level function for computing D matrices for nonlinear dynamics.
 Computes matrix D_i^{jk} in paper notation using Horner scheme and product rule."
function computeD!(d::nlinPOCev, x)
    # Fill all D matrices with zeros
    fill!(d.D, zero(Float64))

    Identity = 1.0 * Matrix(I, d.nx, d.nx)
    # Loop over all control intervals
    for i = 1:d.N
        # Loop over all combinations of two controls on same control interval
        for p = 1:d.n_omega
            for k = p:d.n_omega
                # First the terms with ZZ matrices (identical to naive implementation)
                if norm(d.ZZ, Inf) > 1e-06
                    TempProd = Identity
                    TempSum = d.ZZ[:,:,i,d.nlin,p,k]
                    for l = d.nlin:-1:2
                        TempProd = TempProd * d.expMat[:,:,i,l]
                        TempSum = TempSum * d.expMat[:,:,i,l-1] + TempProd * d.ZZ[:,:,i,l-1,p,k]
                    end
                    d.D[:,:,i,p,k] += TempSum
                end

                # Now the terms with mixed derivatives in Horner scheme
                for idx=1:d.nlin-1
                    counter = 1
                    TempSumK = d.Z[:,:,i,d.nlin,k]
                    TempSumP = d.Z[:,:,i,d.nlin,p]
                    while counter < idx
                        TempSumK = TempSumK * d.expMat[:,:,i,d.nlin-counter]
                        TempSumP = TempSumP * d.expMat[:,:,i,d.nlin-counter]
                        counter += 1
                    end
                    TempSumK = TempSumK * d.Z[:,:,i,d.nlin-counter,p]
                    TempSumP = TempSumP * d.Z[:,:,i,d.nlin-counter,k]
                    TempSum = TempSumK + TempSumP
                    TempProd = I
                    for l = d.nlin-idx:-1:2
                        TempProd = TempProd * d.expMat[:,:,i,l+idx]
                        counter = 1
                        TempSumK = d.Z[:,:,i,l+idx-1,k]
                        TempSumP = d.Z[:,:,i,l+idx-1,p]
                        while counter < idx
                            TempSumK = TempSumK * d.expMat[:,:,i,l+idx-1-counter]
                            TempSumP = TempSumP * d.expMat[:,:,i,l+idx-1-counter]
                            counter += 1
                        end
                        TempSumK = TempSumK * d.Z[:,:,i,l+idx-1-counter,p]
                        TempSumP = TempSumP * d.Z[:,:,i,l+idx-1-counter,k]
                        TempSum = TempSum * d.expMat[:,:,i,l-1] + TempProd * (TempSumK + TempSumP)
                    end
                    d.D[:,:,i,p,k] += TempSum
                end
            end
        end
    end
end

function computeD!(d::linPOCev, x)
    # Fill all D matrices with zeros
    fill!(d.D, zero(Float64))
    for i=1:d.N
        for j=1:d.n_omega
            for k=j:d.n_omega
                d.D[:,:,i,j,k] = d.ZZ[:,:,i,j,k]
            end
        end
    end
end

"Low level function for computing D and G matrices combined for ninlinear dynamics using a Horner scheme.
 Computes matrix D_i^{pk} and G_i^{pk} in paper notation."
function computeDAndG!(d::nlinPOCev, x)

    # Fill all D matrices with zeros
    fill!(d.D, zero(Float64))
    fill!(d.G, zero(Float64))
    Identity = 1.0 * Matrix(I, d.nx, d.nx)

    # Loop over all control intervals
    for i = 1:d.N

        # Loop over all combinations of two controls on same control interval
        for p = 1:d.n_omega
            for k = p:d.n_omega
                Phi = Identity
                for index=1:d.nlin
                    Phi = d.expMat[:,:,i,index] * Phi

                    # First the terms with ZZ matrices
                    if norm(d.ZZ, Inf) > 1e-06
                        TempProd = d.expMat[:,:,i,1]
                        TempSum = d.ZZ[:,:,i,1,p,k]
                        for l = 1:index-1
                            TempSum =  d.expMat[:,:,i,l+1] * TempSum +  d.ZZ[:,:,i,l+1,p,k] * TempProd
                            TempProd = d.expMat[:,:,i,l+1] * TempProd
                        end
                        d.G[:,:,i,p,k] += d.xi * (TempSum' * d.Q * Phi + Phi'* d.Q * TempSum)
                        if index==d.nlin
                            d.D[:,:,i,p,k] += TempSum
                        end
                    end

                    # Now the terms with mixed derivatives
                    # idx is distance in index between to Z matrices:
                    # idx=1 -> Z_i^p and Z_i^k next to each other
                    for idx=1:index-1
                        counter = 1
                        TempSumK = d.Z[:,:,i,1,k]
                        TempSumP = d.Z[:,:,i,1,p]
                        while counter < idx
                            TempSumK = d.expMat[:,:,i,1+counter] * TempSumK
                            TempSumP = d.expMat[:,:,i,1+counter] * TempSumP
                        counter += 1
                        end
                        TempSumK = d.Z[:,:,i,1+counter,p] * TempSumK
                        TempSumP = d.Z[:,:,i,1+counter,k] * TempSumP
                        TempSum = TempSumK + TempSumP

                        TempProd = d.expMat[:,:,i,1]

                        for l = 2:index-idx #1:index-idx-1
                            counter = 1
                            TempSumK = d.Z[:,:,i,l,k]
                            TempSumP = d.Z[:,:,i,l,p]
                            while counter < idx
                                TempSumK =  d.expMat[:,:,i,l+counter] * TempSumK
                                TempSumP =  d.expMat[:,:,i,l+counter] * TempSumP
                                counter += 1
                            end
                            TempSumK = d.Z[:,:,i,l+counter,p] * TempSumK
                            TempSumP = d.Z[:,:,i,l+counter,k] * TempSumP
                            TempSum = d.expMat[:,:,i,l+idx] * TempSum + (TempSumK + TempSumP)*TempProd
                            TempProd =  d.expMat[:,:,i,l] * TempProd

                        end
                        d.G[:,:,i,p,k] += d.xi * (TempSum' * d.Q * Phi + Phi'* d.Q * TempSum)
                        if index==d.nlin
                            d.D[:,:,i,p,k] += TempSum
                        end
                    end
                end
            end
        end
    end
end

"Low level function for computing D and G matrices combined for linear dynamics using a Horner scheme.
 Computes matrix D_i^{pk} and G_i^{pk} in paper notation."
function computeDAndG!(d::linPOCev, x)
    # Fill all D matrices with zeros
    fill!(d.D, zero(Float64))
    fill!(d.G, zero(Float64))
    Identity = 1.0 * Matrix(I, d.nx, d.nx)

    xi = d.xi / d.nlin

    # Loop over all control intervals
    for i = 1:d.N
        currentControl = x[(i-1)*d.n_omega+1:i*d.n_omega]
        A = d.A[:,:,1] + sum([currentControl[j]*d.A[:,:,j+1] for j in eachindex(currentControl)])
        expA = exp(A*xi)
        # Loop over all combinations of two controls on same control interval
        for p = 1:d.n_omega
            for k = p:d.n_omega
                d.D[:,:,i,p,k] = d.ZZ[:,:,i,p,k]

                Phi = Identity
                for index=1:d.nlin
                    Phi = expA * Phi

                    # First the terms with ZZ matrices
                    if norm(d.ZZ, Inf) > 1e-06
                        TempProd = expA
                        TempSum = d.ZZ[:,:,i,p,k]
                        for l = 1:index-1
                            TempSum =  expA * TempSum +  d.ZZ[:,:,i,p,k] * TempProd
                            TempProd = expA * TempProd
                        end
                        d.G[:,:,i,p,k] += xi * (TempSum' * d.Q * Phi + Phi'* d.Q * TempSum)
                    end

                    # Now the terms with mixed derivatives

                    for idx=1:index-1
                        counter = 1
                        TempSumK = d.Z[:,:,i,k]
                        TempSumP = d.Z[:,:,i,p]
                        while counter < idx
                            TempSumK = expA * TempSumK
                            TempSumP = expA * TempSumP
                            counter += 1
                        end
                        TempSumK = d.Z[:,:,i,p] * TempSumK
                        TempSumP = d.Z[:,:,i,k] * TempSumP
                        TempSum = TempSumK + TempSumP

                        TempProd = expA
                        for l = 2:index-idx #1:index-idx-1
                            counter = 1
                            TempSumK = d.Z[:,:,i,k]
                            TempSumP = d.Z[:,:,i,p]
                            while counter < idx
                                TempSumK =  expA * TempSumK
                                TempSumP =  expA * TempSumP
                                counter += 1
                            end
                            TempSumK = d.Z[:,:,i,p] * TempSumK
                            TempSumP = d.Z[:,:,i,k] * TempSumP
                            TempSum = expA * TempSum + (TempSumK + TempSumP)*TempProd
                            TempProd =  expA * TempProd
                        end
                        d.G[:,:,i,p,k] += xi * (TempSum' * d.Q * Phi + Phi'* d.Q * TempSum)
                    end
                end
            end
        end
    end
end

# Constraints
function MathOptInterface.eval_constraint(d::POCev, g, x)
    if d.omega_fun_prev != x
        precomputations_evaluation!(d, x)
        d.omega_fun_prev[:] = x
    end
    linearSystem = typeof(d) <: linPOCev
    nlin = linearSystem ? 1 : d.nlin
    # Find out Dimensions
    numRows = d.N  # Number of one hot constraints

    BoxConstrainedStates = (d.lb.!=-Inf) .| (d.ub .!= Inf)
    numBoxConstraints = sum(BoxConstrainedStates)
    numRows += numBoxConstraints * d.N # Add number of box constraints

    c_x0 = d.c(d.x0, d.t0)
    numGeneralConstraints = isnothing(c_x0) ? 0 : length(c_x0)
    numRows += numGeneralConstraints * d.N # Add number of general constraints

    TerminallyConstrainedStates = (d.x_terminal .!=-Inf) .& (d.x_terminal .!= Inf)
    numTerminalConstraints = sum(TerminallyConstrainedStates)
    numRows += numTerminalConstraints # Add number of terminal constraints

    constraint = zeros(numRows)
    # Evaluate One-Hot-Constraint
    constraint[1:d.N] = d.Ag * x

    # Evaluate Box constraint
    currentRow = d.N
    if numBoxConstraints > 0
        idxBoxConstr = findall(>(0), BoxConstrainedStates)
        for i=1:d.N
            currentIdxStates = i*nlin+1
            constraint[currentRow+1:currentRow+numBoxConstraints] = d.xpts[idxBoxConstr, currentIdxStates]
            currentRow += numBoxConstraints
        end
    end

    # Evaluate general constraint at the N grid points
    if numGeneralConstraints > 0
        for i=1:d.N
            currentIdx = i*nlin+1
            currentTime = currentIdx*d.xi
            constraint[currentRow+1:currentRow+numGeneralConstraints] = d.c(d.xpts[:, currentIdx],currentTime)
            currentRow += numGeneralConstraints
        end
    end

    # Evaluate terminal constraint
    if numTerminalConstraints > 0
        idxTermConstr = findall(>(0), TerminallyConstrainedStates)
        constraint[currentRow+1:currentRow+numTerminalConstraints] = d.xpts[idxTermConstr, end]
    end
    for i in eachindex(constraint)
        g[i] = constraint[i]
    end
end


function MathOptInterface.eval_constraint_jacobian(d::POCev, J, x)
    if d.omega_jac_prev != x
        if d.omega_grad_prev != x
            precomputations_gradient!(d, x)
            d.omega_grad_prev[:] = x
        end
        computePhi!(d)
        d.omega_jac_prev[:] = x
    end

    # Find out Dimensions
    numRows = d.N  # Number of one hot constraints
    linearSystem = typeof(d) <: linPOCev
    nx = linearSystem ? d.nx : d.nx-1
    nlin = linearSystem ? 1 : d.nlin

    BoxConstrainedStates = (d.lb.!=-Inf) .| (d.ub .!= Inf)
    numBoxConstraints = sum(BoxConstrainedStates)
    numRows += numBoxConstraints * d.N # Add number of box constraints

    c_x0 = d.c(d.x0, d.t0)
    numGeneralConstraints = isnothing(c_x0) ? 0 : length(c_x0)
    numRows += numGeneralConstraints * d.N # Add number of general constraints

    TerminallyConstrainedStates = (d.x_terminal .!=-Inf) .& (d.x_terminal .!= Inf)
    numTerminalConstraints = sum(TerminallyConstrainedStates)
    numRows += numTerminalConstraints # Add number of terminal constraints

    Jacobian = zeros(numRows, d.N*d.n_omega)

    # One-Hot-Constraint
    Jacobian[1:d.N, :] = d.Ag

    # Check if box constraints given
    currentRow = d.N
    if numBoxConstraints > 0
        idxBoxConstr = findall(>(0), BoxConstrainedStates)
        for i=1:d.N                 # Box constraint for state at end of this control interval
            for ii=1:i              # All constrol intervals up to i influence this constraint
                for k=1:d.n_omega
                    derivative = (d.Phi[:,:,ii+1,i+1] * d.C[:,:,ii,k] * d.Phi[:,:,1,ii] * d.x0)
                    Jacobian[currentRow+(i-1)*numBoxConstraints+1:currentRow+i*numBoxConstraints, (ii-1)*d.n_omega+k] = derivative[idxBoxConstr]
                end
            end
        end
        currentRow += d.N * numBoxConstraints
    end

    # Evaluate derivative of general (nonlinear) constraints.
    # Apply chain rule, inner times outer derivative
    if numGeneralConstraints > 0
        for i=1:d.N         # General constraint for state at end of this control interval
            currentTime = ((i+1)*nlin+1)*d.xi
            outerDerivative = d.jacobian_c(d.xpts[1:nx,i*nlin+1], currentTime)
            for ii=1:i        # All constrol intervals up to i influence this constraint
                for k=1:d.n_omega
                    derivative = (d.Phi[:,:,ii+1,i+1] * d.C[:,:,ii,k] * d.Phi[:,:,1,ii] * d.x0)
                    derivative = linearSystem ? derivative : derivative[1:nx]
                    Jacobian[currentRow+(i-1)*numGeneralConstraints+1:currentRow+i*numGeneralConstraints, (ii-1)*d.n_omega+k] = outerDerivative*derivative
                end
            end
        end
        currentRow += d.N * numGeneralConstraints
    end

    # Terminal constraint
    if numTerminalConstraints > 0
        idxTermConstr = findall(>(0), TerminallyConstrainedStates)
        for k=1:d.n_omega
            for i=1:d.N
                derivative = (d.Phi[:,:,i+1,d.N+1] * d.C[:,:,i,k] * d.Phi[:,:,1,i] * d.x0)[idxTermConstr]
                Jacobian[currentRow+1:currentRow+numTerminalConstraints,(i-1)*d.n_omega+k] = derivative
            end
        end
    end

    for i=1:length(d.Ig)
        J[i] = Jacobian[d.Ig[i], d.Jg[i]]
    end
end

# Jacobian structure for POC
function MathOptInterface.jacobian_structure(d::POCev)
    a = [(d.Ig[i], d.Jg[i]) for i=1:size(d.Ig)[1]]
    return a
end

# Hessian structure for POC
function MathOptInterface.hessian_lagrangian_structure(d::POCev)
    a = [(d.Itril[i], d.Jtril[i]) for i=1:size(d.Itril)[1]]
    return a
end
