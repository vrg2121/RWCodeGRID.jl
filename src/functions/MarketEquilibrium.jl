# this module contains functions for calculating the market equilibrium in Market.jl
# functions defined: hessinterior, obj, mycon

module MarketEquilibrium
export hessinterior, obj, mycon, obj2, Price_Solve, wage_update_ms, grad_f, Price_Solve2

import DrawGammas: StructParams



function obj(Inputvec::Vector, power::Matrix, shifter::Matrix, KFshifter::Union{SubArray, Vector}, 
        KRshifter::Vector, p_F::Union{Float64, Vector, Int}, params::StructParams)
    mid = length(Inputvec) ÷ 2
    Dvec = Inputvec[1:mid]
    Yvec = Inputvec[1+mid:end]
    #Dsec = Dvec .* ones(1, params.I)
    Dsec = broadcast(*, Dvec, ones(1, params.I))

    power2 = 1 / params.alpha1
    value1 = -(Dsec .^ power) .* shifter
    value1 = sum(value1, dims=2)
    value2 = @. p_F * ((Yvec - KRshifter) / KFshifter ^ params.alpha2) ^ power2
    value = sum(value1) + sum(value2)    
    return value
    
end

function obj2(Inputvec::Vector, power::Float64, shifter::Float64, KFshifter::Float64, 
            KRshifter::Float64, p_F::Union{Float64, Int64}, params::StructParams)
    mid = length(Inputvec) ÷ 2
    Dvec = Inputvec[1:mid]
    Yvec = Inputvec[1+mid:end]
    Dsec = repeat(Dvec, 1, params.I)
    power2 = 1 / params.alpha1
    value1 = -Dsec .^ power .* shifter
    value2 = @. p_F * ((Yvec - KRshifter) / KFshifter ^ params.alpha2) ^ power2
    value1 = sum(value1, dims=2)
    value = sum(value1) + sum(value2)

    return value
    
end

function Price_Solve(Inputvec::Vector{Float64}, shifter::Union{Matrix, Float64}, Jlength::Int64, params::StructParams)
    mid = length(Inputvec) ÷ 2
    Dvec = Inputvec[1:mid]

    prices = sum(((params.Vs[:,2]' .* ones(Jlength, 1)) .+ 
            (params.Vs[:,3]' .* ones(Jlength, 1))) .* ((Dvec) .^ ((params.Vs[:,2]' .* ones(Jlength, 1)) .+ 
            (params.Vs[:,3]' .* ones(Jlength, 1)) .- 1)) .* shifter, dims=2)

    return vec(prices)
end

function location_prices!(pijs::Vector{Matrix{Float64}}, PCs::Matrix{Float64}, Xjdashs::Matrix{Float64}, Yjdashs::Matrix{Float64},
    w0::Vector{Float64}, p_E_D::Vector{Float64}, params, Ej::Matrix{Float64}, 
    p_F::Union{Float64, Vector, Int}, r::Matrix{Float64})

    for i = 1:params.I
        # retrieve slices for i
        ttau = params.tau[i]

        @views tpijs = ttau .* w0 .^ params.Vs[i, 1] .* p_E_D .^ (params.Vs[i, 2] + params.Vs[i, 3]) .* 
        (params.kappa + (params.kappa .* p_F ./ p_E_D) .^ (1 - params.psi)) .^ (-(params.psi ./ (params.psi - 1)) .* params.Vs[i, 3]) .*
        r .^ params.Vs[i, 4] ./
        (params.Z .* params.zsector[:, i] .* params.cdc)

        PCs[:, i] = (sum(tpijs .^ (1 - params.sig), dims=1)) .^ (1 / (1 - params.sig))
        @views Xjdashs[:, i] = sum((tpijs) .^ (1 - params.sig) .* (params.betaS[:, i] .* Ej ./ PCs[:, i].^(1 - params.sig))', dims=2)
        @views Yjdashs[:, i] = sum((tpijs).^(-params.sig) .* (params.betaS[:, i] .* Ej ./ PCs[:, i].^(1 - params.sig))', dims=2)
        pijs[i] .= tpijs
    end
end

function update_wage_data!(params::StructParams, w0::Union{Matrix, Vector}, pES::Union{Vector, Matrix},
    p_F::Union{Float64, Vector, Int}, r::Union{Vector, Matrix}, Ej, PCs::Matrix, Xjdashs::Matrix,
    Yjdashs::Matrix)

    sig = params.sig                    # common exponent
    one_minus_sig = 1 - sig
    inv_exponent = 1 / one_minus_sig     # for later use

    Threads.@threads :static for i in 1:params.I
        tpjs = Matrix{Float64}(undef, 2531, 2531)

        # Precompute parameters for the i-th iteration.
        τ       = params.tau[i]
        exp1    = params.Vs[i, 1]
        exp23   = params.Vs[i, 2] + params.Vs[i, 3]
        exp34   = -(params.psi / (params.psi - 1)) * params.Vs[i, 3]
        exp4    = params.Vs[i, 4]
        κ       = params.kappa
        ψ       = params.psi
        Z       = params.Z
        zsec    = @view params.zsector[:, i]
        cdc     = params.cdc
        beta    = @view params.betaS[:, i]
        
        # Compute tpijs with fused broadcast.
        @. tpjs = τ * (w0 ^ exp1) * (pES ^ exp23) *
                   ((κ + (κ * p_F / pES) ^ (1 - ψ)) ^ exp34) *
                   (r ^ exp4) / (Z * zsec * cdc)
        
        # Compute temporary array: tpijs^(1 - sig) is used twice.
        temp = tpjs .^ one_minus_sig
        
        # Calculate PCs along the columns (summing along rows).
        PCs[:, i] = (sum(temp, dims=1)) .^ inv_exponent
        
        # Compute factor (note: ensure dimensions align with your intended broadcasting).
        factor = (@. beta * Ej / (PCs[:, i] ^ one_minus_sig))'
        
        # Update Xjdashs and Yjdashs with fused operations.
        Xjdashs[:, i] = sum(temp .* factor, dims=2)
        Yjdashs[:, i] = sum((tpjs .^ (-sig)) .* factor, dims=2)
    end

end

function price_adjustments!(PC::Vector{Float64}, PCs::Array{Float64}, params::StructParams, w0::Union{Vector, Matrix}, 
                        Xjdashs::Array{Float64}, Xj::Union{Vector, Matrix}, pES::Union{Vector, Matrix}, 
                        pED::Union{Vector, Matrix}, p_F::Union{Float64, Vector, Int}, W_Real::Vector{Float64}, 
                        w_adjustment_factor::Union{Vector, Matrix}, Xjdash::Matrix{Float64})
    updw=0.5
    PC .= prod(PCs .^ params.betaS, dims=2)
    Xjdash .= sum(Xjdashs, dims=2)
    w_adjustment_factor .= w0 .* min.(max.(1 .+ updw .* (Xjdash .- Xj) ./ Xj, 0.2), 1.1)
    w0 .= w_adjustment_factor ./ (w_adjustment_factor[1])
    pES .= pES ./ (w_adjustment_factor[1])
    pED .= pED ./ (w_adjustment_factor[1])
    p_F = p_F / (w_adjustment_factor[1])
    W_Real .= w0 ./ PC
end

function wage_update_ms(w::Union{Vector, Matrix}, p_E_D::Union{Vector, Matrix},p_E_S::Union{Vector, Matrix}, p_F::Union{Float64, Vector, Int}, D_E::Vector, Y_E::Vector, 
    r::Union{Vector, Matrix}, KP::Union{Vector, Matrix}, Pi::Vector, fossil::Union{Vector, Matrix, Int}, params::StructParams)
    
    # intermediate variable allocations
    PCs = Array{Float64}(undef, 2531, 10)
    Xjdashs = Array{Float64}(undef, 2531, 10)
    Yjdashs = Array{Float64}(undef, 2531, 10)
    w_adjustment_factor = Matrix{Float64}(undef, 2531, 1)
    W_Real = Vector{Float64}(undef, 2531)
    PC = Vector{Float64}(undef, 2531)
    Xjdash = Matrix{Float64}(undef, 2531, 1)

    w0 = copy(w)
    pED = copy(p_E_D)
    pES = copy(p_E_S)

    Xj = @. w0 * params.L + pED * D_E + r * KP  
    Ej = @. w0 * params.L + pES * Y_E + r * KP + Pi + fossil

    update_wage_data!(params, w0, pED, p_F, r, Ej, PCs, Xjdashs, Yjdashs)
    
    # calculate price indices and adjust wages
    price_adjustments!(PC, PCs, params, w0, Xjdashs, Xj, pES, pED, p_F, W_Real, w_adjustment_factor, Xjdash)

    return w0, W_Real, sum(Xj), PC, Xjdashs, PCs, Yjdashs, Xj

end

# ---------------------------- Archived Functions ---------------------------- #

"""

function update_wage_data()
    for i = 1:params.I
        @views tpijs .= params.tau[i] .* 
                        w0 .^ params.Vs[i, 1] .* 
                        pES .^ (params.Vs[i, 2] + params.Vs[i, 3]) .* 
                        (params.kappa + (params.kappa .* p_F ./ pES) .^ (1 - params.psi)) .^ 
                        (-(params.psi / (params.psi - 1)) * params.Vs[i, 3]) .*
                        r .^ params.Vs[i, 4] ./
                        (params.Z .* params.zsector[:, i] .* params.cdc)
                        # this element takes 2156 profiling counts 
        PCs[:, i] = (sum(tpijs .^ (1 - params.sig), dims=1)) .^ (1 / (1 - params.sig))

        factor = (params.betaS[:, i] .* Ej ./ PCs[:, i].^(1 - params.sig))'
        Xjdashs[:, i] = sum(tpijs .^ (1 - params.sig) .* factor, dims=2)
        Yjdashs[:, i] = sum(tpijs .^ (-params.sig) .* factor, dims=2)
    end
end

function mycon(Inputvec::Vector, mat::Matrix, params::StructParams)
    mid = length(Inputvec) ÷ 2

    Dvec = @view Inputvec[1:mid]
    Yvec = @view Inputvec[1+mid:end]
    Pvec = Yvec .- Dvec
    scal = params.Rweight .* Pvec[2:end]' * mat * Pvec[2:end]

    @views ceq = sum(Pvec) - scal[1] - Pvec[1]^2
    return ceq
end

function grad_f(Inputvec, power, shifter, KFshifter, KRshifter, p_F, params)
    mid = length(Inputvec) ÷ 2
    Dvec = Inputvec[1:mid]
    Yvec = Inputvec[1+mid:end]
    Dsec = repeat(Dvec, 1, params.I)
    power2 = 1 / params.alpha1

    grad1f = -sum((power).* Dsec .^ (power .- 1) .* shifter, dims=2)
    grad2f = power2 .* p_F .* (Yvec .- KRshifter) .^ (power2-1) .* (1 ./ KFshifter .^ params.alpha2).^ (power2)
    gradf = sparse([grad1f, grad2f])
    return gradf
end


function hessinterior(Inputvec, lambda, power, shifter, KFshifter, KRshifter, p_F, mat, params)
    Dvec = Inputvec[1:end÷2]  # ÷ is integer division
    Yvec = Inputvec[end÷2+1:end]
    J = length(Dvec)

    Dsec = repeat(Dvec, 1, params.I)
    power2 = (1 / params.alpha1)

    piece1 = -sum((((power .- 1) .* power) .* Dsec .^ (power .- 2) .* shifter), dims=2)
    piece1 = Diagonal(piece1)

    piece2 = (power2 - 1) .* power2 .* p_F .* (Yvec .- KRshifter) .^ (power2 - 2) .* (1 ./ KFshifter .^ params.alpha2) .^ power2
    piece2 = Diagonal(piece2)

    f2 = [piece1 zeros(J, J); zeros(J, J) piece2]
    f2 = sparse(f2)

    xmat = params.Rweight * 2 * mat
    zeros1block = zeros(1, J - 1)
    zeros2block = zeros(J - 1, 1)

    hessc = [-2 zeros1block' 2 zeros1block';
             zeros2block -xmat zeros2block xmat;
             2 zeros1block' -2 zeros1block';
             zeros2block xmat zeros2block -xmat]

    hessc = sparse(hessc)

    h = sparse(f2 + lambda * hessc)
    return h  
end
"""


end