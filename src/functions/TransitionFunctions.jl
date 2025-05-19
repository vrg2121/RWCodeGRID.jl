module TransitionFunctions

import DataFrames: DataFrame
using JuMP, Ipopt, Interpolations
import LinearAlgebra: Transpose, I, Adjoint, mul!
import Random: Random
import SparseArrays: sparse
import ..RegionModel: solve_model

# import relevant structs
import ..DataLoadsFunc: StructGsupply, StructRWParams
import DrawGammas: StructParams, StructAllParams
import ..MarketFunctions: StructMarketEq
import ..SteadyStateFunctions: StructPowerOutput

using ..MarketEquilibrium

export solve_transition_eq, StructTransEq

# ---------------------------------------------------------------------------- #
#                             Functions for Guesses                            #
# ---------------------------------------------------------------------------- #

function guess_path_S!(KR_path_S::Matrix{Float64}, sseq::StructPowerOutput, KR_init_S::Matrix{Float64}, expo::Matrix{Float64})
    @inbounds for kk in 1:501
        @. KR_path_S[:, kk] =  sseq.KR_LR_S + ((KR_init_S - sseq.KR_LR_S) * expo[kk])
    end
end

function guess_path_W!(KR_path_W::Matrix{Float64}, sseq::StructPowerOutput, KR_init_W::Matrix{Float64}, expo::Matrix{Float64})
    @inbounds for kk in 1:501
        @. KR_path_W[:, kk] = sseq.KR_LR_W + ((KR_init_W - sseq.KR_LR_W) * expo[kk])
    end
end

function guess_path_p_E!(p_E_path_guess::Matrix{Float64}, sseq::StructPowerOutput, mrkteq::StructMarketEq, expo1::Matrix{Float64})
    @inbounds for kk in 1:501
        @. p_E_path_guess[:, kk] = sseq.p_E_LR + ((mrkteq.p_E_init - sseq.p_E_LR) * expo1[kk])
    end
end

function guess_path_D!(D_path::Matrix{Float64}, sseq::StructPowerOutput, mrkteq::StructMarketEq, expo1::Matrix{Float64})
    @inbounds for kk in 1:501
        @. D_path[:, kk] = sseq.D_LR + ((mrkteq.D_init - sseq.D_LR) * expo1[kk])
    end
end

function guess_path_Y!(Y_path::Matrix{Float64}, sseq::StructPowerOutput, mrkteq::StructMarketEq, expo1::Matrix{Float64})
    @inbounds for kk in 1:501
        @. Y_path[:, kk] = sseq.YE_LR + ((mrkteq.YE_init - sseq.YE_LR) * expo1[kk])
    end
end

function guess_path_PC!(PC_path_guess::Matrix{Float64}, sseq::StructPowerOutput, mrkteq::StructMarketEq, expo1::Matrix{Float64})
    @inbounds for kk in 1:501
        @. PC_path_guess[:, kk] = sseq.PC_guess_LR + ((mrkteq.PC_guess_init - sseq.PC_guess_LR) * expo1[kk])
    end
end

function guess_path_PI!(PI_path::Matrix{Float64}, sseq::StructPowerOutput, mrkteq::StructMarketEq, expo1::Matrix{Float64})
    @inbounds for kk in 1:501
        @. PI_path[:, kk] = sseq.PI_LR + ((mrkteq.PI_init - sseq.PI_LR) * expo1[kk])
    end
end

function guess_path_w!(w_path_guess::Matrix{Float64}, sseq::StructPowerOutput, wage_init::Vector{Float64}, expo1::Matrix{Float64})
    @inbounds for kk in 1:501
        @. w_path_guess[:, kk] = sseq.w_LR + ((wage_init - sseq.w_LR) * expo1[kk])
    end
end

function guess_path_B!(B_path::Matrix{Float64}, sseq::StructPowerOutput, RWParams::StructRWParams, expo::Matrix{Float64})
    @inbounds for kk in 1:501
        @. B_path[:, kk] = sseq.KR_LR * 2 + ((RWParams.KR * 2 - sseq.KR_LR * 2) * expo[kk])
    end
end

function guess_path_hours!(hoursofstorage_path::Matrix{Float64}, curtailmentswitch::Int, expo::Matrix)
    @inbounds for kk in 1:501
        if curtailmentswitch == 1
            hoursofstorage_path[:, kk] .= 12 + (0-12) * expo[kk]
        else
            @. hoursofstorage_path[:, kk] = 0
        end
    end
end

function guess_path_renewshare!(renewshare_path::Matrix{Float64}, expo::Matrix)
    @inbounds for kk in 1:501
        renewshare_path[:, kk] .= 1 + (0-1) * expo[kk]
    end
end

function guess_path_p_KR_bar!(p_KR_bar_path::Matrix{Float64}, sseq::StructPowerOutput, p_KR_bar_init::Matrix{Float64}, expo1::Matrix{Float64})
    @inbounds for kk in 1:501
        @. p_KR_bar_path[:, kk] = sseq.p_KR_bar_LR + ((p_KR_bar_init - sseq.p_KR_bar_LR) * expo1[kk])
    end
end

function guess_path_p_KR_S!(p_KR_path_guess_S::Matrix{Float64}, sseq::StructPowerOutput, p_KR_init_S::Float64, expo::Matrix{Float64})
    @inbounds for kk in 1:501
        p_KR_path_guess_S[kk] = sseq.p_KR_LR_S + ((p_KR_init_S - sseq.p_KR_LR_S) * expo[kk])
    end
end

function guess_path_p_KR_W!(p_KR_path_guess_W::Matrix{Float64}, sseq::StructPowerOutput, p_KR_init_W::Float64, expo::Matrix{Float64})
    @inbounds for kk in 1:501
        p_KR_path_guess_W[kk] = sseq.p_KR_LR_W + ((p_KR_init_W - sseq.p_KR_LR_W) * expo[kk])
    end
end

function guess_path_KF!(KF_path::Matrix{Float64}, sseq::StructPowerOutput, RWParams::StructRWParams, expo2::Matrix)
    @inbounds for kk in 1:501
        KF_path[:, kk] .= sseq.KF_LR.+((RWParams.KF.-sseq.KF_LR) .* expo2[kk])
    end
end

function guess_path_fossil!(p_F_path_guess::Matrix{Float64}, T::Int64)
    r = 0.05 * ones(T + 1) 
    pF = ones(T + 1) .* 0.05 .* (1 .+ r / 2) .^ LinRange(1, T + 1, T + 1)
    p_F_path_guess .= transpose(pF)
end

function sum_bat(KR::Matrix{Float64}, hoursofstorage_path::Matrix{Float64}, result::Float64)
    for i in 1:length(KR)
        result += KR[i] * hoursofstorage_path[i, 1]
    end
end

function guess_path_p_B!(p_B_path_guess::Matrix, sseq::StructPowerOutput, hoursofstorage_path::Matrix{Float64}, gammaB::Float64, Initialprod::Int, 
    RWParams::StructRWParams, expo1::Matrix)

    result = 0.0
    sum_bat(RWParams.KR, hoursofstorage_path, result) # 940.909 ns (0 allocations: 0 bytes)
    p_B_init = (Initialprod + result)^(-gammaB) # 81.761 ns (4 allocations: 64 bytes)

    @inbounds for kk in 1:501
        @. p_B_path_guess[:, kk] = sseq.p_B + ((p_B_init - sseq.p_B) * expo1[kk])
    end
end

# ---------------------------------------------------------------------------- #
#                             Sectoral Allocations                             #
# ---------------------------------------------------------------------------- #
function fill_lpg!(lpg::Array{Float64,3}, expo1::Matrix{Float64})
    expo1_reshaped = reshape(expo1, 1, :, 1)
    lpg .= expo1_reshaped
end

function sectoral_allocations!(Lsectorpath_guess::Array{Float64, 3}, laboralloc_path::Array{Float64, 3}, laboralloc_init::Matrix, sseq::StructPowerOutput, T::Int, expo1::Matrix{Float64})
    
    lpg = Array{Float64}(undef, 2531, 501, 10) # 2 allocations, 1.164 ms

    fill_lpg!(lpg, expo1); #6.884 ms (2 allocations: 96 bytes)

    Lsectorpath_guess .= permutedims(lpg, [1, 3, 2]); # 30.283 ms (9 allocations: 96.74 MiB)
    @. Lsectorpath_guess = sseq.laboralloc_LR + (laboralloc_init - sseq.laboralloc_LR) * Lsectorpath_guess;
    Lsectorpath_guess .= Lsectorpath_guess .* (params.L .* ones(1, params.I, T + 1)); #9.443 ms (6 allocations: 39.38 KiB)'
    laboralloc_path .= Lsectorpath_guess ./ (params.L .* ones(1, params.I, T + 1))#11.185 ms (6 allocations: 224 bytes)

end

# ---------------------------------------------------------------------------- #
#                          Functions to update prices                          #
# ---------------------------------------------------------------------------- #
function calculate_I_path!(I_path, KR_path, Depreciation, nrows, ncols)
    # Assuming I_path is already allocated with correct size (1, ncols-1)
    @inbounds for col in 1:(ncols-1)
        col_sum = 0.0
        for row in 1:nrows
            # Calculate difference between current and previous capital
            diff = KR_path[row, col+1] - KR_path[row, col] + Depreciation[row, col]
            # Apply max(diff, 0) and add to sum
            if diff > 0
                col_sum += diff
            end
        end
        I_path[1, col] = col_sum
    end
    return I_path
end

function Qtotal_calc!(Qtotal_path_S::Matrix{Float64}, Qtotal_path_W::Matrix{Float64}, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, 
    KR_init_S::Matrix{Float64}, KR_init_W::Matrix{Float64}, deltaR::Float64, iota::Float64, Initialprod::Int)
    Depreciation_S = Matrix{Float64}(undef, 2531, 501)
    Depreciation_W = Matrix{Float64}(undef, 2531, 501)

    mul!(Depreciation_S, KR_path_S, deltaR) # 649.700 μs (0 allocations: 0 bytes)
    mul!(Depreciation_W, KR_path_W, deltaR) # 653.800 μs (0 allocations: 0 bytes)
    
    # Preallocate arrays
    nrows_S = size(KR_path_S, 1)
    ncols_S = size(KR_path_S, 2)
    nrows_W = size(KR_path_W, 1)
    ncols_W = size(KR_path_W, 2)
    
    I_path_S = zeros(1, ncols_S-1)
    I_path_W = zeros(1, ncols_W-1)
    
    # Calculate I_path values without temporary allocations
    calculate_I_path!(I_path_S, KR_path_S, Depreciation_S, nrows_S, ncols_S) # 829.100 μs (0 allocations: 0 bytes)
    calculate_I_path!(I_path_W, KR_path_W, Depreciation_W, nrows_W, ncols_W)

    l = length(I_path_S)
    power_vector = Vector{Float64}(undef, l)
    @. power_vector = iota ^ (1:l) # 4.188 μs (6 allocations: 160 bytes)
    s_inprod_sum = sum(KR_init_S) + Initialprod
    w_inprod_sum = sum(KR_init_W) + Initialprod

    @inbounds for i=1:l
        pv = @view power_vector[1:i] # 19.960 ns (2 allocations: 80 bytes)
        cumsum_S = sum(reverse(I_path_S[1:i]) .* pv) # 506.771 ns (7 allocations: 2.80 KiB)
        cumsum_W = sum(reverse(I_path_W[1:i]) .* pv) # 474.490 ns (7 allocations: 2.73 KiB) * 500            

        Qtotal_path_S[i+1] = cumsum_S + (s_inprod_sum)*(iota).^(i+1) # 337.838 ns (6 allocations: 128 bytes)
        Qtotal_path_W[i+1] = cumsum_W + (w_inprod_sum)*(iota).^(i+1)
    end

    Qtotal_path_S[1] = sum(KR_init_S) + Initialprod # 136.426 ns (2 allocations: 32 bytes)
    Qtotal_path_W[1] = sum(KR_init_W) + Initialprod # 136.364 ns (2 allocations: 32 bytes)

end

function up_cap!(p_KR_path::Matrix{Float64}, Qtotal_path::Matrix{Float64}, gamma::Float64)
    @. p_KR_path = Qtotal_path ^ (-gamma)
end

function up_cap_path!(p_KR_path_guess::Matrix{Float64}, p_KR_path::Matrix{Float64})
    @. p_KR_path_guess = (0.5) * p_KR_path + 0.5 * p_KR_path_guess
end

function update_capital_price!(p_KR_path_S::Matrix{Float64}, p_KR_path_W::Matrix{Float64}, p_KR_path_guess_S::Matrix{Float64}, 
    p_KR_path_guess_W::Matrix{Float64}, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, 
    params::StructParams, Initialprod::Int, KR_init_S::Matrix{Float64}, KR_init_W::Matrix{Float64})

    # initialize intermediate variables
    Qtotal_path_S = zeros(1, 501)
    Qtotal_path_W = zeros(1, 501)
    
    Qtotal_calc!(Qtotal_path_S, Qtotal_path_W, KR_path_S, KR_path_W, KR_init_S, KR_init_W, params.deltaR, params.iota, Initialprod) # 7.476 ms (13526 allocations: 25.79 MiB)

    # update capital prices and paths
    up_cap!(p_KR_path_S, Qtotal_path_S, params.gammaS) # 4.757 μs (1 allocation: 16 bytes)
    up_cap!(p_KR_path_W, Qtotal_path_W, params.gammaW) # 4.757 μs (1 allocation: 16 bytes)
    up_cap_path!(p_KR_path_guess_S, p_KR_path_S) # 749.180 ns (0 allocations: 0 bytes)
    up_cap_path!(p_KR_path_guess_W, p_KR_path_W) # 747.541 ns (0 allocations: 0 bytes)
end

# ---------------------------------------------------------------------------- #
#                              Update Solar Shares                             #
# ---------------------------------------------------------------------------- #
function up_SShare!(SShare_path::Matrix{Float64}, thetaS::Matrix{Float64}, p_KR_path_guess_S::Matrix{Float64}, varrho::Float64, thetaW::Matrix{Float64}, 
    p_KR_path_guess_W::Matrix{Float64})
    @. SShare_path = ((thetaS / p_KR_path_guess_S) ^ varrho) / (((thetaS / p_KR_path_guess_S) ^
        varrho) + ((thetaW / p_KR_path_guess_W) ^ varrho))
end

function up_thbar!(thetabar_path::Matrix{Float64}, thetaS::Matrix{Float64}, SShare_path::Matrix{Float64}, thetaW::Matrix{Float64})
    @. thetabar_path = thetaS * SShare_path + thetaW * (1 - SShare_path)
end 

function up_KR_bar!(p_KR_bar_path::Matrix{Float64}, SShare_path::Matrix{Float64}, p_KR_path_guess_S::Matrix{Float64}, p_KR_path_guess_W::Matrix{Float64})
    @. p_KR_bar_path = SShare_path * p_KR_path_guess_S + (1 - SShare_path) * p_KR_path_guess_W
end

# ---------------------------------------------------------------------------- #
#                             Update Battery Prices                            #
# ---------------------------------------------------------------------------- #
function up_KR_path!(KR_path::Matrix{Float64}, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64})
    @. KR_path = KR_path_S + KR_path_W
end

function up_DepB!(DepreciationB::Matrix{Float64}, KR_path::Matrix{Float64}, deltaB::Float64, hoursofstorage_path::Matrix{Float64})
    @. DepreciationB = KR_path * deltaB * hoursofstorage_path #   1.048 ms (5 allocations: 144 bytes)
end

function calculate_I_path_B!(I_path, KR_path, Depreciation, hoursofstorage_path, nrows, ncols)
    # Assuming I_path is already allocated with correct size (1, ncols-1)
    @inbounds for col in 1:(ncols-1)
        col_sum = 0.0
        for row in 1:nrows
            # Calculate difference between current and previous capital
            diff = (KR_path[row, col+1] - KR_path[row, col]) * hoursofstorage_path[row, col] + Depreciation[row, col]
            # Apply max(diff, 0) and add to sum
            if diff > 0
                col_sum += diff
            end
        end
        I_path[1, col] = col_sum
    end
    return I_path
end

function QtotalB_calc!(Qtotal_path_B::Matrix{Float64}, iota::Float64, KR_path::Matrix{Float64}, deltaB::Float64, hoursofstorage_path::Matrix{Float64}, 
    KR::Matrix{Float64}, Initialprod::Int)

    DepreciationB = Matrix{Float64}(undef, 2531, 501)
    up_DepB!(DepreciationB, KR_path, deltaB, hoursofstorage_path) # 998.300 μs (1 allocation: 16 bytes)
    
    nrows = size(KR_path, 1)
    ncols = size(KR_path, 2)
    I_path_B = Matrix{Float64}(undef, 1, 500)
    calculate_I_path_B!(I_path_B, KR_path, DepreciationB, hoursofstorage_path, nrows, ncols) # 1.149 ms (0 allocations: 0 bytes)

    l = length(I_path_B)
    power_vector = Vector{Float64}(undef, l)
    @. power_vector = iota ^ (1:l) # 4.188 μs (6 allocations: 160 bytes)
    inprod_sum = sum(KR .* view(hoursofstorage_path, :, 1)) + Initialprod #  4.225 μs (8 allocations: 20.08 KiB

    for i=1:l
        pv = @view power_vector[1:i] # 19.960 ns (2 allocations: 80 bytes)
        cumsum_B = sum(reverse(I_path_B[1:i]) .* pv) # 506.771 ns (7 allocations: 2.80 KiB)
        Qtotal_path_B[i+1] = cumsum_B + (inprod_sum)*(iota).^(i+1) # 311.382 ns (5 allocations: 112 bytes)
    end
end

function update_battery_prices!(KR_path::Matrix{Float64}, Qtotal_path_B::Matrix{Float64}, # variables to be modified
    KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, params::StructParams, 
    hoursofstorage_path::Matrix{Float64}, RWParams::StructRWParams, Initialprod::Int, hoursofstorage::Int)    

    # update battery prices
    up_KR_path!(KR_path, KR_path_S, KR_path_W) # 1.074 ms (0 allocations: 0 bytes)

    QtotalB_calc!(Qtotal_path_B, params.iota, KR_path, params.deltaB, hoursofstorage_path, RWParams.KR, Initialprod) # 4.983 ms (5008 allocations: 12.88 MiB)

end

# ---------------------------------------------------------------------------- #
#                         Update Capital and Power Path                        #
# ---------------------------------------------------------------------------- #
function up_rP_path!(rP_path::Matrix{Float64}, r_path::Adjoint{Float64, Vector{Float64}}, deltaP::Float64, PC_path_guess::Matrix{Float64})
    @views @. rP_path[:, 1:end-1] = (r_path - 1 + deltaP) .* PC_path_guess[:, 1:end-1]
    @views @. rP_path[:, end] .= rP_path[:, end-1]
end

function up_pg_path!(pg_path_s::Array{Float64, 3}, w_path_guess::Matrix{Float64}, Vs::Matrix{Float64}, J::Int, p_E_path_guess::Matrix{Float64},
    kappa::Matrix{Float64}, p_F_path_guess::Matrix{Float64}, psi::Float64, Z::Matrix{Float64}, zsector::Matrix{Float64}, cdc::Float64)
    @views V1 = Vs[:, 1]' .* ones(J, 1)
    @views V2 = Vs[:,2]' .^ (ones(J, 1)) + (Vs[:,3]' .* ones(J, 1))
    @views V3 = Vs[:,4]' .* ones(J, 1)
    p1 = 1 - psi
    @views p2 = -(psi ./ (psi-1))* Vs[:,3]'
    zz = Matrix{Float64}(undef, 2531, 10)
    @. zz = Z * zsector * cdc

    @inbounds for jj=1:T+1
        @views pg_path_s[:,:,jj] .= w_path_guess[:, jj] .^ (V1) .* 
            p_E_path_guess[:,jj] .^ (V2) .*
            (kappa + (kappa .* p_F_path_guess[:,jj] ./ p_E_path_guess[:,jj]) .^ (p1)) .^ (p2) .*
            rP_path[:,jj] .^ (V3) ./ zz #   1.221 ms (95 allocations: 717.23 KiB)
    end # 0.699484 seconds (48.10 k allocations: 350.928 MiB, 4.89% gc time)
end

function up_YR_path!(YR_path::Matrix{Float64}, thetaS::Matrix{Float64}, KR_path_S::Matrix{Float64}, thetaW::Matrix{Float64}, KR_path_W::Matrix{Float64})
    @. YR_path = (thetaS * KR_path_S) + (thetaW * KR_path_W) #  10.019 ms (12 allocations: 29.02 MiB)
end

function data_set_up_transition(t::Int, kk::Int, majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc_path::Array, Lsectorpath_guess::Array, params::StructParams,
    w_path_guess::Union{Matrix, Vector}, rP_path::Matrix, pg_path_s::Array, p_E_path_guess::Union{Vector, Matrix}, kappa::Float64, regionParams::StructRWParams, KF_path::Matrix, p_F_path_guess::Matrix{Float64}, 
    linconscount::Int, KR_path_S::Matrix, KR_path_W::Matrix)
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    n = majorregions.n[kk]
    l_ind = Linecounts.rowid2[kk]:Linecounts.rowid[kk]
    gam = RWParams.Gam[kk]
    l_n = Linecounts.n[kk]

    secalloc = laboralloc_path[ind, :, t]
    Lshifter = Lsectorpath_guess[ind,:,t]
    Kshifter=Lsectorpath_guess[ind,:,t] .* (params.Vs[:,4]' .* ones(n,1)) ./ (params.Vs[:,1]' .* ones(n,1)) .*
            (w_path_guess[ind, t] ./ rP_path[ind,t])
    #Ltotal = sum(Lshifter, dims=2)

    # define data for inequality constraints
    linecons = RWParams.Zmax[l_ind]
    Gammatrix = zeros(size(gam, 1), size(gam, 2) + 1)
    Gammatrix[:, 2:end] = gam 
    if linconscount < l_n
        Random.seed!(1)
        randvec = rand(l_n)
        randvec = randvec .> (linconscount / l_n)
        Gammatrix[randvec, :] .= 0
    end
    stacker = [-Matrix(I, n, n) Matrix(I, n, n)]
    Gammatrix = sparse(Gammatrix) * sparse(stacker)
    Gammatrix = [Gammatrix; -Gammatrix]
    linecons = [linecons; linecons]

    # define shifters for objective function
    pg_s = pg_path_s[ind, :, t]
    p_F_in = p_F_path_guess[:,t]
    prices = p_E_path_guess[ind,t] .* ones(1, params.I)
    power = (params.Vs[:,2]' .* ones(n, 1)) .+ (params.Vs[:,3]' .* ones(n, 1))
    shifter = pg_s .*
            (kappa .+ (prices ./ (kappa .* p_F_in)) .^ (params.psi - 1)) .^ (params.psi / (params.psi-1) .* (params.Vs[:,3]' .* ones(n,1))) .*
            (1 .+ (params.Vs[:,3]' .* ones(n,1)) ./ (params.Vs[:,2]' .* ones(n, 1))) .^ (-(params.Vs[:,2]' .* ones(n,1)) .- (params.Vs[:,2]' .* ones(n,1))) .*
            params.Z[ind] .*
            params.zsector[ind, :] .*
            Lshifter .^ (params.Vs[:,1]' .* ones(n,1)) .*
            Kshifter .^ (params.Vs[:,4]' .* ones(n,1)) 
    shifter = shifter .* secalloc .^ power
    KRshifter = regionParams.thetaS[ind] .* KR_path_S[ind, t] .+ regionParams.thetaW[ind] .* KR_path_W[ind, t]
    KFshifter = KF_path[ind, t]

    # define bounds
    YFmax = KF_path[ind,t] # DIFFERENT
    LB = [zeros(n); KRshifter]
    UB = [fill(1000, n); YFmax .+ KRshifter .+ 0.01]

    # define guess
    guess = [KRshifter; KRshifter .+ 0.0001]
    l_guess = length(guess)
    mid = l_guess ÷ 2

    return l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid, p_F_in

end

# ---------------------------------------------------------------------------- #
#                                  Transition                                  #
# ---------------------------------------------------------------------------- #
function fill_shifter!(psi::Float64, pg_path_s::Array{Float64, 3}, kappa::Float64, p_F_in::Vector{Float64},
    Z::Matrix{Float64}, ind, zsector::Matrix{Float64}, Lsectorpath_guess::Array{Float64, 3}, Kshifter::Matrix{Float64}, V1::Matrix{Float64}, V2::Matrix{Float64},
    V3::Matrix{Float64}, V4::Matrix{Float64}, t::Int64, shifter::Matrix{Float64}, power::Matrix{Float64})
    
    p1 = psi - 1
    p2 = psi / p1

    @views prices = p_E_path_guess[ind, t] .* ones(1, params.I)
    @views pg_s = pg_path_s[ind, :, t]

    @views shifter .= pg_s .* (kappa .+ (prices ./ (kappa .* p_F_in)) .^ (p1)) .^ (p2 .* (V1)) .*
            (1 .+ (V1) ./ (V2)) .^ (-(V2) .- (V2)) .* Z[ind] .* zsector[ind, :] .* Lsectorpath_guess[ind,:,t] .^ (V3) .* Kshifter .^ (V4) # 505.100 μs (49 allocations: 60.48 KiB)
    @. @views shifter = shifter * laboralloc_path[ind, :, t] ^ power # 281.900 μs (8 allocations: 480 bytes)
end

function data_set_up_transition!(t::Int, kk::Int, majorregions::DataFrame, Lsectorpath_guess::Array, params::StructParams,
    w_path_guess::Union{Matrix, Vector}, rP_path::Matrix, pg_path_s::Array, kappa::Float64, regionParams::StructRWParams, KF_path::Matrix, 
    p_F_path_guess::Matrix{Float64}, KR_path_S::Matrix, KR_path_W::Matrix, LB::Vector{Float64}, UB::Vector{Float64}, guess::Vector{Float64}, power::Matrix{Float64}, 
    KFshifter::Vector{Float64}, KRshifter::Vector{Float64}, n::Int, p_F_in::Vector{Float64})

    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    shifter = Matrix{Float64}(undef, n, params.I) # 1.290 μs (2 allocations: 56.86 KiB)

    @views V1 = params.Vs[:,3]' .* ones(n,1)
    @views V2 = params.Vs[:,2]' .* ones(n,1)
    @views V3 = params.Vs[:,1]' .* ones(n,1)
    @views V4 = params.Vs[:,4]' .* ones(n,1)

    @views Kshifter = Lsectorpath_guess[ind,:,t] .* (V4) ./ (V3) .* (w_path_guess[ind, t] ./ rP_path[ind,t]) # 29.900 μs (16 allocations: 58.02 KiB

    # define shifters for objective function
    p_F_in .= p_F_path_guess[:,t]
    power .= (V2) .+ (V1)

    fill_shifter!(params.psi, pg_path_s, kappa, p_F_in, params.Z, ind, params.zsector, Lsectorpath_guess, Kshifter, V1, V2, V3, V4, t, shifter, power) # 569.300 μs (61 allocations: 118.64 KiB)

    @views KRshifter .= (regionParams.thetaS[ind] .* KR_path_S[ind, t]) + (regionParams.thetaW[ind] .* KR_path_W[ind, t]) # 1.870 μs (15 allocations: 6.94 KiB)
    @views KFshifter .= KF_path[ind, t]

    # define bounds
    @views K_UB = KF_path[ind,t] .+ KRshifter .+ 0.01
    LB .= [zeros(n); KRshifter]
    UB .= [fill(1000, n); K_UB]

    # define guess
    guess .= [KRshifter; KRshifter .+ 0.0001]
end

function fill_paths_og!(result_price_path::Matrix{Matrix{Float64}}, result_Dout_path::Matrix{Matrix{Float64}}, 
    result_Yout_path::Matrix{Matrix{Float64}}, result_YFout_path::Matrix{Matrix{Float64}}, kk::Int64, capT::Int64)
    @inbounds for t = 1:capT
        fill!(result_price_path[kk, t], 0)
        fill!(result_Dout_path[kk, t], 0)
        fill!(result_Yout_path[kk, t], 0)
        fill!(result_YFout_path[kk, t], 0)
    end #   0.000142 seconds (21 allocations: 672 bytes)

end

function set_up_opt!(Kshifter::Matrix{Float64}, guess::Vector{Float64}, p_F_in::Vector{Float64}, shifter::Matrix{Float64}, 
    KRshifter::Vector{Float64}, KFshifter::Vector{Float64}, YFmax::Vector{Float64}, ind::UnitRange{Int64},
    Lsectorpath_guess::Array{Float64, 3}, V1::Matrix{Float64}, V2::Matrix{Float64}, V3::Matrix{Float64}, V4::Matrix{Float64}, 
    w_path_guess::Matrix{Float64}, rP_path::Matrix{Float64}, D_path::Matrix{Float64}, Y_path::Matrix{Float64}, 
    p_F_path_guess::Matrix{Float64}, params::StructParams, pg_path_s::Array{Float64, 3}, power::Matrix{Float64}, regionParams::StructRWParams, 
    KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, KF_path::Matrix{Float64}, t::Int64, n::Int64)
    
    # set up optimization problem for region kk
    Kshifter .= Lsectorpath_guess[ind, :, t] .* V4 ./ V3 .* (w_path_guess[ind, t] ./ rP_path[ind, t])
    
    guess[1:n] .= D_path[ind, t]
    guess[n+1:end] .= Y_path[ind, t]

    @. p_F_in = p_F_path_guess[:, t] 

    fill_shifter!(params.psi, pg_path_s, kappa, p_F_in, params.Z, ind, params.zsector, Lsectorpath_guess, Kshifter, V1, V2, V3, V4, t, shifter, power) 

    @. @views KRshifter = regionParams.thetaS[ind] * KR_path_S[ind, t] + regionParams.thetaW[ind] * KR_path_W[ind, t]
    @views KFshifter .= KF_path[ind, t] 
    @views YFmax .= KF_path[ind, t] 
    #   0.000495 seconds (93 allocations: 126.641 KiB)
end

function transition_electricity_US_Europe!(result_price_path::Matrix, result_Dout_path::Matrix, result_Yout_path::Matrix, result_YFout_path::Matrix, # modified variables
    majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc_path::Array,
    Lsectorpath_guess::Array, params::StructParams, w_path_guess::Matrix{Float64}, rP_path::Matrix{Float64},
    linconscount::Int, pg_path_s::Array, p_F_path_guess::Matrix{Float64}, p_E_path_guess::Matrix{Float64},
    kappa::Float64, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, KF_path::Matrix{Float64}, capT::Int, regionParams::StructRWParams)

    for kk=1:2
        
        Threads.@threads :static for t in 1:capT
            local l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid, p_F_in = data_set_up_transition(t, kk, majorregions, Linecounts, RWParams, laboralloc_path, Lsectorpath_guess, params,
                                                                                            w_path_guess, rP_path, pg_path_s, p_E_path_guess, kappa, regionParams, KF_path, p_F_path_guess, 
                                                                                            linconscount, KR_path_S, KR_path_W)

            local P_out = solve_model(kk, l_guess, LB, UB, guess, regionParams, params, power, 
                                    shifter, KFshifter, KRshifter, p_F_in, mid)

            result_price_path[kk, t] .= Price_Solve(P_out, shifter, n, params)
            @views result_Dout_path[kk, t] .= P_out[1:mid]
            @views result_Yout_path[kk,t] .= P_out[1+mid:end]
            @views result_YFout_path[kk,t] .= P_out[1+mid:end] .- KRshifter

        end
        # 159.979501 seconds (23.40 M allocations: 8.741 GiB, 1.74% gc time, 3.29% compilation time) 
    end
end

function transition_electricity_other_countries!(result_price_path::Matrix, result_Dout_path::Matrix, result_Yout_path::Matrix, result_YFout_path::Matrix, # modified variables
    majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc_path::Array,
    Lsectorpath_guess::Array, params::StructParams, w_path_guess::Matrix{Float64}, rP_path::Matrix{Float64},
    linconscount::Int, pg_path_s::Array, p_F_path_guess::Matrix{Float64}, p_E_path_guess::Matrix{Float64},
    kappa, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, KF_path::Matrix{Float64}, capT::Int, regionParams::StructRWParams)

    
    for kk=3:(params.N - 1)
        Threads.@threads :static for t = 1:capT
            local l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid, p_F_in = data_set_up_transition(t, kk, majorregions, Linecounts, RWParams, laboralloc_path, Lsectorpath_guess, params,
                            w_path_guess, rP_path, pg_path_s, p_E_path_guess, kappa, regionParams, KF_path, p_F_path_guess, 
                            linconscount, KR_path_S, KR_path_W)

            local P_out = solve_model(kk, l_guess, LB, UB, guess, regionParams, params, power, 
                            shifter, KFshifter, KRshifter, p_F_in, mid)

            result_price_path[kk, t].= Price_Solve(P_out, shifter, n, params)
            @views result_Dout_path[kk, t] .= P_out[1:mid]
            @views result_Yout_path[kk,t] .= P_out[1+mid:end]
            @views result_YFout_path[kk,t] .= P_out[1+mid:end] .- KRshifter
        end

    end
end

function transition_electricity_off_grid!(result_price_path::Matrix{Matrix{Float64}}, result_Dout_path::Matrix{Matrix{Float64}}, result_Yout_path::Matrix{Matrix{Float64}}, result_YFout_path::Matrix{Matrix{Float64}}, # modified variables
    majorregions::DataFrame, laboralloc_path::Array{Float64, 3}, Y_path::Matrix{Float64}, Lsectorpath_guess::Array{Float64, 3}, params::StructParams, w_path_guess::Matrix{Float64}, 
    rP_path::Matrix{Float64}, pg_path_s::Array{Float64, 3}, p_F_path_guess::Matrix{Float64}, p_E_path_guess::Matrix{Float64},
    kappa::Float64, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, KF_path::Matrix{Float64}, capT::Int64, regionParams::StructRWParams, D_path::Matrix{Float64})

    kk = params.N
    for t=1:capT
        fill!(result_price_path[kk, t], 0)
        fill!(result_Dout_path[kk, t], 0)
        fill!(result_Yout_path[kk, t], 0)
        fill!(result_YFout_path[kk, t], 0)
    end

    @inbounds for t=1:capT
        ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
        n = majorregions.n[kk]

        # set up optimization problem for region kk
        secalloc = laboralloc_path[ind, :, t]
        #Yvec = Y_path[ind, t]
        Lshifter = Lsectorpath_guess[ind, :, t]
        Kshifter = Lsectorpath_guess[ind, :, t] .* (params.Vs[:,4]' .* ones(n, 1)) ./
                    (params.Vs[:,1]' .* ones(n, 1)) .*
                    (w_path_guess[ind, t] ./ rP_path[ind, t])
        #Ltotal = sum(Lshifter, dims=2)
        guess = [D_path[ind, t]; Y_path[ind, t]]

        # define shifters for objective function
        pg_s = pg_path_s[ind, :, t]
        p_F_in = p_F_path_guess[:, t]
        prices = (p_E_path_guess[ind, t] .* ones(1, params.I))
        power = (params.Vs[:,2]' .* ones(n, 1)) .+ (params.Vs[:, 3]' .* ones(n, 1))
        shifter = pg_s .* 
                (kappa .+ (prices ./ (kappa .* p_F_in))) .^ (params.psi / (params.psi - 1) .* (params.Vs[:,3]' .* ones(n, 1))) .*
                (1 .+ (params.Vs[:, 3]' .* ones(n, 1)) ./ (params.Vs[:, 2]' .* ones(n, 1))) .^ (-(params.Vs[:,2]' .* ones(n,1)) .- (params.Vs[:, 2]' .* ones(n, 1))) .*
                params.Z[ind] .*
                params.zsector[ind, :] .*
                Lshifter .^ (params.Vs[:,1]' .* ones(n, 1)) .*
                Kshifter .^ (params.Vs[:, 4]' .* ones(n, 1))
        shifter = shifter .* secalloc .^ power
        KRshifter = regionParams.thetaS[ind] .* KR_path_S[ind, t] .+ regionParams.thetaW[ind] .* KR_path_W[ind, t]
        KFshifter = KF_path[ind, t]

        # define bounds
        YFmax = KF_path[ind, t]

        Threads.@threads :static for jj = 1:n
            # solve market equilibrium
            local con = [1 -1]
            local guess = [1; KRshifter[jj]]
            local LB = [0; KRshifter[jj]]
            local UB = [10^6; YFmax[jj] + KRshifter[jj]]
            local l_guess = length(guess)

            local x = Vector{Float64}(undef, 2)
            model = Model(Ipopt.Optimizer) # 266 calls on profiler
            set_silent(model)
            @variable(model, LB[i] <= x[i=1:l_guess] <= UB[i], start=guess[i]) # 177 calls on profiler
            @constraint(model, c1, con * x <= 0) 
            @objective(model, Min, obj2(x, power[jj], shifter[jj], KFshifter[jj], KRshifter[jj], p_F_in[1], params))
            optimize!(model) # 21705 calls on profiler


            local P_out = value.(x)
            #local P_out = solve_model2(jj, x, LB, UB, guess, params, power, shifter, KFshifter, KRshifter, p_F_in)
            result_Dout_path[kk, t][jj] = P_out[1]
            result_Yout_path[kk, t][jj] = P_out[2]
            result_YFout_path[kk, t][jj] = P_out[2] - KRshifter[jj]
            result_price_path[kk, t][jj] = Price_Solve(P_out, shifter[jj], 1, params)[1]
            result_price_path[kk, t] = clamp.(result_price_path[kk, t], 0.001, 1)
            jj

        end
    end
end

function up_paths!(t::Int64, kk::Int64, majorregions::DataFrame, p_E_path_guess::Matrix{Float64}, D_path::Matrix{Float64},
    Y_path::Matrix{Float64}, YF_path::Matrix{Float64}, PI_path::Matrix{Float64}, # variables to be updated
    result_price_path::Matrix{Matrix{Float64}}, result_Dout_path::Matrix{Matrix{Float64}}, result_Yout_path::Matrix{Matrix{Float64}},
    result_YFout_path::Matrix{Matrix{Float64}}, L::Matrix{Float64})

    ind = majorregions.rowid2[kk]:majorregions.rowid[kk] # 81.725 ns (2 allocations: 48 bytes)
    @views p_E_path_guess[ind, t] .= result_price_path[kk, t]
    @views D_path[ind, t] .= result_Dout_path[kk, t]
    @views Y_path[ind, t] .= result_Yout_path[kk, t]
    @views YF_path[ind, t] .= result_YFout_path[kk, t]
    @views PI_path[ind, t] .= sum(p_E_path_guess[ind, t] .* (D_path[ind, t] .- Y_path[ind, t])) .* L[ind, 1] ./ sum(L[ind, 1])
end

function fill_paths!(p_E_path_guess::Matrix, D_path::Matrix, Y_path::Matrix, YF_path::Matrix, PI_path::Matrix, 
                    params::StructParams, majorregions::DataFrame, result_price_path::Matrix, result_Dout_path::Matrix, 
                    result_Yout_path::Matrix, result_YFout_path::Matrix, capT::Int)

    
    @inbounds for t in 1:capT
        for kk = 1:(params.N - 1)
            up_paths!(t, kk, majorregions, p_E_path_guess, D_path, Y_path, YF_path, PI_path, # variables to be updated
                result_price_path, result_Dout_path, result_Yout_path, result_YFout_path, params.L) # 4.150 μs (31 allocations: 7.45 KiB)
        end   

        kk = params.N
        ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
        p_E_path_guess[ind, t] .= vec(result_price_path[kk, t])
        D_path[ind, t] .= vec(result_Dout_path[kk, t])
        Y_path[ind, t] .= vec(result_Yout_path[kk, t])
        YF_path[ind, t] .= vec(result_YFout_path[kk, t])
        PI_path[ind, t] .= sum(p_E_path_guess[ind, t] .* (D_path[ind, t] .- Y_path[ind, t])) .* params.L[ind, 1] ./ sum(params.L[ind, 1])
        
    end
end

function smooth_prices!(p_E_path_guess::Matrix{Float64}, D_path::Matrix{Float64}, Y_path::Matrix{Float64}, YF_path::Matrix{Float64},
    sseq::StructPowerOutput, expo3::Vector, expo4::Vector, capT::Int, T::Int)
    
    @inbounds for kk in capT+1:T
        jj = kk - capT
        p_E_path_guess[:, kk] .= sseq.p_E_LR .+ ((p_E_path_guess[:, kk-1] .- sseq.p_E_LR) .* expo3[jj])
    end

    @inbounds for kk in capT+1:T+1
        jj = kk - capT
        D_path[:, kk] .= sseq.D_LR .+ ((D_path[:, kk-1] .- sseq.D_LR) .* expo4[jj])
        Y_path[:, kk] .= sseq.YE_LR .+ ((Y_path[:, kk-1] .- sseq.YE_LR) .* expo4[jj])
        YF_path[:, kk] .= sseq.YF_LR .+ ((YF_path[:, kk-1] .- sseq.YF_LR) .* expo4[jj]) 
    end
end

function update_labour_mrkt!(w_path_guess::Matrix{Float64}, Lsectorpath_guess::Array{Float64, 3}, 
        Lsector::Matrix{Float64}, params::StructParams, rP_path::Matrix{Float64}, p_E_path_guess::Matrix{Float64}, 
        p_F_path_guess::Matrix{Float64}, D_path::Matrix{Float64}, Y_path::Matrix{Float64}, KP_path_guess::Matrix{Float64}, 
        PI_path::Matrix{Float64}, w_update::Matrix{Float64}, w_real_path::Matrix{Float64}, PC::Matrix{Float64})

    @inbounds for i in 1:capT
        # get capital vec
        Ksector = Lsector .* (params.Vs[:, 4]' .* ones(params.J, 1)) ./
                (params.Vs[:, 1]' .* ones(params.J, 1)) .* (w_path_guess[:, i] ./ rP_path[:, i]) #  77.000 μs (32 allocations: 278.11 KiB)
        KP_path_guess[:, i] = sum(Ksector, dims=2) # 6.620 μs (7 allocations: 19.95 KiB)
        w_update[:, i], w_real_path[:, i], Incomefactor, PC[:, i] = wage_update_ms(w_path_guess[:, i], p_E_path_guess[:, i], p_E_path_guess[:, i], 
                                                                        p_F_path_guess[:, i], D_path[:, i], Y_path[:, i],
                                                                        rP_path[:, i], KP_path_guess[:, i], PI_path[:, i], 1, params) #   5.467817 seconds (341 allocations: 1.482 GiB, 1.87% gc time)
        #relexp = Xjdashs ./ (Xjdashs[:, 1] .* ones(1, params.I))
        w_path_guess[:, i] .= 0.5 * w_update[:, i] + (1- 0.5) * w_path_guess[:, i]   #8.600 μs (15 allocations: 59.86 KiB)
        Lsectorpath_guess[:, :, i] .= Lsectorpath_guess[:, :, i] ./ sum(Lsectorpath_guess[:,:,i], dims = 2) .* params.L #  27.000 μs (19 allocations: 20.64 KiB)

    end
end

# ---------------------------------------------------------------------------- #
#                        Update Fossil Market Functions                        #
# ---------------------------------------------------------------------------- #

function update_fossil_market!(fossilsales_path::Matrix, p_F_path_guess::Matrix{Float64},
                                laboralloc_path::Array, D_path::Matrix, params::StructParams, p_E_path_guess::Matrix, YF_path::Matrix, 
                                KF_path::Matrix, p_F_int, regions::DataFrame, T::Int, interp1, g::Float64, r_path::Adjoint{Float64, Vector{Float64}}, 
                                fusage_total_path::Matrix, p_F_update::Matrix)
    # compute electricity and fossil fuel usage in industry and electricity
    e2_path = Matrix{Float64}(undef, 2531, 501)
    fusage_ind_path = Matrix{Float64}(undef, 2531, 501)
    fusage_power_path = Matrix{Float64}(undef, 2531, 501)

    @inbounds for t = 1:T+1
        e2_path[:, t] .= sum(laboralloc_path[:,:,t] .* (D_path[:, t] .* ones(1, params.I)) .* 
            ((params.Vs[:,2]' .* ones(params.J, 1)) ./ ((params.Vs[:,2]' .* ones(params.J, 1)) .+ (params.Vs[:, 3]' .* ones(params.J, 1)))), dims=2)
        fusage_ind_path[:, t] .= (params.kappa) .^ (-1) .* e2_path[:, t] .* (p_E_path_guess[:,t] ./ p_F_path_guess[:, t]) .^ params.psi
        # last element of e2_path[:, 1] is really high for some reason 
        fusage_power_path[:, t] .= (YF_path[:,t] ./ KF_path[:, t] .^ params.alpha2) .^ (1/params.alpha1)
        fusage_total_path[:,t] .= (sum(fusage_power_path[:, t]) .+ sum(fusage_ind_path[:, t]))
    end

    # Get S_t from sequence
    S_t = zeros(1, T+1)
    S_t[1] = fusage_total_path[1]
    @inbounds for i=2:T+1
        S_t[i] = fusage_total_path[i] + S_t[i-1]
    end

    # the initial fossil fuel price is a free variable, can look up in table

    p_F_update[1] = p_F_int
    @inbounds for t=1:T
        p_F_update[t+1] = (p_F_update[t] - (interp1(S_t[t]) / ((1 + g)^t)) * (1 - (1/((r_path[t])*(1+g))))) * r_path[t]
    end
    # sum(fusage_ind_path[:, 1]) = 42042 > 761.6655 (!!!)
    # sum(fusage_power_path[:, 1]) = 905.7099 > 627.5190
    # S_t[1] = 42947.95 > 2832.1 (!!!!)
    # p_F_update[1] = exactly the same 
    # interp1(S_t[1]) = 0.048378 > 0.0322
    # interp1(S_t[t]) / ((1 + g)^t) = 0.047899 > 0.0019


    # get fossil sales path for income
    fossilsales_path .= fusage_total_path .* p_F_update .* regions.reserves

    # compute max difference and update fossilfuel price
    #diffpF = maximum(abs.(p_F_update[1:100] .- p_F_path_guess[1:100]) ./ p_F_path_guess[1:100])
    p_F_path_guess .= 0.1 .* p_F_update .+ (1 - 0.1) .* p_F_path_guess

end

function up_decaymat!(decaymat::Array{Float64, 3}, expo5::Vector{Float64}, J::Int64, I::Int64, T::Int64, capT::Int64)
    one_vec = ones(J, I)
    @inbounds for kk in 1:(T-capT)
        decaymat[:, :, kk] .= (expo5[kk] .* one_vec)
    end
end

function up_wpathguess!(w_path_guess::Matrix{Float64}, w_LR::Matrix{Float64}, expo3::Vector{Float64}, T::Int64, capT::Int64)
    @inbounds for i = 1:(T - capT)
        kk = capT + i
        jj = i
        w_path_guess[:, kk] .= w_LR .+ (w_path_guess[:, capT] .- w_LR) .* expo3[jj] 
    end # 0.004911 seconds (6.24 k allocations: 9.478 MiB)
end

function gen_curtailment!(renewshare_path::Matrix{Float64}, SShare_region_path::Matrix{Float64}, 
                curtailmentfactor_path_S::Matrix{Float64}, curtailmentfactor_path_W::Matrix{Float64}, curtailmentfactor_path::Matrix{Float64}, # modified variables
                N::Int64, majorregions::DataFrame, YF_path::Matrix{Float64}, Y_path::Matrix{Float64}, SShare_path::Matrix{Float64}, interp3, hoursofstorage_path::Matrix{Float64})
            
    @inbounds for kk = 1:(N-1)
        ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
        n = majorregions.n[kk]

        renewshare_path[ind, :] .= (1 .- (sum(YF_path[ind, :], dims=1) ./ sum(Y_path[ind, :], dims=1)) .* ones(n, 1))
        SS = sum(SShare_path[ind, :] .* Y_path[ind, :] ./ sum(Y_path[ind, :], dims = 1), dims=1)
        SShare_region_path[ind, :] .= (SS .* ones(n, 1))
        curtailmentfactor_path_S[ind, :] .= renewshare_path[ind, :] .* SShare_region_path[ind, :]
        curtailmentfactor_path_W[ind, :] .= renewshare_path[ind, :] .* (1 .- SShare_region_path[ind, :])
        curtailmentfactor_path[ind, :] .= interp3.(curtailmentfactor_path_W[ind, :], curtailmentfactor_path_S[ind,:], 
                                                hoursofstorage_path[ind, :])    # interp3 is defined in SteadyState.jl

    end #  0.121566 seconds (1.33 k allocations: 109.539 MiB)

end

function up_kfac!(kfac::Matrix{Float64}, expo6::Vector{Float64}, p_E_path_guess::Matrix{Float64}, pE_S_FE_path::Matrix{Float64})
    tmp = clamp.((0.2 .*(p_E_path_guess .- pE_S_FE_path) ./ pE_S_FE_path), -0.1, 0.1)
    @inbounds for i in 1:501
        kfac[:, i] .= 1 .+ expo6[i] .* tmp[:, i]
    end #  1.412821 seconds (10.02 k allocations: 4.743 GiB, 13.03% gc time)
end

function up_KRpathmin!(KR_path_min::Matrix{Float64}, KR_path::Matrix{Float64}, deltaR::Float64, T::Int64)
    @inbounds for i = 2:T
        KR_path_min[:, i] .= KR_path[:, i - 1] .* (1 - deltaR)
    end  # 0.004870 seconds (4.99 k allocations: 9.784 MiB)
end

function up_KRpath_smooth!(KR_path::Matrix{Float64}, capT::Int64, T::Int64, KR_LR::Vector{Float64}, expo3::Vector{Float64})
    @inbounds for i = capT+1:T
        jj = i - capT
        KR_path[:, i] .= KR_LR .+ (KR_path[:, capT] .- KR_LR) .* expo3[jj]
    end #   0.007283 seconds (6.24 k allocations: 9.478 MiB)    
end

function calc_difftrans(kfac::Matrix{Float64}, capT::Int64, KR_path::Matrix{Float64})
    difftrans = maximum(maximum(((kfac[:, 2:capT] .- 1) .* KR_path[:, 2:capT]), dims=1))     #get at substantial adjustments to K
    # 433.600 μs (23 allocations: 1.10 MiB)
    return difftrans
end

mutable struct StructTransEq
    difftrans::Float64
    ll::Int
    r_path::Adjoint{Float64, Vector{Float64}}
    KR_path::Matrix{Float64}
    p_KR_bar_path::Matrix{Float64}
    KF_path::Matrix{Float64}
    PC_path_guess::Matrix{Float64}
    fossilsales_path::Matrix{Float64}
    w_path_guess::Matrix{Float64}
    YF_path::Matrix{Float64}
    YR_path::Matrix{Float64}
    Y_path::Matrix{Float64}
    p_KR_path_W::Matrix{Float64}
    p_KR_path_S::Matrix{Float64}
    p_KR_path_guess_W::Matrix{Float64}
    p_F_path_guess::Matrix{Float64}
    KP_path_guess::Matrix{Float64}
    p_E_path_guess::Matrix{Float64}
    fusage_total_path::Matrix{Float64}
    p_B_path_guess::Matrix{Float64}
end

function solve_transition_eq(R_LR::Float64, GsupplyCurves::StructGsupply, decayp::Float64, T::Int64, params::StructParams, sseq::StructPowerOutput, KR_init_S::Matrix, 
    KR_init_W::Matrix, mrkteq::StructMarketEq, Initialprod::Int, RWParams::StructRWParams, curtailmentswitch::Int64, 
    p_KR_bar_init::Matrix, laboralloc_init::Matrix, regionParams::StructRWParams, majorregions::DataFrame, Linecounts::DataFrame, linconscount::Int, 
    kappa::Float64, regions::DataFrame, Transiter::Int, st::Matrix, hoursofstorage::Int, pB_shifter::Float64, g::Float64, 
    wage_init::Vector, p_KR_init_S::Float64, p_KR_init_W::Float64, p_F_int::Float64, interp3::Any)

    # ---------------------------------------------------------------------------- #
    #                                Initialization                                #
    # ---------------------------------------------------------------------------- #

    capT=20
    updwk = 1

    # guess for interest rate
    r_path = [R_LR for _ in 1:T]'

    KR_path_S = Matrix{Float64}(undef, 2531, 501)
    KR_path_W = Matrix{Float64}(undef, 2531, 501)
    p_E_path_guess = Matrix{Float64}(undef, 2531, 501)
    D_path = Matrix{Float64}(undef, 2531, 501)
    Y_path = Matrix{Float64}(undef, 2531, 501)
    PC_path_guess = Matrix{Float64}(undef, 2531, 501)
    PI_path = Matrix{Float64}(undef, 2531, 501)
    w_path_guess = Matrix{Float64}(undef, 2531, 501)
    B_path = Matrix{Float64}(undef, 2531, 501)
    hoursofstorage_path = Matrix{Float64}(undef, 2531, 501)
    renewshare_path = Matrix{Float64}(undef, 2531, 501)
    KF_path = Matrix{Float64}(undef, 2531, 501)
    p_KR_bar_path = Matrix{Float64}(undef, 2531, 501)
    p_KR_path_guess_S = Matrix{Float64}(undef, 1, 501)
    p_KR_path_guess_W = Matrix{Float64}(undef, 1, 501)
    p_F_path_guess = Matrix{Float64}(undef, 1, 501)

    p_B_path = Matrix{Float64}(undef, 2531, 501)
    p_B_path_guess = Matrix{Float64}(undef, 1, 501)

    decaymat = zeros(2531, 10, 480)

    # initialize variables within while loop
    p_KR_path_S = Matrix{Float64}(undef, 1, 501)
    p_KR_path_W = Matrix{Float64}(undef, 1, 501)
    SShare_path = Matrix{Float64}(undef, 2531, 501)
    thetabar_path = Matrix{Float64}(undef, 2531, 501)
    KR_path = Matrix{Float64}(undef, 2531, 501)
    Qtotal_path_B = Matrix{Float64}(undef, 1, 501)
    rP_path = Matrix{Float64}(undef, 2531, 501)
    pg_path_s = zeros(2531, 10, 501)
    YR_path = Matrix{Float64}(undef, 2531, 501)
    #P_E_path = Matrix{Float64}(undef, 2531, 501)

    sizes = [727, 755, 30, 53, 13, 15, 46, 11, 26, 78, 125, 320, 332]
    result_price_path = Array{Matrix{Float64},2}(undef, 13, capT)
    result_Dout_path = Array{Matrix{Float64},2}(undef, 13, capT)
    result_Yout_path = Array{Matrix{Float64},2}(undef, 13, capT)
    result_YFout_path = Array{Matrix{Float64},2}(undef, 13, capT)

    for i in 1:13
        for j in 1:capT
            if i == 13
                # Row vectors for row 13
                result_price_path[i, j] = Matrix{Float64}(undef, 1, sizes[i])
                result_Dout_path[i, j] = Matrix{Float64}(undef, 1, sizes[i])
                result_Yout_path[i, j] = Matrix{Float64}(undef, 1, sizes[i])
                result_YFout_path[i, j] = Matrix{Float64}(undef, 1, sizes[i])
            else
                # Column vectors for other rows
                result_price_path[i, j] = Matrix{Float64}(undef, sizes[i], 1)
                result_Dout_path[i, j] = Matrix{Float64}(undef, sizes[i], 1)
                result_Yout_path[i, j] = Matrix{Float64}(undef, sizes[i], 1)
                result_YFout_path[i, j] = Matrix{Float64}(undef, sizes[i], 1)
            end
        end
    end

    YF_path = Matrix{Float64}(undef, 2531, 501)
    KP_path_guess = Matrix{Float64}(undef, 2531, 501)
    w_update = Matrix{Float64}(undef, 2531, 20)
    w_real_path = Matrix{Float64}(undef, 2531, 20)
    PC = Matrix{Float64}(undef, 2531, 20)
    fossilsales_path = Matrix{Float64}(undef, 2531, 501)
    SShare_region_path = Matrix{Float64}(undef, 2531, 501)
    curtailmentfactor_path_S = Matrix{Float64}(undef, 2531, 501)
    curtailmentfactor_path_W = Matrix{Float64}(undef, 2531, 501)
    curtailmentfactor_path = Matrix{Float64}(undef, 2531, 501)
    pE_S_FE_path = Matrix{Float64}(undef, 2531, 501)
    Capinvest = falses(2531, 501)
    Capdisinvest = falses(2531, 501)
    kfac = Matrix{Float64}(undef, 2531, 501)
    KR_path_min = Matrix{Float64}(undef, 2531, 500)
    KR_path_update = Matrix{Float64}(undef, 2531, 500)
    Lsectorpath_guess = zeros(2531, 10, 501)
    laboralloc_path = zeros(2531, 10, 501)
    fusage_total_path = Matrix{Float64}(undef, 1, 501)

    p_F_update = Matrix{Float64}(undef, 1, 501)


    interp1 = linear_interpolation(vec(GsupplyCurves.Q), vec(GsupplyCurves.P))

    expo = Matrix{Float64}(undef, 501, 1)
    expo .= exp.(0.1 * decayp * (0:T))

    expo1 = Matrix{Float64}(undef,501, 1)
    expo1 .= exp.(decayp * (0:T))
    round.(expo1, digits=4)
    expo1

    expo2 = Matrix{Float64}(undef, 501, 1)
    expo2 .= exp.(-params.delta * (0:T))

    expo3 = Vector{Float64}(undef, 480)
    expo3 .= exp.(decayp .* (1:T-capT))

    expo4 = Vector{Float64}(undef, 481)
    expo4 .= exp.(decayp .* (1:T-capT+1))

    expo5 = Vector{Float64}(undef, 480)
    expo5 .= exp.(decayp .* (capT+1:T)) 

    expo6 = Vector{Float64}(undef, 501)
    expo6 .= exp.(-0.01 .* (0:T))

    # ---------------------------------------------------------------------------- #
    #                              Set Initial Guesses                             #
    # ---------------------------------------------------------------------------- #
    
    #Intial guess for long run capital, slowly initially 
    guess_path_S!(KR_path_S, sseq, KR_init_S, expo)
    guess_path_W!(KR_path_W, sseq, KR_init_W, expo)

    # initial guess for prices
    guess_path_p_E!(p_E_path_guess, sseq, mrkteq, expo1)

    # initial guess for demand and supply
    guess_path_D!(D_path, sseq, mrkteq, expo1)
    guess_path_Y!(Y_path, sseq, mrkteq, expo1)

    # initial guess for goods prices
    guess_path_PC!(PC_path_guess, sseq, mrkteq, expo1)

    # initial guess for electricity profits
    guess_path_PI!(PI_path, sseq, mrkteq, expo1)

    # initial guess for wages
    guess_path_w!(w_path_guess, sseq, wage_init, expo1)

    # initial guess for battery path
    guess_path_B!(B_path, sseq, RWParams, expo)

    # initial guess for hours of storage path
    guess_path_hours!(hoursofstorage_path, curtailmentswitch, expo)
    guess_path_renewshare!(renewshare_path, expo)

    # capital path
    guess_path_KF!(KF_path, sseq, RWParams, expo2)
    guess_path_p_KR_bar!(p_KR_bar_path, sseq, p_KR_bar_init, expo1)
    guess_path_p_KR_S!(p_KR_path_guess_S, sseq, p_KR_init_S, expo)
    guess_path_p_KR_W!(p_KR_path_guess_W, sseq, p_KR_init_W, expo)


    # initial guess for sectoral allocations
    sectoral_allocations!(Lsectorpath_guess, laboralloc_path, laboralloc_init, sseq, T, expo1)

    # Initial guess for fossil path    
    guess_path_fossil!(p_F_path_guess, T)

    # set path for battery prices     
    guess_path_p_B!(p_B_path_guess, sseq, hoursofstorage_path, params.gammaB, Initialprod, RWParams, expo1)

    # ---------------------------------------------------------------------------- #
    #                             SOLVE TRANSITION PATH                            #
    # ---------------------------------------------------------------------------- #

    # --------------------- Initialize Intermediate Variables -------------------- #

    difftrans=1
    ll=1

    while difftrans>10^(-2) && ll<=Transiter
        update_capital_price!(p_KR_path_S, p_KR_path_W, p_KR_path_guess_S, p_KR_path_guess_W,
                        KR_path_S, KR_path_W, params, Initialprod, KR_init_S, KR_init_W)# 7.456 ms (15516 allocations: 25.83 MiB)
        # updates p_KR_path_S, p_KR_path_W, p_KR_path_guess_S, p_KR_path_guess_W

        # get solar shares

        up_SShare!(SShare_path, regionParams.thetaS, p_KR_path_guess_S, params.varrho, regionParams.thetaW, p_KR_path_guess_W) # 47.407 ms (1 allocation: 16 bytes)
        # updates SShare_path
        up_thbar!(thetabar_path, regionParams.thetaS, SShare_path, regionParams.thetaW) # 1.003 ms (0 allocations: 0 bytes)
        # update thetabar_path
        up_KR_bar!(p_KR_bar_path, SShare_path, p_KR_path_guess_S, p_KR_path_guess_W) #  1.115 ms (0 allocations: 0 bytes)
        # update p_KR_bar_path

        update_battery_prices!(KR_path, Qtotal_path_B, KR_path_S, KR_path_W, params, hoursofstorage_path, RWParams, Initialprod, hoursofstorage) # 6.117 ms (5006 allocations: 12.88 MiB)
        # updates KR_path and Qtotal_path_B

        # set returns on capital
        up_rP_path!(rP_path, r_path, params.deltaP, PC_path_guess) # 679.000 μs (1 allocation: 16 bytes)

        up_pg_path!(pg_path_s, w_path_guess, params.Vs, params.J, p_E_path_guess, params.kappa, p_F_path_guess, params.psi, params.Z, params.zsector, params.cdc) #  679.115 ms (20566 allocations: 21.93 MiB)

        # power path
        up_YR_path!(YR_path, regionParams.thetaS, KR_path_S, regionParams.thetaW, KR_path_W) # 1.273 ms (0 allocations: 0 bytes)
        #P_E_path .= Y_path .- D_path

        # ---------------------------------------------------------------------------- #
        #                      SOLVE TRANSITION ELECTRICITY MARKET                     #
        # ---------------------------------------------------------------------------- #

        transition_electricity_US_Europe!(result_price_path, result_Dout_path, result_Yout_path, result_YFout_path, # modified variables
            majorregions, Linecounts, RWParams, laboralloc_path, Lsectorpath_guess, params, w_path_guess, rP_path, 
            linconscount, pg_path_s, p_F_path_guess, p_E_path_guess, kappa, KR_path_S, KR_path_W, KF_path, capT, regionParams)

            # NOT correctly updating result_price_path

        # ------------------------------ Other Countries ----------------------------- #

        transition_electricity_other_countries!(result_price_path, result_Dout_path, result_Yout_path, result_YFout_path, # modified variables
            majorregions, Linecounts, RWParams, laboralloc_path, Lsectorpath_guess, params, w_path_guess, rP_path,
            linconscount, pg_path_s, p_F_path_guess, p_E_path_guess, kappa, KR_path_S, KR_path_W, KF_path, capT, regionParams)
        
        # transition_electricity_other_countries!()   
        # updates result_price_path, result_Dout_path, result_Yout_path, result_YFout_path

        # --------------------------------- Off Grid --------------------------------- #
        transition_electricity_off_grid!(result_price_path, result_Dout_path, result_Yout_path, result_YFout_path, # modified variables
            majorregions, laboralloc_path, Y_path, Lsectorpath_guess, params, w_path_guess, rP_path,
            pg_path_s, p_F_path_guess, p_E_path_guess, kappa, KR_path_S, KR_path_W, KF_path, capT, regionParams, D_path) #  31.026000 seconds (31.94 M allocations: 1.177 GiB)

        fill_paths!(p_E_path_guess, D_path, Y_path, YF_path, PI_path, params, majorregions, result_price_path, 
            result_Dout_path, result_Yout_path, result_YFout_path, capT) #    746.000 μs (7500 allocations: 1.08 MiB)
            # theoretically this should have no allocations
        # updates p_E_path_guess, D_path, Y_path, YF_path, PI_path

        # out from cap T set prices based on smoothing
        smooth_prices!(p_E_path_guess, D_path, Y_path, YF_path, sseq, expo3, expo4, capT, T) #  15.168 ms (3846 allocations: 37.29 MiB)

        # updates p_E_path_guess, D_path, Y_path, YF_path

        # ---------------------------------------------------------------------------- #
        #                        UPDATE TRANSITION LABOUR MARKET                       #
        # ---------------------------------------------------------------------------- #
        
        update_labour_mrkt!(w_path_guess, Lsectorpath_guess, sseq.Lsector, params, rP_path, 
            p_E_path_guess, p_F_path_guess, D_path, Y_path, KP_path_guess, PI_path, w_update, 
            w_real_path, PC) # 58.696 s (7961 allocations: 29.64 GiB)

        # ---------------------------------------------------------------------------- #
        #                             UPDATE FOSSIL MARKET                             #
        # ---------------------------------------------------------------------------- #
        #diffpF = 1.0

        update_fossil_market!(fossilsales_path, p_F_path_guess,
                                laboralloc_path, D_path, params, p_E_path_guess, YF_path, KF_path,
                                p_F_int, regions, T, interp1, g, r_path, fusage_total_path, p_F_update) # 150.126 ms (17546 allocations: 329.80 MiB)
        # updates p_F_path_guess, fossilsales_path, diffpF

        # ------------------------ UPDATE RENEWABLE INVESTMENT ----------------------- #

        # out from cap T set prices based on smoothing
        # initial guess for sectoral allocations
        up_decaymat!(decaymat, expo5, params.J, params.I, T, capT) # 7.141 ms (3 allocations: 197.81 KiB)

        
        Lsectorpath_guess[:, :, (capT + 1):T] .= sseq.laboralloc_LR .+ (Lsectorpath_guess[:, :, capT] .- sseq.laboralloc_LR) .* decaymat; # 11.621 ms (14 allocations: 198.22 KiB)

        up_wpathguess!(w_path_guess, sseq.w_LR, expo3, T, capT) #  2.307 ms (960 allocations: 9.31 MiB)

        # lookup battery requirements
        if curtailmentswitch==1
            hoursofstorage_path = interp1(renewshare_path)
        else
            hoursofstorage_path = (hoursofstorage .* ones(params.J, T+1))
        end

        # set battery prices
        p_B_path_guess .= (Initialprod .+ Qtotal_path_B) .^ (-params.gammaB)   # shift avoids NaN when 0 investment
        p_B_path .= pB_shifter .* hoursofstorage_path .* p_B_path_guess

        # generate curtailment factor for grid regions
        gen_curtailment!(renewshare_path, SShare_region_path, curtailmentfactor_path_S, curtailmentfactor_path_W, curtailmentfactor_path, # modified variables
                params.N, majorregions, YF_path, Y_path, SShare_path, interp3, hoursofstorage_path) # 112.160 ms (1765 allocations: 109.55 MiB)

        # generate curtailment factor for off grid-regions
        kk = params.N
        ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
        n = majorregions.n[kk]

        renewshare_path[ind, :] .= 1 .- YF_path[ind, :] ./ Y_path[ind, :]
        SShare_region_path[ind,:] .= SShare_path[ind, :]
        curtailmentfactor_path_S[ind, :] .= renewshare_path[ind, :] .* SShare_region_path[ind, :]
        curtailmentfactor_path_W[ind, :] .= renewshare_path[ind, :] .* (1 .- SShare_region_path[ind, :])
        curtailmentfactor_path[ind, :] .= interp3.(curtailmentfactor_path_W[ind, :], curtailmentfactor_path_S[ind, :], hoursofstorage_path[ind, :])
        # 0.015977 seconds (87 allocations: 12.693 MiB)
        

        # if market price is above free entry price, increase renewable capital
        # decrease otherwise
        pE_S_FE_path[:, 2:end] .= (p_KR_bar_path[:,1:end-1] .+ p_B_path[:, 1:end-1] .- (p_KR_bar_path[:, 2:end] .+ p_B_path[:, 2:end]) .* (1-params.deltaR) ./ r_path) .*
                    regionParams.costshifter ./ thetabar_path[:, 1:end-1]
        pE_S_FE_path[:, 1] .= pE_S_FE_path[:, 2]
        pE_S_FE_path .= pE_S_FE_path .- st
        Capinvest .= p_E_path_guess .> pE_S_FE_path
        Capdisinvest .= p_E_path_guess .<= pE_S_FE_path

        
        up_kfac!(kfac, expo6, p_E_path_guess, pE_S_FE_path) # 4.945 ms (1004 allocations: 19.39 MiB)

        KR_path .= KR_path_S .+ KR_path_W
        sseq.KR_LR .= sseq.KR_LR_S .+ sseq.KR_LR_W

        up_KRpathmin!(KR_path_min, KR_path, params.deltaR, T) # 4.612 ms (999 allocations: 9.68 MiB)

        KR_path_min[:, 1] .= KR_path[:, 1]

        # update the capital path
        KR_path_update .= KR_path[:, 2:end] .* kfac[:, 2:end]
        KR_path_update .= max.(KR_path_update, KR_path_min)
        KR_path[:, 2:end] .= KR_path[:, 2:end] .* (1 - updwk) .+ KR_path_update .* updwk

        # out from cap T set capital based on smoothing
        up_KRpath_smooth!(KR_path, capT, T, sseq.KR_LR, expo3) # 2.780 ms (960 allocations: 9.31 MiB)

        # split out into solar / wind shares
        KR_path_S .= SShare_path .* KR_path
        KR_path_W .= (1 .- SShare_path) .* KR_path

        difftrans = calc_difftrans(kfac, capT, KR_path) #  522.700 μs (10 allocations: 1.10 MiB)

        ll += 1
        println("Transition Difference = ", difftrans)
        println("Loop number = ", ll)

    end

    return StructTransEq(
        difftrans, 
        ll, 
        r_path, 
        KR_path, 
        p_KR_bar_path, 
        KF_path, 
        PC_path_guess, 
        fossilsales_path, 
        w_path_guess, 
        YF_path, 
        YR_path, 
        Y_path, 
        p_KR_path_S, 
        p_KR_path_W,
        p_KR_path_guess_W, 
        p_F_path_guess, 
        KP_path_guess,
        p_E_path_guess,
        fusage_total_path,
        p_B_path_guess
        )

end

end