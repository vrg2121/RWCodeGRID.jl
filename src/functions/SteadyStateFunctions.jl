module SteadyStateFunctions

import DataFrames: DataFrame
import MAT: matopen
import Statistics: mean
import ..DataLoadsFunc: StructGsupply, StructRWParams
import ..MarketFunctions: StructMarketEq
using JuMP, Ipopt
using ..RegionModel, ..MarketEquilibrium

import DrawGammas: StructParams
import ..ModelConfiguration: ModelConfig

export solve_power_output, StructPowerOutput, fill_wr!, up_e2_LR!


# ---------------------------------------------------------------------------- #
#                                   Load Data                                  #
# ---------------------------------------------------------------------------- #

function ss_load_mat(G::String)

    alr = matopen("$G/alloc_LR_guess.mat")
    laboralloc_LR = read(alr, "laboralloc_LR")
    close(alr)

    kls = matopen("$G/KR_LR_S_guess.mat")
    KR_LR_S = read(kls, "KR_LR_S")
    close(kls)

    klw = matopen("$G/KR_LR_W_guess.mat")
    KR_LR_W = read(klw, "KR_LR_W")
    close(klw)

    pel = matopen("$G/p_E_LR_guess.mat")
    p_E_LR = read(pel, "p_E_LR")
    close(pel)

    wl = matopen("$G/w_LR_guess.mat")
    w_LR = read(wl, "w_LR")
    close(wl)

    dol = matopen("$G/Dout_guess_LR.mat")
    result_Dout_LR = read(dol, "result_Dout_LR")
    close(dol)

    yol = matopen("$G/Yout_guess_LR.mat")
    result_Yout_LR = read(yol, "result_Yout_LR")
    close(yol)

    pcl = matopen("$G/PC_guess_LR.mat")
    PC_guess_LR = read(pcl, "PC_guess_LR")
    close(pcl)
    return laboralloc_LR, KR_LR_S, KR_LR_W, p_E_LR, w_LR, result_Dout_LR, result_Yout_LR, PC_guess_LR
end

# ---------------------------------------------------------------------------- #
#                           Set Long Run Good Prices                           #
# ---------------------------------------------------------------------------- #

function ss_optimize_region!(result_price_LR::Vector, result_Dout_LR::Matrix, result_Yout_LR::Matrix, result_YFout_LR::Vector, Lossfac_LR::Matrix, # modified variables
        pg_LR_s::Matrix, majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc_LR::Matrix, Lsector::Matrix, 
        params::StructParams, w_LR::Matrix, rP_LR::Matrix, p_E_LR::Matrix, kappa::Float64, regionParams::StructRWParams, KF_LR::Matrix, p_F_LR::Float64,
        linconscount::Int, KR_LR_S::Matrix, KR_LR_W::Matrix, result_Yout_init::Matrix)

    Threads.@threads for kk in 1:(params.N - 1)

        local l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid = data_set_up(kk, majorregions, Linecounts, RWParams, laboralloc_LR, Lsector, params, w_LR, 
                                                                                        rP_LR, pg_LR_s, p_E_LR, kappa, regionParams, KF_LR, p_F_LR, linconscount, KR_LR_S, KR_LR_W, "steadystate")
        local P_out = solve_model(kk, l_guess, LB, UB, guess, regionParams, params, power, shifter, KFshifter, KRshifter, p_F_LR, mid)
    
        result_price_LR[kk] .= Price_Solve(P_out, shifter, n, params)
        @views result_Dout_LR[kk] .= P_out[1:end÷2]
        @views result_Yout_LR[kk] .= P_out[end÷2+1:end]
        @views result_YFout_LR[kk] .= P_out[end÷2+1:end] .- KRshifter
        local Pvec = P_out[end÷2+1:end] .- P_out[1:end÷2]
        local Losses = Pvec[2:end]' * regionParams.B[kk] * Pvec[2:end] .* params.Rweight
        @views Lossfac_LR[1, kk] = Losses / sum(result_Yout_init[kk])
    end

end

function ss_second_loop(majorregions::DataFrame, Lsector::Matrix, laboralloc::Matrix, params::StructParams, w_LR::Matrix, rP_LR::Union{Matrix, Vector},
    result_Dout_LR::Matrix, result_Yout_LR::Matrix, pg_LR_s::Matrix, p_E_LR::Matrix, kappa::Float64,
    regionParams, KR_LR_S::Matrix, KR_LR_W::Matrix, KF_LR::Matrix{Float64}, kk::Int64, p_F_LR::Float64)

    
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    nr = majorregions.rowid[kk] - majorregions.rowid2[kk] + 1
    Jlength = majorregions.n[kk]

    Kshifter = Matrix{Float64}(undef, nr, size(Lsector, 2))

    # get prices for places off the grid
    # set up optimization problem for region kk
    @views secalloc = laboralloc[ind, :]
    @views Lshifter = Lsector[ind, :]
    @views Kshifter .= Lsector[ind, :] .* (params.Vs[:,4]'.* ones(majorregions.n[kk], 1)) ./
                    ((params.Vs[:,1]'.* ones(majorregions.n[kk], 1))) .* (w_LR[ind] ./ rP_LR[ind])
    @views Ltotal = sum(Lshifter, dims=2)

    @views guess = [result_Dout_LR[kk]; result_Yout_LR[kk]] # 728.455 ns (1 allocation: 5.31 KiB)

    # define shifters for objective function
    @views pg_s = pg_LR_s[ind, :]
    @views prices = (p_E_LR[ind, :].* ones(1, params.I))
    @views power = ((params.Vs[:,2]'.* ones(Jlength, 1)) + (params.Vs[:,3]'.* ones(Jlength, 1)))

    @views shifter = pg_s .* (kappa .+ (prices ./ (kappa .* p_F_LR)).^(params.psi - 1)).^(params.psi / (params.psi - 1) .* 
                    (params.Vs[:,3]'.* ones(majorregions.n[kk], 1))) .*
                    (1 .+ (params.Vs[:,3]'.* ones(majorregions.n[kk], 1)) ./ (params.Vs[:,2]'.* ones(majorregions.n[kk], 1))) .^
                    (-(params.Vs[:,2]'.* ones(majorregions.n[kk], 1)) - (params.Vs[:,2]'.* ones(majorregions.n[kk], 1))) .*
                    params.Z[majorregions.rowid2[kk]:majorregions.rowid[kk]] .*
                    params.zsector[majorregions.rowid2[kk]:majorregions.rowid[kk], :] .*
                    Lshifter.^(params.Vs[:,1]'.* ones(majorregions.n[kk], 1)) .*
                    Kshifter .^ (params.Vs[:,4]'.* ones(majorregions.n[kk], 1))

    shifter .= shifter .* secalloc .^ power
    @views KRshifter = regionParams.thetaS[ind] .* KR_LR_S[ind] .+ regionParams.thetaW[ind] .* KR_LR_W[ind]
    @views KFshifter = KF_LR[ind]
    # define bounds
    @views YFmax = KF_LR[ind]

    return KRshifter, YFmax, guess, power, KFshifter, shifter
end

function ss_optimize_offgrid!(result_Dout_LR::Matrix{Any}, result_Yout_LR::Matrix{Any}, result_price_LR::Vector{Vector}, 
                    result_YFout_LR::Vector{Vector}, # modified variables
                    majorregions::DataFrame, Lsector::Matrix{Float64}, laboralloc::Matrix{Float64}, params::StructParams, w_LR::Matrix{Float64}, 
                    rP_LR::Matrix{Float64}, pg_LR_s::Matrix{Float64}, p_E_LR::Matrix{Float64}, kappa::Float64, regionParams::StructRWParams,
                    KR_LR_S::Matrix{Float64}, KR_LR_W::Matrix{Float64}, KF_LR::Matrix{Float64}, p_F_LR::Float64)
            
    kk = params.N
    
    KRshifter, YFmax, guess, power, KFshifter, shifter = ss_second_loop(majorregions, Lsector, laboralloc, params, w_LR, rP_LR,
                                                        result_Dout_LR, result_Yout_LR, pg_LR_s, p_E_LR, kappa,
                                                        regionParams, KR_LR_S, KR_LR_W, KF_LR, kk, p_F_LR) 


    mm = majorregions.n[kk] 
    P_out = Matrix{Float64}(undef, 2, mm)
    @inbounds for jj in 1:mm
        guess = [1; KRshifter[jj]]
        LB = [0; KRshifter[jj]]
        UB = [10^6; YFmax[jj] + KRshifter[jj]]
        l_guess = length(guess)

        x = Vector{Float64}(undef, 2)
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, LB[i] <= x[i=1:l_guess] <= UB[i], start=guess[i])
        @constraint(model, c1, x[1] - x[2] == 0) 
        @objective(model, Min, obj2(x, power[jj], shifter[jj], KFshifter[jj], KRshifter[jj], p_F_LR, params))
        optimize!(model)
        value.(x)

        P_out[:, jj] .= value.(x)
        #price = Price_Solve(P_out[:, jj], shifter[jj], 1, params)[1]

        result_Dout_LR[kk][jj] = P_out[1, jj]        
        result_Yout_LR[kk][jj] = P_out[2, jj] 
        result_price_LR[kk][1][1, jj] = Price_Solve(P_out[:, jj], shifter[jj], 1, params)[1] # this is fine
    end

    @inbounds for jj in 1:mm
        result_YFout_LR[kk][1][1, jj] = P_out[1, jj] - KRshifter[jj]
    end 

end

function fill_LR!(p_E_LR::Matrix{Float64}, D_LR::Vector{Float64}, YE_LR::Vector{Float64}, YF_LR::Vector{Float64}, PI_LR::Vector{Float64}, result_price_LR::Vector{Vector},# modified variables
                majorregions::DataFrame, result_Dout_LR::Matrix{Any}, 
                result_Yout_LR::Matrix{Any}, result_YFout_LR::Vector{Vector}, L::Matrix{Float64}, N::Int64)

    for kk = 1:(N - 1)
        up_LR!(p_E_LR, D_LR, YE_LR, YF_LR, PI_LR, # modified variables
                    majorregions, result_price_LR, result_Dout_LR, result_Yout_LR, result_YFout_LR, L, kk)
    end

    kk = N
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]

    YF_LR[ind] .= vec(result_YFout_LR[kk][1])
    result_price_LR[kk][1] .= clamp.(result_price_LR[kk][1], 0.001, 1) # bound the prices
    p_E_LR[ind] .= vec(result_price_LR[kk][1])
    D_LR[ind] .= vec(result_Dout_LR[kk])
    YE_LR[ind] .= vec(result_Yout_LR[kk])
    PI_LR[ind] .= sum(result_price_LR[kk][1] .* (result_Dout_LR[kk]-result_Yout_LR[kk])) .*
                    L[ind, 1] ./ sum(L[ind, 1])
end

function up_LR!(p_E_LR::Matrix{Float64}, D_LR::Vector{Float64}, YE_LR::Vector{Float64}, YF_LR::Vector{Float64}, PI_LR::Vector{Float64}, # modified variables
                majorregions::DataFrame, result_price_LR::Vector{Vector}, result_Dout_LR::Matrix{Any}, 
                result_Yout_LR::Matrix{Any}, result_YFout_LR::Vector{Vector}, L::Matrix{Float64}, kk::Int64)

    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    p_E_LR[ind] .= result_price_LR[kk]
    D_LR[ind].= result_Dout_LR[kk]
    YE_LR[ind] .= result_Yout_LR[kk]
    YF_LR[ind] .= result_YFout_LR[kk]
    PI_LR[ind, 1] .= sum(result_price_LR[kk] .* (result_Dout_LR[kk].- result_Yout_LR[kk])) .*
                    L[ind, 1] ./ sum(L[ind, 1])
end

# ---------------------------------------------------------------------------- #
#                               Update Parameters                              #
# ---------------------------------------------------------------------------- #

function up_KP_LR!(KP_LR::Matrix{Float64}, Lsector::Matrix{Float64}, Vs::Matrix{Float64}, J::Int64, w_LR::Matrix{Float64}, rP_LR::Matrix{Float64})        
    Ksector = Lsector .* (Vs[:,4]'.* ones(J, 1)) ./ (Vs[:,1]'.* ones(J, 1)) .* (w_LR ./ rP_LR)
    KP_LR .= sum(Ksector, dims = 2)
end

function up_prices_wages!(w_LR::Matrix{Float64}, w_real::Vector{Float64}, PC_LR::Vector{Float64}, laboralloc_LR::Matrix{Float64}, Lsector::Matrix{Float64}, # modified variables
                        p_E_LR::Matrix{Float64}, p_F_LR::Float64, D_LR::Vector{Float64}, YE_LR::Vector{Float64}, rP_LR::Matrix{Float64}, 
                        KP_LR::Matrix{Float64}, PI_LR::Vector{Float64}, params::StructParams)
    
    w_update, W_Real, Incomefactor, PC, Xjdashs = wage_update_ms(w_LR, p_E_LR, p_E_LR, p_F_LR, D_LR, YE_LR, rP_LR, KP_LR, PI_LR, 0, params);

    w_LR .= 0.2 .* w_update .+ (1 - 0.2) .* w_LR
    w_real .= W_Real
    PC_LR .= PC
    
    # update sectoral allocations
    laboralloc_LR .= Lsector ./ params.L
    
    relexp = Xjdashs ./ (Xjdashs[:,1] .* ones(1, params.I))
    relab = laboralloc_LR ./ (laboralloc_LR[:,1].* ones(1, params.I))
    
    Lsector .= Lsector .* clamp.(1 .+ 0.2 .* (relexp .- relab) ./ relab, 0.8, 1.2)
    Lsector .= Lsector ./ sum(Lsector, dims=2) .* params.L
end

function fill_cs!(cumsum::Matrix{Float64}, Depreciation_LR::Float64, iota::Float64, T::Int64)
    @inbounds for i = 1:T
        cumsum[1, i] = Depreciation_LR * (iota ^ i)
    end
end

function up_cap_prices(KR_LR_S, KR_LR_W, deltaR, iota, T, Initialprod, gammaS, gammaW)
    Dep_LR = Matrix{Float64}(undef, 2531, 1)
    cumsum = Matrix{Float64}(undef, 1, 500)

    Dep_LR .= KR_LR_S .* deltaR #   945.000 ns (3 allocations: 80 bytes)
    Depreciation_LR_S = sum(Dep_LR)

    Dep_LR = KR_LR_W .* deltaR
    Depreciation_LR_W = sum(Dep_LR)
    
    fill_cs!(cumsum, Depreciation_LR_S, iota, T) #   5.700 μs (1 allocation: 16 bytes
    Qtotal_LR_S = sum(cumsum)

    fill_cs!(cumsum, Depreciation_LR_W, iota, T)
    Qtotal_LR_W = sum(cumsum)

    p_KR_LR_S = (Initialprod * (iota ^ T) + Qtotal_LR_S) ^ (-gammaS) #   150.424 ns (7 allocations: 112 bytes)
    p_KR_LR_W = (Initialprod * (iota ^ T) + Qtotal_LR_W) ^ (-gammaW)

    return Depreciation_LR_S, Depreciation_LR_W, p_KR_LR_S, p_KR_LR_W
end

function calc_solar_shares!(SShare_LR, thetabar_LR, p_KR_bar_LR, # modified variables
                    thetaS, p_KR_LR_S, varrho, thetaW, p_KR_LR_W)

    SShare_LR .= (thetaS ./ p_KR_LR_S) .^ varrho ./ ((thetaS ./ p_KR_LR_S) .^ varrho + (thetaW ./ p_KR_LR_W) .^ varrho) #   79.700 μs (25 allocations: 79.95 KiB)
    thetabar_LR .= thetaS .* SShare_LR + thetaW .* (1 .- SShare_LR)
    p_KR_bar_LR .= SShare_LR .* p_KR_LR_S + (1 .- SShare_LR) .* p_KR_LR_W
end

function ss_update_params!(p_KR_bar_LR::Matrix, p_addon, params::StructParams, R_LR, regionParams, thetabar_LR::Matrix, 
                            curtailmentswitch, curtailmentfactor::Vector, p_E_LR::Matrix, KR_LR::Vector, 
                            KR_LR_S::Matrix, KR_LR_W::Matrix, SShare_LR::Matrix, pE_S_FE::Vector,
                            config::ModelConfig)
    
    pE_S_FE .= (p_KR_bar_LR .+ p_addon .- (p_KR_bar_LR .+ p_addon) .* (1 .- params.deltaR) / R_LR) .* 
                regionParams.costshifter ./ thetabar_LR ./ (1 .- curtailmentswitch .* curtailmentfactor)

    if config.RunExog==0
        kfac = 1 .+ clamp.(0.05 * ((p_E_LR .- pE_S_FE) ./ pE_S_FE), -0.05, 0.05) 
    elseif config.RunExog !== 0
        kfac = 1 .+ clamp.(0.05 * ((p_E_LR .- pE_S_FE) ./ pE_S_FE), -0.02, 0.02)
    end
    
    KR_LR_dash = KR_LR .* kfac
    KR_LR .= KR_LR_dash
    KR_LR_S .= SShare_LR .* KR_LR
    KR_LR_W .= (1 .- SShare_LR) .* KR_LR

    # Calculate the differences
    diffp = maximum(kfac .- 1)
    diffK = mean(abs.((kfac .- 1) .* KR_LR_dash))
    return diffK, diffp
end

function set_battery(KR_LR::Vector, hoursofstorage::Int64, params::StructParams, Initialprod::Int64, T::Int64)
    B_LR = KR_LR .* hoursofstorage
    Depreciation_B = B_LR .* params.deltaB
    cs = sum(Depreciation_B)
    cumsum = cs .* (params.iota).^(1:500)
    Qtotal_LR_B = sum(cumsum)
    p_B = (Initialprod .* (params.iota) .^ (T) + Qtotal_LR_B) .^ (-params.gammaB)
    return p_B
end

function fill_cs_vec!(cumsum::Vector{Float64}, Depreciation_LR::Float64, iota::Float64, T::Int64)
    @inbounds for i = 1:T
        cumsum[i] = Depreciation_LR * (iota ^ i)
    end
end

function update_battery(KR_LR::Vector{Float64}, hoursofstorage::Int64, params::StructParams)
    cumsum = Vector{Float64}(undef, 500)

    B_LR = KR_LR .* hoursofstorage
    Depreciation_B = B_LR .* params.deltaB
    cs = sum(Depreciation_B)

    fill_cs_vec!(cumsum, cs, params.iota, 500)
    Qtotal_LR_B = sum(cumsum)
    p_B = (0.001 + Qtotal_LR_B) ^ (-params.gammaB) # Shift avoids NAN when 0 investment
    return p_B
end

function gen_curtailment!(curtailmentfactor_S::Matrix{Float64}, curtailmentfactor_W::Matrix{Float64}, curtailmentfactor::Vector{Float64}, # modified variables
    N::Int64, majorregions::DataFrame, YF_LR::Vector{Float64}, YE_LR::Vector{Float64}, SShare_LR::Matrix{Float64}, hoursofstorage::Int64, interp3)
    SShare_region_LR = zeros(2531, 12)
    renewshare_LR = zeros(2531, 12)

    @inbounds for kk in 1:(N - 1)
        ind = majorregions.rowid2[kk]:majorregions.rowid[kk]

        renewshare_LR[1, kk] = 1 - ((sum(YF_LR[ind, :]) ./ sum(YE_LR[ind, :])))
        SShare_region_LR[1, kk] = sum(SShare_LR[ind] .* YE_LR[ind] / sum(YE_LR[ind, :]))
        curtailmentfactor_S[kk] = renewshare_LR[1, kk] * SShare_region_LR[1, kk]
        curtailmentfactor_W[kk] = renewshare_LR[1, kk] * (1 - SShare_region_LR[1, kk])

        itp_values = interp3(curtailmentfactor_W[kk], curtailmentfactor_S[kk], hoursofstorage)
        curtailmentfactor[ind, 1] .= itp_values
    end

    # generate curtailment factor for off grid-regions
    kk = N
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    renewshare_LR[ind, 1] .= 1 .- (YF_LR[ind,:] ./ YE_LR[ind,:])   
    SShare_region_LR[ind, 1] = SShare_LR[ind]   
    curtailmentfactor_S[ind] .= renewshare_LR[ind, 1] .* SShare_region_LR[ind]   
    curtailmentfactor_W[ind] .= renewshare_LR[ind, 1] .* (1 .- SShare_region_LR[ind])

    x_vals = curtailmentfactor_S[ind]
    y_vals = curtailmentfactor_W[ind]
    z_vals = repeat([hoursofstorage], length(ind))

    @inbounds for i in eachindex(ind)
        curtailmentfactor[ind[i], 1] = interp3(y_vals[i], x_vals[i], z_vals[i])
    end
end

# ---------------------------------------------------------------------------- #
#                           SteadyState.jl Functions                           #
# ---------------------------------------------------------------------------- #

function fill_wr!(wr::Vector{Float64}, wageresults::Matrix{Float64})
    @inbounds for i in 1:2531
        wr[i] = (wageresults[i, 2] / wageresults[i, 1]) - 1
    end
end

function up_e2_LR!(e2_LR::Matrix{Float64}, Vs::Matrix{Float64}, laboralloc::Matrix{Float64}, D_LR::Vector{Float64})
    VS2 = Vs[:,2]'
    VS3 = Vs[:,3]'
    @inbounds for i in 1:10
        e2_LR[:, i] .= laboralloc[:, i] .* D_LR .* VS2[i] ./ (VS2[i] + VS3[i])
    end
end

mutable struct StructPowerOutput
    KR_LR_S::Matrix{Float64} #
    KR_LR_W::Matrix{Float64} #
    p_E_LR::Matrix{Float64} #
    D_LR::Vector{Float64} #
    YE_LR::Vector{Float64} #
    PC_guess_LR::Matrix{Float64}
    PI_LR::Vector{Float64} #
    w_LR::Matrix{Float64} #
    laboralloc_LR::Matrix{Float64}
    p_KR_bar_LR::Matrix{Float64}
    KR_LR::Vector{Float64} #
    KF_LR::Matrix{Float64} #
    p_KR_LR_S::Float64 ##
    p_KR_LR_W::Float64 ##
    p_B::Float64 #
    p_F_LR::Float64 #
    Lsector::Matrix{Float64}
    YF_LR::Vector{Float64} #
    diffK::Float64 #
    diffp::Float64 #
    result_Dout_LR::Matrix{Any} #
    result_Yout_LR::Matrix{Any} #
    result_YFout_LR::Vector{Any} #
    result_price_LR::Vector{Any} #
    KP_LR::Matrix{Float64} ##
    rP_LR::Matrix{Float64} #
    Depreciation_LR_S::Float64 ##
    Depreciation_LR_W::Float64 ##
    w_real::Vector{Float64} #
    PC_LR::Vector{Float64} #
end

function solve_power_output(RWParams::StructRWParams, params::StructParams, RunBatteries::Int, RunCurtailment::Int,
    Initialprod::Int, R_LR::Float64, majorregions::DataFrame, Linecounts::DataFrame, linconscount::Int,
    regionParams::StructRWParams, curtailmentswitch::Int, interp3, T::Int, kappa::Float64, mrkteq::StructMarketEq, config::ModelConfig, 
    pB_shifter::Float64, G::String)
    

    # ---------------------------------------------------------------------------- #
    #                                Initialization                                #
    # ---------------------------------------------------------------------------- #

    # Vectors and Matrices
    KF_LR = Matrix{Float64}(undef, 2531, 1)
    Lsector = Matrix{Float64}(undef, 2531, 10)
    KR_LR = Vector{Float64}(undef, 2531)

    sizes = [727, 755, 30, 53, 13, 15, 46, 11, 26, 78, 125, 320, 332]
    last_element = [reshape(Vector{Float64}(undef, sizes[end]), 1, :)]

    Lossfac_LR = Matrix{Float64}(undef, 1, 12)
    result_price_LR = [Vector{Float64}(undef, size) for size in sizes[1:end-1]]
    result_price_LR = [result_price_LR..., last_element]
    result_YFout_LR = [Vector{Float64}(undef, size) for size in sizes[1:end-1]]
    result_YFout_LR = [result_YFout_LR..., last_element]
    D_LR = Vector{Float64}(undef, 2531)
    YE_LR = Vector{Float64}(undef, 2531)
    YF_LR = Vector{Float64}(undef, 2531)
    PI_LR = Vector{Float64}(undef, 2531)
    curtailmentfactor_S = zeros(1, 2531)
    curtailmentfactor_W = zeros(1, 2531)
    curtailmentfactor = Vector{Float64}(undef, 2531)
    rP_LR = Matrix{Float64}(undef, 2531, 1)
    pE_S_FE = Vector{Float64}(undef, 2531)

    pg_LR_s = Matrix{Float64}(undef, 2531, 10)
    KP_LR = Matrix{Float64}(undef, 2531, 1)
    w_real = Vector{Float64}(undef, 2531)
    PC_LR = Vector{Float64}(undef, 2531)
    SShare_LR = Matrix{Float64}(undef, 2531, 1)
    thetabar_LR = Matrix{Float64}(undef, 2531, 1)
    p_KR_bar_LR = Matrix{Float64}(undef, 2531, 1)

    Depreciation_LR_S = 0.0
    Depreciation_LR_W = 0.0
    p_B = 0.0
    p_addon = 0.0
    p_KR_LR_S = 0.0
    p_KR_LR_W = 0.0

    # Constants
    p_F_LR = 1.0

    laboralloc_LR, KR_LR_S, KR_LR_W, p_E_LR, w_LR, result_Dout_LR, result_Yout_LR, PC_guess_LR = ss_load_mat(G) # 2.506 ms (981 allocations: 377.28 KiB)

    # ---------------------------------------------------------------------------- #
    #                              Set Initial Guesses                             #
    # ---------------------------------------------------------------------------- #

    KF_LR .= RWParams.KF ./ 10000    # add in minimum of generation ; 1.370 μs (2 allocations: 64 bytes)
    Lsector .= laboralloc_LR .* params.L # 4.500 μs (2 allocations: 64 bytes)
    KR_LR_S .+= 0.01
    KR_LR_W .+= 0.01
    KR_LR .= KR_LR_S .+ KR_LR_W

    if RunBatteries==0 && RunCurtailment==0
        pB_shifter = 0
        config.hoursofstorage=0
    end

    # set long-run capital returns
    rP_LR .= (R_LR - 1 + params.deltaP) .* PC_guess_LR     # long run return on production ; 721.642 ns (5 allocations: 112 bytes)

    # ---------------------------------------------------------------------------- #
    #                          SOLVE LONG RUN STEADYSTATE                          #
    # ---------------------------------------------------------------------------- #

    # --------------------- Initialize Intermediate Variables -------------------- #
    niters = 1
    niters_in = 1
    diffK = 1.0
    diffp = 1.0    

    while diffK > 10^(-2)
        println("Number of iterations outer while loop: ", niters)
        println("Diffk: ", diffK)
        println("Number of iterations inner while loop: ", niters_in)

        diffend = 1
        jj=1

        # ------------------------- set long run goods prices ------------------------ #
        pg_LR_s .= w_LR .^ (params.Vs[:, 1]' .* ones(params.J, 1)) .* 
                p_E_LR .^ ((params.Vs[:, 2]'.* ones(params.J, 1)) .+ (params.Vs[:, 3]'.* ones(params.J, 1))) .* 
                ((params.kappa .+ (params.kappa .* p_F_LR ./ p_E_LR) .^ (1 .- params.psi)) .^ (-(params.psi ./ (params.psi .- 1)) .* params.Vs[:, 3]')) .* rP_LR .^ (params.Vs[:, 4]'.* ones(params.J, 1)) ./ 
                (params.Z .* params.zsector .* params.cdc) # 1.680 ms (79 allocations: 83.02 KiB)


        ss_optimize_region!(result_price_LR, result_Dout_LR, result_Yout_LR, result_YFout_LR, Lossfac_LR, # modified variables
                pg_LR_s, majorregions, Linecounts, RWParams, laboralloc_LR, Lsector, params, w_LR, 
                rP_LR, p_E_LR, kappa, regionParams, KF_LR, p_F_LR, linconscount, KR_LR_S, KR_LR_W, mrkteq.result_Yout_init) # 37.425073 seconds (3.52 M allocations: 1.074 GiB, 0.55% gc time)

        ss_optimize_offgrid!(result_Dout_LR, result_Yout_LR, result_price_LR, result_YFout_LR, # modified variables
                    majorregions, Lsector, mrkteq.laboralloc, params, w_LR, rP_LR, pg_LR_s, p_E_LR, kappa, regionParams,
                    KR_LR_S, KR_LR_W, KF_LR, p_F_LR)

        fill_LR!(p_E_LR, D_LR, YE_LR, YF_LR, PI_LR, result_price_LR, # modified variables
                majorregions, result_Dout_LR, result_Yout_LR, result_YFout_LR, params.L, params.N)

        jj = jj+1

        # get production capital vec        
        up_KP_LR!(KP_LR, Lsector, params.Vs, params.J, w_LR, rP_LR) # 58.900 μs (15 allocations: 257.75 KiB)

        # update prices and wages

        up_prices_wages!(w_LR, w_real, PC_LR, laboralloc_LR, Lsector, # modified variables
                        p_E_LR, p_F_LR, D_LR, YE_LR, rP_LR, KP_LR, PI_LR, params) # 5.447 s (316 allocations: 1.48 GiB)

        diffend = 0.01
        niters_in += 1

        # update consumption price guess
        PC_guess_LR .= 0.2 .* PC_LR .+ (1 - 0.2) .* PC_guess_LR #   1.650 μs (6 allocations: 224 bytes)

        # update capital prices

        Depreciation_LR_S, Depreciation_LR_W, p_KR_LR_S, p_KR_LR_W = up_cap_prices(KR_LR_S, KR_LR_W, params.deltaR, params.iota, T, Initialprod, params.gammaS, params.gammaW) #   14.600 μs (10 allocations: 43.89 KiB)

        # get solar shares

        calc_solar_shares!(SShare_LR::Matrix{Float64}, thetabar_LR::Matrix{Float64}, p_KR_bar_LR::Matrix{Float64}, # modified variables
                    regionParams.thetaS::Matrix{Float64}, p_KR_LR_S::Float64, params.varrho::Float64, regionParams.thetaW::Matrix{Float64}, p_KR_LR_W::Float64) #   88.200 μs (19 allocations: 178.75 KiB)

        jj = jj + 1

        # update battery prices
        p_B = update_battery(KR_LR, config.hoursofstorage, params) #   8.175 μs (6 allocations: 43.80 KiB)
        p_addon = pB_shifter * config.hoursofstorage * p_B

        # generate curtailment factor for grid regions
        gen_curtailment!(curtailmentfactor_S, curtailmentfactor_W, curtailmentfactor, # modified variables
            params.N, majorregions, YF_LR, YE_LR, SShare_LR, config.hoursofstorage, interp3) #   136.000 μs (2518 allocations: 723.05 KiB)


        diffK, diffp = ss_update_params!(p_KR_bar_LR, p_addon, params, R_LR, regionParams, thetabar_LR, curtailmentswitch,
                    curtailmentfactor, p_E_LR, KR_LR, KR_LR_S, KR_LR_W, SShare_LR, pE_S_FE, config)

        
        niters += 1
    end

    return StructPowerOutput(
        KR_LR_S,
        KR_LR_W,
        p_E_LR,
        D_LR,
        YE_LR,
        PC_guess_LR,
        PI_LR,
        w_LR,
        laboralloc_LR,
        p_KR_bar_LR,
        KR_LR,
        KF_LR,
        p_KR_LR_S,
        p_KR_LR_W,
        p_B,
        p_F_LR,
        Lsector,
        YF_LR,
        diffK,
        diffp,
        result_Dout_LR,
        result_Yout_LR,
        result_YFout_LR,
        result_price_LR,
        KP_LR,
        rP_LR,
        Depreciation_LR_S,
        Depreciation_LR_W,
        w_real,
        PC_LR
    )
end

# ---------------------------------------------------------------------------- #
#                                   Addendum                                   #
# ---------------------------------------------------------------------------- #
# ----------------------------- unused variables ----------------------------- #
"""
maxF_SS=KF_LR
pFK=1                  # price of fossil fuel capital

Deprecations_S = Vector{Float64}(undef, 2531)
Deprecations_W = Vector{Float64}(undef, 2531)

# get guess for long run renewable capital prices
Deprecations_S .= KR_LR_S .* params.deltaR # unused
Deprecations_W .= KR_LR_W .* params.deltaR # unused

const c_S = sum(Deprecations_S) # intermediate var
const c_W = sum(Deprecations_W) # intermediate var
cumsum_S .= c_S .* (params.iota) .^ (1:500) # intermediate var
cumsum_W = c_W .* (params.iota) .^ (1:500) # intermediate var
Qtotal_LR_S = sum(cumsum_S) # unused
Qtotal_LR_W = sum(cumsum_W) # unused

p_B = set_battery(KR_LR, config.hoursofstorage, params, Initialprod, T) # 9.000 μs (6 allocations: 43.80 KiB); unused
Ddiff = 1
"""


end