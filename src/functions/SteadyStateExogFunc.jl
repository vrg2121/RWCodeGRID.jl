module SteadyStateExogFunc

import DataFrames: DataFrame
import MAT: matopen
import Statistics: mean
import ..DataLoadsFunc: StructGsupply, StructRWParams
using JuMP, Ipopt
using ..RegionModel, ..MarketEquilibrium

import DrawGammas: StructParams
import ..ModelConfiguration: ModelConfig
export StructPowerOutputExog, solve_power_output_exog


function ss_optimize_region!(result_price_LR::Vector, result_Dout_LR::Matrix, result_Yout_LR::Matrix, result_YFout_LR::Vector, Lossfac_LR::Matrix,
        pg_LR_s::Matrix, majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc_LR::Matrix, Lsector::Matrix, params::StructParams, 
        w_LR::Matrix, rP_LR::Vector, p_E_LR::Matrix, kappa::Float64, regionParams::StructRWParams, KF_LR::Matrix, p_F_LR::Int,
        linconscount::Int, KR_LR_S::Matrix, KR_LR_W::Matrix)

        n_regions = params.N - 1
        tasks = Vector{Task}(undef, n_regions)
    
        for kk in 1:n_regions
            tasks[kk] = Threads.@spawn begin
                l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid, Gammatrix, linecons = data_set_up(kk, majorregions, Linecounts, RWParams, laboralloc_LR, Lsector, 
                                params, w_LR, rP_LR, pg_LR_s, p_E_LR, kappa, regionParams, 
                                KF_LR, p_F_LR, linconscount, KR_LR_S, KR_LR_W, "steadystate")
                
                # Solve the model for region kk.
                P_out = solve_model(kk, l_guess, LB, UB, guess, regionParams, params, 
                                        power, shifter, KFshifter, KRshifter, p_F_LR, mid, Gammatrix, linecons)
                
                # Compute local outputs.
                local_price  = Price_Solve(P_out, shifter, n, params)
                local_Dout   = P_out[1:(end ÷ 2)]
                local_Yout   = P_out[(end ÷ 2 + 1):end]
                local_YFout  = local_Yout .- KRshifter
                Pvec         = local_Yout .- local_Dout
                Losses       = Pvec[2:end]' * regionParams.B[kk] * Pvec[2:end] .* params.Rweight
                local_Lossfac = Losses / sum(local_Yout)
                
                return (kk = kk, local_price = local_price, local_Dout = local_Dout,
                        local_Yout = local_Yout, local_YFout = local_YFout, local_Lossfac = local_Lossfac)
            end
        end
    
        for kk in 1:n_regions
            result = fetch(tasks[kk])
            result_price_LR[result.kk] .= result.local_price
            result_Dout_LR[result.kk] .= result.local_Dout
            result_Yout_LR[result.kk] .= result.local_Yout
            result_YFout_LR[result.kk] .= result.local_YFout
            Lossfac_LR[1, result.kk] = result.local_Lossfac
        end
end

function ss_second_loop(majorregions::DataFrame, Lsector::Matrix, laboralloc::Matrix, params::StructParams, w_LR::Matrix, rP_LR::Union{Matrix, Vector},
    result_Dout_LR::Matrix, result_Yout_LR::Matrix, pg_LR_s::Matrix, p_E_LR::Matrix, kappa::Float64,
    regionParams::StructRWParams, KR_LR_S::Matrix, KR_LR_W::Matrix, KF_LR::Matrix, kk::Int, p_F_LR::Int)
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    nr = majorregions.rowid[kk] - majorregions.rowid2[kk] + 1
    Kshifter = Matrix{Float64}(undef, nr, size(Lsector, 2))

    # get prices for places off the grid
    # set up optimization problem for region kk
    @views secalloc = laboralloc[ind, :]
    @views Lshifter = Lsector[ind, :]
    @views Kshifter .= Lsector[ind, :] .* (params.Vs[:,4]'.* ones(majorregions.n[kk], 1)) ./
            ((params.Vs[:,1]'.* ones(majorregions.n[kk], 1))) .* (w_LR[ind] ./ rP_LR[ind])
    @views Ltotal = sum(Lshifter, dims=2)
    Jlength = majorregions.n[kk]
    @views guess = [result_Dout_LR[kk]; result_Yout_LR[kk]]

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
    @views KRshifter = regionParams.thetaS[ind] .* KR_LR_S[ind] .+
                regionParams.thetaW[ind] .* KR_LR_W[ind]
    @views KFshifter = KF_LR[ind]
    # define bounds
    @views YFmax = KF_LR[ind]

    return KRshifter, YFmax, guess, power, KFshifter, shifter
end


function new_obj2(x...)
    return obj2(collect(x), power[jj], shifter[jj], KFshifter[jj], KRshifter[jj], p_F_LR, params)
end

function new_grad2(g, x...)
    g = grad_f(collect(x), power[jj], shifter[jj], KFshifter[jj], KRshifter[jj], p_F_LR, params)
    return
end

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

function ss_update_params!(p_E_LR::Matrix, KR_LR::Matrix, 
                            KR_LR_S::Matrix, KR_LR_W::Matrix, SShare_LR::Matrix, diffK, pE_S_FE::Matrix,
                            config::ModelConfig)

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
    diffK = mean(abs.((kfac .- 1) .* KR_LR_dash))
    return diffK
end

function new_obj_f(x...)
    return obj(collect(x), power, shifter, KFshifter, KRshifter, p_F_LR, params)
end

function new_grad_f(g, x...)
    g = grad_f(collect(x), power, shifter, KFshifter, KRshifter, p_F_LR, params)
    return
end

function set_battery(KR_LR::Matrix, hoursofstorage::Int64, params::StructParams, Initialprod::Int64, T::Int64)
    B_LR = KR_LR .* hoursofstorage
    Depreciation_B = B_LR .* params.deltaB
    cumsum = sum(Depreciation_B)
    cumsum = cumsum .* (params.iota).^(1:500)
    Btotal_LR = sum(cumsum)
    p_B = (Initialprod .* (params.iota) .^ (T) + Btotal_LR) .^ (-params.gammaB)
    return p_B
end

function update_battery(KR_LR::Matrix, hoursofstorage::Int64, params::StructParams)
    B_LR = KR_LR .* hoursofstorage
    Depreciation_B = B_LR .* params.deltaB
    cumsum = sum(Depreciation_B)
    cumsum = cumsum .* (params.iota .^ (1:500))
    Btotal_LR = sum(cumsum)
    p_B = (0.001 .+ Btotal_LR) .^ (-params.gammaB) # Shift avoids NAN when 0 investment
    return p_B
end

mutable struct StructPowerOutputExog
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
    result_Dout_LR::Matrix{Any} #
    result_Yout_LR::Matrix{Any} #
    result_YFout_LR::Vector{Any} #
    result_price_LR::Vector{Any} #
    KP_LR::Matrix{Float64} ##
    rP_LR::Matrix{Float64} #
    w_real::Vector{Float64} #
    PC_LR::Vector{Float64} #
end

function solve_power_output_exog(RWParams::StructRWParams, params::StructParams, RunBatteries::Int,
    Initialprod::Int, R_LR::Float64, majorregions::DataFrame, Linecounts::DataFrame, linconscount::Int,
    regionParams::StructRWParams, pB_shifter::Float64, T::Int, mrkteq::NamedTuple, 
    projectionssolar::Matrix, projectionswind::Matrix, config::ModelConfig, exogindex::Int, 
    p_KR_init_S::Float64, p_KR_init_W::Float64, kappa::Float64, G::String)

    laboralloc_LR, KR_LR_S, KR_LR_W, p_E_LR, w_LR, result_Dout_LR, result_Yout_LR, PC_guess_LR = ss_load_mat(G);

    global laboralloc_LR
    global KR_LR_S
    global KR_LR_W
    global p_E_LR
    global w_LR
    global result_Dout_LR
    global result_Yout_LR
    global PC_guess_LR



    KF_LR=RWParams.KF/10000    # add in minimum of generation
    #maxF_SS=KF_LR
    #pFK=1                  # price of fossil fuel capital

    p_F_LR=10

    ####

    # get guess for long run renewable capital prices
    p_KR_LR_S = p_KR_init_S * projectionssolar[end, exogindex+1]
    p_KR_LR_W = p_KR_init_W .* projectionswind[end, exogindex+1]

    Capinvest = ones(params.J, 1)

    global Lsector = laboralloc_LR .* params.L

    KR_LR_S = KR_LR_S .+ 0.01
    KR_LR_W = KR_LR_W .+ 0.01
    KR_LR = KR_LR_S .+ KR_LR_W

    if RunBatteries==0
        pB_shifter = 0
        config.hoursofstorage=0
    end

    # output p_B
    p_B = set_battery(KR_LR, config.hoursofstorage, params, Initialprod, T)


    # initialise run
    #Ddiff = 1
    niters = 1
    niters_in = 1
    diffK = 1.0

    # initiate storage vectors
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
    rP_LR = Vector{Float64}(undef, 2531)


    rP_LR .= (R_LR - 1 + params.deltaP) .* PC_guess_LR

    while diffK > 10^(-2)
        println("Number of iterations outer while loop: ", niters)
        diffend = 1
        tol = 0.1
        jj=1

        #while diffend > tol
        # set long run goods prices 
        pg_LR_s = w_LR .^ (params.Vs[:, 1]' .* ones(params.J, 1)) .* 
            p_E_LR .^ ((params.Vs[:, 2]'.* ones(params.J, 1)) .+ (params.Vs[:, 3]'.* ones(params.J, 1))) .* 
            ((params.kappa .+ (params.kappa .* p_F_LR ./ p_E_LR) .^ (1 .- params.psi)) .^ (-(params.psi ./ (params.psi .- 1)) .* params.Vs[:, 3]')) .* rP_LR .^ (params.Vs[:, 4]'.* ones(params.J, 1)) ./ 
            (params.Z .* params.zsector .* params.cdc)


        ss_optimize_region!(result_price_LR, result_Dout_LR, result_Yout_LR, result_YFout_LR, Lossfac_LR,
                pg_LR_s, majorregions, Linecounts, RWParams, laboralloc_LR, Lsector, params, w_LR, 
                rP_LR, p_E_LR, kappa, regionParams, KF_LR, p_F_LR, linconscount, KR_LR_S, KR_LR_W)


        kk = params.N

        KRshifter, YFmax, guess, power, KFshifter, shifter = ss_second_loop(majorregions, Lsector, mrkteq.laboralloc, params, w_LR, rP_LR,
                                                                result_Dout_LR, result_Yout_LR, pg_LR_s, p_E_LR, kappa,
                                                                regionParams, KR_LR_S, KR_LR_W, KF_LR, kk, p_F_LR)

        for kk=1:(params.N-1)
            ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
            p_E_LR[ind] .= result_price_LR[kk]
            D_LR[ind].= result_Dout_LR[kk]
            YE_LR[ind] .= result_Yout_LR[kk]
            YF_LR[ind] .= result_YFout_LR[kk]
            PI_LR[ind, 1] .= sum(result_price_LR[kk] .* (result_Dout_LR[kk].- result_Yout_LR[kk])) .*
                            params.L[ind, 1] ./
                            sum(params.L[ind, 1])
        end

        mm = majorregions.n[kk] 
        P_out2 = Matrix{Float64}(undef, 2, mm)
        for jj in 1:mm
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

            P_out2[:, jj] .= value.(x)
            price = Price_Solve(P_out2[:, jj], shifter[jj], 1, params)[1]

            result_Dout_LR[kk][jj] = P_out2[1, jj]        
            result_Yout_LR[kk][jj] = P_out2[2, jj] 
            result_price_LR[kk][1][1, jj] = price # this is fine
        end

        for jj in 1:mm
            result_YFout_LR[kk][1][1, jj] = P_out2[1, jj] - KRshifter[jj]
        end

        ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
        YF_LR[ind] .= vec(result_YFout_LR[kk][1])
        result_price_LR[kk][1] .= clamp.(result_price_LR[kk][1], 0.001, 1) # bound the prices
        p_E_LR[ind] .= vec(result_price_LR[kk][1])
        D_LR[ind] .= vec(result_Dout_LR[kk])
        YE_LR[ind] .= vec(result_Yout_LR[kk])
        PI_LR[ind] .= sum(result_price_LR[kk][1] .* (result_Dout_LR[kk]-result_Yout_LR[kk])) .*
                                                                params.L[ind, 1] ./ sum(params.L[ind, 1])

        jj = jj+1

        # get production capital vec
        Ksector = Lsector .* (params.Vs[:,4]'.* ones(params.J, 1)) ./ (params.Vs[:,1]'.* ones(params.J, 1)) .* (w_LR ./ rP_LR)
        global KP_LR = sum(Ksector, dims = 2)

        # update prices and wages
        w_update, w_real, Incomefactor, PC_LR, Xjdashs = wage_update_ms( w_LR, p_E_LR, p_E_LR, p_F_LR, D_LR, YE_LR, rP_LR, KP_LR, PI_LR, 0, params);
        #diff_w = maximum(abs.(w_LR .- w_update) ./ w_LR)
        global w_LR = 0.2 .* w_update .+ (1 - 0.2) .* w_LR
        global w_real
        global PC_LR
        # update sectoral allocations
        global laboralloc_LR = Lsector ./ params.L
        relexp = Xjdashs ./ (Xjdashs[:,1] .* ones(1, params.I))
        relab = laboralloc_LR ./ (laboralloc_LR[:,1].* ones(1, params.I))
        global Lsector = Lsector .* clamp.(1 .+ 0.2 .* (relexp .- relab) ./ relab, 0.8, 1.2)
        Lsector = Lsector ./ sum(Lsector, dims=2) .* params.L

        #diffend = 0.01
        niters_in += 1
        #end

        # update consumption price guess
        PC_guess_LR .= 0.2 .* PC_LR .+ (1 - 0.2) .* PC_guess_LR

        # don't update capital prices

        # get solar shares
        SShare_LR = (regionParams.thetaS ./ p_KR_LR_S) .^ params.varrho ./ ((regionParams.thetaS ./ p_KR_LR_S) .^ params.varrho + (regionParams.thetaW ./ p_KR_LR_W) .^ params.varrho)
        thetabar_LR = regionParams.thetaS .* SShare_LR + regionParams.thetaW .* (1 .- SShare_LR)
        global p_KR_bar_LR = SShare_LR .* p_KR_LR_S + (1 .- SShare_LR) .* p_KR_LR_W

        jj = jj + 1

        # update battery prices
        p_B = update_battery(KR_LR, config.hoursofstorage, params)
        p_addon = pB_shifter .* config.hoursofstorage .* p_B

        # if market price is above free entry price, increase renewable capital; decrease otherwise
        pE_S_FE = (p_KR_bar_LR .+ p_addon .- (p_KR_bar_LR .+ p_addon) .* (1-params.deltaR) ./ R_LR) .* regionParams.costshifter ./ thetabar_LR
        Capinvest .= p_E_LR .> pE_S_FE

        diffK = ss_update_params!(p_E_LR, KR_LR, KR_LR_S, KR_LR_W, SShare_LR, diffK, pE_S_FE, config)
        println(diffK)
        niters += 1
        global KRshifter
    end
    return StructPowerOutputExog(
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
        result_Dout_LR,
        result_Yout_LR,
        result_YFout_LR,
        result_price_LR,
        KP_LR,
        rP_LR,
        w_real,
        PC_LR
    )
end


end