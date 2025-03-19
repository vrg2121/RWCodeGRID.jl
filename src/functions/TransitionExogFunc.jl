module TransitionExogFunc

import DataFrames: DataFrame
using JuMP, Ipopt, Interpolations
import LinearAlgebra: Transpose, I, Adjoint
import Random: Random
import SparseArrays: sparse
import ..RegionModel: solve_model
import ..DataLoadsFunc: StructGsupply, StructRWParams
import ..ParamsFunctions: StructParams
import ..ModelConfiguration: ModelConfig
using Ipopt, JuMP

using ..MarketEquilibrium

export solve_transitioneq_exog

function sectoral_allocations!(Lsectorpath_guess::Array{Float64, 3}, laboralloc_path::Array{Float64, 3}, decayp::Float64, params::StructParams, laboralloc_init::Matrix, sseqE::NamedTuple, T::Int)
    lpg = [exp(decayp * i) for j in 1:params.J, i in 0:T, k in 1:params.I]
    Lsectorpath_guess .= permutedims(lpg, [1, 3, 2])
    laboralloc_path .= sseqE.laboralloc_LR .+ (laboralloc_init .- sseqE.laboralloc_LR) .* Lsectorpath_guess
    Lsectorpath_guess .= laboralloc_path .* (params.L .* ones(1, params.I, T + 1))
end

function update_battery_prices!(KR_path::Matrix{Float64}, Qtotal_path_B::Matrix{Float64}, # variables to be modified
    KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, params::StructParams, 
    hoursofstorage::Int, RWParams::StructRWParams, Initialprod::Int)

    # initialize intermediate variables
    DepreciationB = Matrix{Float64}(undef, 2531, 501)
    I_pathB = Matrix{Float64}(undef, 2531, 500)
    I_total_pathB = Matrix{Float64}(undef, 1, 500)
    cumsum_B = Matrix{Float64}(undef, 1, 500)

    # update battery prices
    KR_path .= KR_path_S .+ KR_path_W
    DepreciationB .= KR_path .* params.deltaB .* hoursofstorage
    I_pathB = max.((KR_path[:, 2:end] - KR_path[:, 1:end-1]) .* hoursofstorage + DepreciationB[:, 1:end-1], 0)
    I_total_pathB = sum(I_pathB, dims=1)
    

    for i=1:length(I_total_pathB)
        csB = reverse(I_total_pathB[1:i])
        csB .= csB .* (params.iota) .^ (1+i)
        csB = csB'
        cumsum_B = csB
        Qtotal_path_B[i + 1] = sum(cumsum_B) + (Initialprod + sum(RWParams.KR .* hoursofstorage)) * (params.iota) .^ (i+1)
    end

    Qtotal_path_B[1] = Initialprod+sum(RWParams.KR) * hoursofstorage
end

function data_set_up_transition(t::Int, kk::Int, majorregions::DataFrame, Linecounts::DataFrame, RWParams, laboralloc_path::Array, Lsectorpath_guess::Array, params,
    w_path_guess::Union{Matrix, Vector}, rP_path::Matrix, pg_path_s::Array, p_E_path_guess::Union{Vector, Matrix}, kappa::Int, regionParams, KF_path::Matrix, p_F_path_guess::Transpose, 
    linconscount::Int, KR_path_S::Matrix, KR_path_W::Matrix)
    local ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    local n = majorregions.n[kk]
    local l_ind = Linecounts.rowid2[kk]:Linecounts.rowid[kk]
    local gam = RWParams.Gam[kk]
    local l_n = Linecounts.n[kk]

    local secalloc = laboralloc_path[ind, :, t]
    local Lshifter = Lsectorpath_guess[ind,:,t]
    local Kshifter=Lsectorpath_guess[ind,:,t] .* (params.Vs[:,4]' .* ones(n,1)) ./
            (params.Vs[:,1]' .* ones(n,1)) .*
            (w_path_guess[ind, t] ./ rP_path[ind,t])
    #Ltotal = sum(Lshifter, dims=2)

    # define data for inequality constraints
    local linecons = RWParams.Zmax[l_ind]
    local Gammatrix = zeros(size(gam, 1), size(gam, 2) + 1)
    local Gammatrix[:, 2:end] = gam 
    if linconscount < l_n
        Random.seed!(1)
        randvec = rand(l_n)
        randvec = randvec .> (linconscount / l_n)
        Gammatrix[randvec, :] .= 0
    end
    local stacker = [-Matrix(I, n, n) Matrix(I, n, n)]
    local Gammatrix = sparse(Gammatrix) * sparse(stacker)
    local Gammatrix = [Gammatrix; -Gammatrix]
    local linecons = [linecons; linecons]

    # define shifters for objective function
    local pg_s = pg_path_s[ind, :, t]
    local p_F_in = p_F_path_guess[:,t]
    local prices = p_E_path_guess[ind,t] .* ones(1, params.I)
    local power = (params.Vs[:,2]' .* ones(n, 1)) .+ (params.Vs[:,3]' .* ones(n, 1))
    local shifter = pg_s .*
            (kappa .+ (prices ./ (kappa .* p_F_in)) .^ (params.psi - 1)) .^ (params.psi / (params.psi-1) .* (params.Vs[:,3]' .* ones(n,1))) .*
            (1 .+ (params.Vs[:,3]' .* ones(n,1)) ./ (params.Vs[:,2]' .* ones(n, 1))) .^ (-(params.Vs[:,2]' .* ones(n,1)) .- (params.Vs[:,2]' .* ones(n,1))) .*
            params.Z[ind] .*
            params.zsector[ind, :] .*
            Lshifter .^ (params.Vs[:,1]' .* ones(n,1)) .*
            Kshifter .^ (params.Vs[:,4]' .* ones(n,1)) 
    local shifter = shifter .* secalloc .^ power
    local KRshifter = regionParams.thetaS[ind] .* KR_path_S[ind, t] .+ regionParams.thetaW[ind] .* KR_path_W[ind, t]
    local KFshifter = KF_path[ind, t]

    # define bounds
    local YFmax = KF_path[ind,t] # DIFFERENT
    local LB = [zeros(n); KRshifter]
    local UB = [fill(1000, n); YFmax .+ KRshifter .+ 0.01]

    # define guess
    local guess = [KRshifter; KRshifter .+ 0.0001]
    local l_guess = length(guess)
    local mid = l_guess ÷ 2

    return l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid, p_F_in

end
    
function transition_electricity_US_Europe!(result_price_path::Matrix, result_Dout_path::Matrix, result_Yout_path::Matrix, result_YFout_path::Matrix, # modified variables
    majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc_path::Array,
    Lsectorpath_guess::Array, params::StructParams, w_path_guess::Matrix{Float64}, rP_path::Matrix{Float64},
    linconscount::Int, pg_path_s::Array, p_F_path_guess::Transpose, p_E_path_guess::Matrix{Float64},
    kappa::Int, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, KF_path::Matrix{Float64}, capT::Int, regionParams::StructRWParams)

    for kk=1:2
        #kk=1
        
        Threads.@threads for t in 1:capT
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


function transition_electricity_other_countries!(result_price_path::Matrix, result_Dout_path::Matrix, result_Yout_path::Matrix, result_YFout_path::Matrix, # modified variables
    majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc_path::Array,
    Lsectorpath_guess::Array, params::StructParams, w_path_guess::Matrix{Float64}, rP_path::Matrix{Float64},
    linconscount::Int, pg_path_s::Array, p_F_path_guess::Transpose, p_E_path_guess::Matrix{Float64},
    kappa, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, KF_path::Matrix{Float64}, capT::Int, regionParams::StructRWParams)
                                        # Other countries
    Threads.@threads for kk=3:(params.N - 1)
        for t = 1:capT
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

function transition_electricity_off_grid!(result_price_path::Matrix, result_Dout_path::Matrix, result_Yout_path::Matrix, result_YFout_path::Matrix, # modified variables
    majorregions::DataFrame, laboralloc_path::Array,
    Y_path::Matrix{Float64}, Lsectorpath_guess::Array, params::StructParams, w_path_guess::Matrix{Float64}, rP_path::Matrix{Float64},
    pg_path_s::Array, p_F_path_guess::Transpose, p_E_path_guess::Matrix{Float64},
    kappa::Int, KR_path_S::Matrix{Float64}, KR_path_W::Matrix{Float64}, KF_path::Matrix{Float64}, capT::Int, regionParams::StructRWParams, D_path::Matrix{Float64})

    kk = params.N
    for t=1:capT
        fill!(result_price_path[kk, t], 0)
        fill!(result_Dout_path[kk, t], 0)
        fill!(result_Yout_path[kk, t], 0)
        fill!(result_YFout_path[kk, t], 0)
    end

    for t=1:capT
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

        Threads.@threads for jj = 1:n
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

function fill_paths!(p_E_path_guess::Matrix, D_path::Matrix, Y_path::Matrix, YF_path::Matrix, PI_path::Matrix, 
                    params::StructParams, majorregions::DataFrame, result_price_path::Matrix, result_Dout_path::Matrix, 
                    result_Yout_path::Matrix, result_YFout_path::Matrix, capT::Int)
    Threads.@threads for t in 1:capT
        for kk = 1:params.N
            local ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
            if kk > params.N
                p_E_path_guess[ind, t] .= result_price_path[kk, t]
                D_path[ind, t] .= result_Dout_path[kk, t]
                Y_path[ind, t] .= result_Yout_path[kk, t]
                YF_path[ind, t] .= result_YFout_path[kk, t]
                PI_path[ind, t] .= sum(p_E_path_guess[ind, t] .*
                                    (D_path[ind, t] .- Y_path[ind, t])) .*
                                    params.L[ind, 1] ./ 
                                    sum(params.L[ind, 1])            
            else
                p_E_path_guess[ind, t] .= vec(result_price_path[kk, t])
                D_path[ind, t] .= vec(result_Dout_path[kk, t])
                Y_path[ind, t] .= vec(result_Yout_path[kk, t])
                YF_path[ind, t] .= vec(result_YFout_path[kk, t])
                PI_path[ind, t] .= sum(p_E_path_guess[ind, t] .*
                                    (D_path[ind, t] .- Y_path[ind, t])) .*
                                    params.L[ind, 1] ./ 
                                    sum(params.L[ind, 1])
            end
        end
    end
end

function smooth_prices!(p_E_path_guess::Matrix{Float64},
     D_path::Matrix{Float64}, Y_path::Matrix{Float64}, YF_path::Matrix{Float64},
    sseqE::NamedTuple, expo3::Vector, expo4::Vector, capT::Int, T::Int)
    for kk in capT+1:T
        jj = kk - capT
        p_E_path_guess[:, kk] .= sseqE.p_E_LR .+ ((p_E_path_guess[:, kk-1] .- sseqE.p_E_LR) .* expo3[jj])
    end

    for kk in capT+1:T+1
        jj = kk - capT
        D_path[:, kk] .= sseqE.D_LR .+ ((D_path[:, kk-1] .- sseqE.D_LR) .* expo4[jj])
        Y_path[:, kk] .= sseqE.YE_LR .+ ((Y_path[:, kk-1] .- sseqE.YE_LR) .* expo4[jj])
        YF_path[:, kk] .= sseqE.YF_LR .+ ((YF_path[:, kk-1] .- sseqE.YF_LR) .* expo4[jj]) 
    end
end

function update_fossil_market!(diffpF::Float64, fossilsales_path::Matrix, p_F_path_guess::Transpose{Float64, Vector{Float64}},
                                laboralloc_path::Array, D_path::Matrix, params::StructParams, p_E_path_guess::Matrix, YF_path::Matrix, 
                                KF_path::Matrix, p_F_int, regions::DataFrame, T::Int, interp1, g::Float64, r_path::Adjoint{Float64, Vector{Float64}}, 
                                fusage_total_path::Matrix, p_F_update::Matrix)
    # compute electricity and fossil fuel usage in industry and electricity
    e2_path = Matrix{Float64}(undef, 2531, 501)
    fusage_ind_path = Matrix{Float64}(undef, 2531, 501)
    fusage_power_path = Matrix{Float64}(undef, 2531, 501)

    for t = 1:T+1
        @views e2_path[:, t] .= sum(laboralloc_path[:,:,t] .* (D_path[:, t] .* ones(1, params.I)) .* 
        ((params.Vs[:,2]' .* ones(params.J, 1)) ./ ((params.Vs[:,2]' .* ones(params.J, 1)) .+ (params.Vs[:, 3]' .* ones(params.J, 1)))), dims=2)
        @views fusage_ind_path[:, t] .= (params.kappa) .^ (-1) .* e2_path[:, t] .* (p_E_path_guess[:,t] ./ p_F_path_guess[:, t]) .^ params.psi 
        @views fusage_power_path[:, t] .= (YF_path[:,t] ./ KF_path[:, t] .^ params.alpha2) .^ (1/params.alpha1)
        @views fusage_total_path[:,t] .= (sum(fusage_power_path[:, t]) .+ sum(fusage_ind_path[:, t]))
    end

    # Get S_t from sequence
    S_t = zeros(1, T+1)
    S_t[1] = fusage_total_path[1]
    for i=2:T+1
        S_t[i] = fusage_total_path[i] + S_t[i-1]
    end

    # the initial fossil fuel price is a free variable, can look up in table

    p_F_update[1]=p_F_int
    for t=1:T
        p_F_update[t+1] = (p_F_update[t] - (interp1(S_t[t]) / (1 + g)^t) * (1 - 1/((r_path[t])*(1+g)))) * r_path[t]
    end

    # get fossil sales path for income
    fossilsales_path .= fusage_total_path .* p_F_update .* regions.reserves

    # compute max difference and update fossilfuel price
    diffpF = maximum(abs.(p_F_update[1:100] .- p_F_path_guess[1:100]) ./ p_F_path_guess[1:100])
    p_F_path_guess .= 0.1 .* p_F_update .+ (1 - 0.1) .* p_F_path_guess

end

function solve_transitioneq_exog(R_LR::Float64, GsupplyCurves::StructGsupply, decayp::Float64, T::Int, params::StructParams, sseqE::NamedTuple, KR_init_S::Matrix, 
    KR_init_W::Matrix, mrkteq::NamedTuple, Initialprod::Int, RWParams::StructRWParams, 
    p_KR_bar_init::Matrix, laboralloc_init::Matrix, regionParams::StructRWParams, majorregions::DataFrame, Linecounts::DataFrame, linconscount::Int, 
    kappa::Int, regions::DataFrame, Transiter::Int, st::Matrix, hoursofstorage::Int, pB_shifter::Float64, g::Float64, 
    wage_init::Vector, p_KR_init_S::Float64, p_KR_init_W::Float64, p_F_int::Float64, exogindex::Int, projectionssolar::Matrix, projectionswind::Matrix)
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
    renewshare_path = Matrix{Float64}(undef, 2531, 501)
    KF_path = Matrix{Float64}(undef, 2531, 501)
    p_KR_bar_path = Matrix{Float64}(undef, 2531, 501)
    p_KR_path_guess_S = Matrix{Float64}(undef, 1, 501)
    p_KR_path_guess_W = Matrix{Float64}(undef, 1, 501)

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

    for kk in 1:501

        #Intial guess for long run capital, slowly initially 
        KR_path_S[:, kk] .=  sseqE.KR_LR_S.+((KR_init_S.-sseqE.KR_LR_S) .* expo[kk])
        KR_path_W[:, kk] .=  sseqE.KR_LR_W.+((KR_init_W.-sseqE.KR_LR_W) .* expo[kk])

        # initial gues[:, kk]s for prices
        p_E_path_guess[:, kk] .= sseqE.p_E_LR.+((mrkteq.p_E_init.-sseqE.p_E_LR) .* expo1[kk])

        # initial guess for demand and supply
        D_path[:, kk] .= sseqE.D_LR .+ ((mrkteq.D_init .- sseqE.D_LR) .* expo1[kk])
        # sseqE.D_LR should be greater than D_init (market equilibrium)
        Y_path[:, kk] .= sseqE.YE_LR .+ ((mrkteq.YE_init .- sseqE.YE_LR) .* expo1[kk])

        # initial guess for goods prices
        PC_path_guess[:, kk] .= sseqE.PC_guess_LR.+((mrkteq.PC_guess_init.-sseqE.PC_guess_LR) .* expo1[kk])

        # initial guess for electricity profits
        PI_path[:, kk] .= sseqE.PI_LR.+((mrkteq.PI_init.-sseqE.PI_LR) .* expo1[kk])

        # initial guess for wages
        w_path_guess[:, kk] .= sseqE.w_LR.+((wage_init.-sseqE.w_LR) .* expo1[kk])

        # initial guess for battery path
        B_path[:, kk] .= sseqE.KR_LR.*2 .+ ((RWParams.KR.*2 .- sseqE.KR_LR.*2) .* expo[kk]) 

        renewshare_path[:, kk] .= 1 + (0-1) * expo[kk]

        # capital path
        KF_path[:, kk] .= sseqE.KF_LR.+((RWParams.KF.-sseqE.KF_LR) .* expo2[kk])

        # initial guess for capital prices path
        p_KR_bar_path[:,kk] .= sseqE.p_KR_bar_LR.+((p_KR_bar_init.-sseqE.p_KR_bar_LR) .* expo1[kk])
        #p_KR_path_guess_S[kk] = sseqE.p_KR_LR_S+((p_KR_init_S-sseqE.p_KR_LR_S) * expo[kk])
        #p_KR_path_guess_W[kk] = sseqE.p_KR_LR_W+((p_KR_init_W-sseqE.p_KR_LR_W) * expo[kk])

    end

    p_KR_path_guess_S = p_KR_init_S .* projectionssolar[:, exogindex+1]
    p_KR_path_guess_S = [p_KR_path_guess_S; fill(p_KR_path_guess_S[end], T - size(p_KR_path_guess_S, 1) + 1)]'

    p_KR_path_guess_W = p_KR_init_W .* projectionswind[:, exogindex+1]
    p_KR_path_guess_W = [p_KR_path_guess_W; fill(p_KR_path_guess_W[end], T - length(p_KR_path_guess_W) + 1)]'

    p_KR_path_S=p_KR_path_guess_S
    p_KR_path_W=p_KR_path_guess_W

    # initial guess for sectoral allocations
    sectoral_allocations!(Lsectorpath_guess, laboralloc_path, decayp, params, laboralloc_init, sseqE, T)


    # Initial guess for fossil path
    r = 0.05 * ones(T + 1) 
    p_F_path_guess = 0.05 * (1 .+ r / 2) .^ LinRange(1, T + 1, T + 1)
    p_F_path_guess = transpose(p_F_path_guess)

    # set path for battery prices
    p_B_init = (Initialprod + sum(RWParams.KR .* hoursofstorage)).^(-params.gammaB)

    for kk in 1:501
        p_B_path_guess[:, kk].= sseqE.p_B.+((p_B_init .- sseqE.p_B) .* expo1[kk])
    end

    # ---------------------------------------------------------------------------- #
    #                             SOLVE TRANSITION PATH                            #
    # ---------------------------------------------------------------------------- #

    difftrans=1
    ll=1

    while difftrans>10^(-2) && ll<=Transiter
        # don't update capital prices

        # get solar shares
        SShare_path .= (regionParams.thetaS ./ p_KR_path_guess_S) .^ params.varrho ./ ((regionParams.thetaS ./ p_KR_path_guess_S) .^
            params.varrho + (regionParams.thetaW ./ p_KR_path_guess_W) .^ params.varrho)
        thetabar_path .= regionParams.thetaS .* SShare_path .+ regionParams.thetaW .* (1 .- SShare_path)
        p_KR_bar_path .= SShare_path .* p_KR_path_guess_S .+ (1 .- SShare_path) .* p_KR_path_guess_W

        update_battery_prices!(KR_path, Qtotal_path_B, KR_path_S, KR_path_W, params, hoursofstorage, RWParams, Initialprod)
        # updates KR_path and Qtotal_path_B

        # set returns on capital
        rP_path[:, 1:end-1] .= (r_path .- 1 .+ params.deltaP) .* PC_path_guess[:, 1:end-1]
        rP_path[:, end] .= rP_path[:, end-1]

        for jj=1:T+1
            pg_path_s[:,:,jj] .= w_path_guess[:, jj] .^ (params.Vs[:,1]' .* ones(params.J, 1)) .* 
                p_E_path_guess[:,jj] .^ (params.Vs[:,2]' .^ (ones(params.J, 1)) + (params.Vs[:,3]' .* ones(params.J, 1))) .*
                (params.kappa + (params.kappa .* p_F_path_guess[:,jj] ./ p_E_path_guess[:,jj]) .^ (1 - params.psi)) .^ (-(params.psi ./ (params.psi-1))* params.Vs[:,3]') .*
                rP_path[:,jj] .^ (params.Vs[:,4]' .* ones(params.J, 1)) ./
                (params.Z .* params.zsector .* params.cdc)
        end

        # power path
        YR_path .= regionParams.thetaS .* KR_path_S + regionParams.thetaW .* KR_path_W

        # ---------------------------------------------------------------------------- #
        #                      SOLVE TRANSITION ELECTRICITY MARKET                     #
        # ---------------------------------------------------------------------------- #

        # US and Europe    

        transition_electricity_US_Europe!(result_price_path, result_Dout_path, result_Yout_path, result_YFout_path,
                            majorregions, Linecounts, RWParams, laboralloc_path,
                            Lsectorpath_guess, params, w_path_guess, rP_path,
                            linconscount, pg_path_s, p_F_path_guess, p_E_path_guess,
                            kappa, KR_path_S, KR_path_W, KF_path, capT, regionParams)
        # updates result_price_path, result_Dout_path, result_Yout_path, result_YFout_path

        # other countries
        transition_electricity_other_countries!(result_price_path, result_Dout_path, result_Yout_path, result_YFout_path,
                                    majorregions, Linecounts, RWParams, laboralloc_path,
                                    Lsectorpath_guess, params, w_path_guess, rP_path,
                                    linconscount, pg_path_s, p_F_path_guess, p_E_path_guess,
                                    kappa, KR_path_S, KR_path_W, KF_path, capT, regionParams)   
        # updates result_price_path, result_Dout_path, result_Yout_path, result_YFout_path

        # places that are off the grid
        transition_electricity_off_grid!(result_price_path, result_Dout_path, result_Yout_path, result_YFout_path, # modified variables
            majorregions, laboralloc_path,
            Y_path, Lsectorpath_guess, params, w_path_guess, rP_path,
            pg_path_s, p_F_path_guess, p_E_path_guess,
            kappa, KR_path_S, KR_path_W, KF_path, capT, regionParams, D_path)

        
        fill_paths!(p_E_path_guess, D_path, Y_path, YF_path, PI_path, 
            params, majorregions, result_price_path, result_Dout_path, result_Yout_path, result_YFout_path, capT)
        # updates p_E_path_guess, D_path, Y_path, YF_path, PI_path

        # out from cap T set prices based on smoothing
        smooth_prices!(p_E_path_guess, D_path, Y_path, YF_path,
            sseqE, expo3, expo4, capT, T)
        # updates p_E_path_guess, D_path, Y_path, YF_path

        # ---------------------------------------------------------------------------- #
        #                        UPDATE TRANSITION LABOUR MARKET                       #
        # ---------------------------------------------------------------------------- #

        Threads.@threads for i in 1:capT
            # get capital vec
            local Ksector = sseqE.Lsector .* (params.Vs[:, 4]' .* ones(params.J, 1)) ./
                (params.Vs[:, 1]' .* ones(params.J, 1)) .* 
                (w_path_guess[:, i] ./ rP_path[:, i])
            KP_path_guess[:, i] .= sum(Ksector, dims=2)
            w_update[:, i], w_real_path[:, i], Incomefactor, PC[:, i]= wage_update_ms(w_path_guess[:, i], p_E_path_guess[:, i], p_E_path_guess[:, i], 
                                                                            p_F_path_guess[:, i], D_path[:, i], Y_path[:, i],
                                                                            rP_path[:, i], KP_path_guess[:, i], PI_path[:, i], 1, params)
            #relexp = Xjdashs ./ (Xjdashs[:, 1] .* ones(1, params.I))
            @views w_path_guess[:, i] .= 0.5 * w_update[:, i] + (1- 0.5) * w_path_guess[:, i]
            @views Lsectorpath_guess[:, :, i] .= Lsectorpath_guess[:, :, i] ./ sum(Lsectorpath_guess[:,:,i], dims = 2) .* params.L
        end

        for kk = (capT+1):(T+1)
            jj = kk - capT
            KP_path_guess[:, kk] .= sseqE.KP_LR .+ (KP_path_guess[:, capT] .- sseqE.KP_LR) .* expo4[jj]
        end

        # ---------------------------------------------------------------------------- #
        #                             UPDATE FOSSIL MARKET                             #
        # ---------------------------------------------------------------------------- #
        diffpF = 1.0

        update_fossil_market!(diffpF, fossilsales_path, p_F_path_guess,
                                laboralloc_path, D_path, params, p_E_path_guess, YF_path, KF_path,
                                p_F_int, regions, T, interp1, g, r_path, fusage_total_path, p_F_update)
        # updates p_F_path_guess, fossilsales_path, diffpF

        # ---------------------------------------------------------------------------- #
        #                          UPDATE RENEWABLE INVESTMENT                         #
        # ---------------------------------------------------------------------------- #

        # out from cap T set prices based on smoothing
        # initial guess for sectoral allocations

        for kk in 1:(T-capT)
            decaymat[:, :, kk] .= (expo5[kk] .* ones(params.J, params.I))
        end
        
        Lsectorpath_guess[:, :, (capT + 1):T] .= sseqE.laboralloc_LR .+ (Lsectorpath_guess[:, :, capT] .- sseqE.laboralloc_LR) .* decaymat
        
        for i = 1:(T - capT)
            kk = capT + i
            jj = i
            w_path_guess[:, kk] .= sseqE.w_LR .+ (w_path_guess[:, capT] .- sseqE.w_LR) .* expo3[jj]
        end

        # set battery prices
        p_B_path_guess .= (Initialprod .+ Qtotal_path_B) .^ (-params.gammaB)   # shift avoids NaN when 0 investment
        p_B_path .= pB_shifter .* hoursofstorage .* p_B_path_guess

        # if market price is above free entry price, increase renewable capital
        # decrease otherwise
        pE_S_FE_path[:, 2:end] .= (p_KR_bar_path[:,1:end-1] .+ p_B_path[:, 1:end-1] .- (p_KR_bar_path[:, 2:end] .+ p_B_path[:, 2:end]) .* (1-params.deltaR) ./ r_path) .*
                    regionParams.costshifter ./ thetabar_path[:, 1:end-1]
        pE_S_FE_path[:, 1] .= pE_S_FE_path[:, 2]
        pE_S_FE_path .= pE_S_FE_path .- st
        Capinvest .= p_E_path_guess .> pE_S_FE_path
        Capdisinvest .= p_E_path_guess .<= pE_S_FE_path

        for i in 1:501
            @views kfac[:, i] .= 1 .+ expo6[i] .* clamp.((0.2 .*(p_E_path_guess .- pE_S_FE_path) ./ pE_S_FE_path), -0.1, 0.1)[:, i]
        end

        KR_path .= KR_path_S .+ KR_path_W
        sseqE.KR_LR .= sseqE.KR_LR_S .+ sseqE.KR_LR_W

        for i = 2:T
            @views KR_path_min[:, i] .= KR_path[:, i - 1] .* (1 - params.deltaR)
        end
        KR_path_min[:, 1] .= KR_path[:, 1]

        # update the capital path
        @views KR_path_update .= KR_path[:, 2:end] .* kfac[:, 2:end]
        KR_path_update .= max.(KR_path_update, KR_path_min)
        @views KR_path[:, 2:end] .= KR_path[:, 2:end] .* (1 - updwk) .+ KR_path_update .* updwk

        # out from cap T set capital based on smoothing
        for i = capT+1:T
            jj = i - capT
            KR_path[:, i] .= sseqE.KR_LR .+ (KR_path[:, capT] .- sseqE.KR_LR) .* expo3[jj]
        end

        # split out into solar / wind shares
        KR_path_S .= SShare_path .* KR_path
        KR_path_W .= (1 .- SShare_path) .* KR_path

        difftrans = maximum(maximum(((kfac[:, 2:capT] .- 1) .* KR_path[:, 2:capT]), dims=1))     #get at substantial adjustments to K

        ll += 1
        println("Transition Difference = ", difftrans)
        println("Loop number = ", ll)

    end

    return (difftrans = difftrans, 
        ll = ll, 
        r_path = r_path, 
        KR_path = KR_path, 
        p_KR_bar_path = p_KR_bar_path, 
        KF_path = KF_path, 
        PC_path_guess = PC_path_guess, 
        fossilsales_path = fossilsales_path, 
        w_path_guess = w_path_guess, 
        YF_path = YF_path, 
        YR_path = YR_path, 
        Y_path = Y_path, 
        p_KR_path_S = p_KR_path_S, 
        p_KR_path_W = p_KR_path_W,
        p_KR_path_guess_W = p_KR_path_guess_W, 
        p_F_path_guess = p_F_path_guess, 
        KP_path_guess = KP_path_guess,
        p_E_path_guess = p_E_path_guess,
        fusage_total_path = fusage_total_path,
        p_B_path_guess = p_B_path_guess
        )

end

end