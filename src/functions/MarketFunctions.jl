module MarketFunctions

export StructMarketEq, solve_initial_equilibrium, up_SShare_pKRbar_init!

using DataFrames, Statistics, MAT
using ..RegionModel, ..MarketEquilibrium
using JuMP, Ipopt

import ..DataLoadsFunc: StructRWParams
import DrawGammas: StructParams

# ---------------------------------------------------------------------------- #
#                                  Set Up Data                                 #
# ---------------------------------------------------------------------------- #

function calc_subsidyUS(p_E_init::Vector, regions::DataFrame, majorregions::DataFrame)
    p_E_weighted = Vector{Float64}(undef, 2531)
    @. p_E_weighted = p_E_init * regions.employment / regions.employment_region
    pricetable = DataFrame(p_E_weighted = p_E_weighted, region_groups = regions.region_groups)
    tblstats = combine(groupby(pricetable, :region_groups), :p_E_weighted => mean => :mean_p_E_weighted)
    p_E_weighted = tblstats.mean_p_E_weighted .* majorregions.n
    #p_E_weighted_rel = p_E_weighted ./ p_E_weighted[1]

    # set subsidy amount for transition
    subsidy_US = p_E_weighted[1]*0.15
    return subsidy_US
end

function calc_expenditure!(e2_init::Matrix, fusage_ind_init::Vector, 
    fusage_power_init::Vector, Expenditure_init::Vector,
    fossilsales::Matrix, laboralloc::Matrix, D_init::Vector, params::StructParams, p_E_init::Vector, p_F::Float64, 
    YF_init::Vector, regionParams::StructRWParams, wage_init::Vector, rP_init::Vector, KP_init::Vector, YE_init::Vector,
    regions::DataFrame, PI_init::Vector)
    # compute electricty and fossil fuel usage in industry and electricity
    e2_init .= laboralloc .* (D_init .* ones(1, params.I)) .* ((params.Vs[:,2]' .* ones(params.J, 1)) ./ 
    ((params.Vs[:,2]' .* ones(params.J, 1))+(params.Vs[:,3]' .* ones(params.J, 1))))
    fusage_ind_init .= (params.kappa .^ -1) .* sum(e2_init, dims=2) .* (p_E_init ./ p_F) .^ params.psi
    fusage_power_init .= (YF_init ./ regionParams.maxF .^ params.alpha2) .^ (1 / params.alpha1)
    fusage_total_init = sum(fusage_power_init) + sum(fusage_ind_init)

    # compute fossil usage as a share of GDP
    #GDP=sum(wage_init.*params.L+p_E_init.*D_init+rP_init.*KP_init+p_F.*fusage_ind_init)
    #Fossshare_init=sum(p_F.*fusage_total_init)./GDP

    # generate expenditure shares on each
    fossilsales .= p_F .* fusage_total_init .* regions.reserves_share
    Expenditure_init .= wage_init .* params.L + rP_init .* KP_init + fossilsales + 
                        p_E_init .* YE_init - p_F .* fusage_power_init + p_F .* (fusage_power_init + fusage_ind_init) + PI_init
    #fossshare_j = fossilsales ./ Expenditure_init
end

function elec_fuel_expenditure!(laboralloc::Matrix, D_init::Vector, params::StructParams, p_E_init::Vector, p_F::Float64, 
    YF_init::Vector, regionParams::StructRWParams, wage_init::Vector, rP_init::Vector, KP_init::Vector, fossilsales::Matrix, 
    YE_init::Vector, regions::DataFrame, PI_init::Vector)
    e2_init = Matrix{Float64}(undef, 2531, 10)
    fusage_ind_init = Vector{Float64}(undef, 2531)
    fusage_power_init = Vector{Float64}(undef, 2531)
    Expenditure_init = Vector{Float64}(undef, 2531)

    calc_expenditure!(e2_init, fusage_ind_init, fusage_power_init, Expenditure_init, fossilsales, 
    laboralloc, D_init, params, p_E_init, p_F, YF_init, regionParams,
    wage_init, rP_init, KP_init, YE_init, regions, PI_init)

    return fossilsales, Expenditure_init
end

function open_mat_var(G::String)
    # opening variables in mat files
    wf = matopen("$G/w_guess_mat.mat")
    w_guess = read(wf, "w_guess")
    close(wf)

    pEf = matopen("$G/p_E_guessmat.mat") 
    p_E_init = read(pEf, "p_E_init")
    close(pEf)
    p_E_init = vec(p_E_init)

    pof = matopen("$G/Pout_guess_init.mat")
    result_Pout_init = read(pof, "result_Pout_init")
    close(pof)

    lf = matopen("$G/laboralloc_guess.mat")
    laboralloc = read(lf, "laboralloc")
    close(lf)

    PCf = matopen("$G/PC_guess_init.mat")
    PC_guess_init = read(PCf, "PC_guess_init")
    close(PCf)

    dof = matopen("$G/Dout_guess_init.mat")
    result_Dout_init = read(dof, "result_Dout_init")
    close(dof)

    yof = matopen("$G/Yout_guess_init.mat")
    result_Yout_init = read(yof, "result_Yout_init")
    close(yof)

    pff = matopen("$G/p_F_path_guess_saved.mat")
    p_F_path_guess = read(pff, "p_F_path_guess")
    close(pff)

    wf = matopen("$G/wedge_vec.mat")
    wedge = read(wf, "wedge")
    close(wf)

    psf = matopen("$G/priceshifterupdate_vec.mat")
    priceshifterupdate = read(psf, "priceshifterupdate")
    close(psf)

    ff = matopen("$G/fossilsales_guess.mat")
    fossilsales = read(ff, "fossilsales")
    close(ff)

    return w_guess, p_E_init, laboralloc, PC_guess_init, wedge, priceshifterupdate, fossilsales, result_Pout_init, result_Yout_init, p_F_path_guess, result_Dout_init
end

# ---------------------------------------------------------------------------- #
#                           Solve Market Equilibrium                           #
# ---------------------------------------------------------------------------- #

function solve_offgrid!(result_Dout_init::Matrix{Any}, result_Yout_init::Matrix{Any}, result_price_init::Vector{Vector}, # modified variables
                majorregions::DataFrame, KRshifter::Vector{Float64}, YFmax::Vector{Float64}, power::Matrix{Float64}, KFshifter::Vector{Float64}, 
                shifter::Matrix{Float64}, kk::Int64, p_F::Float64, params::StructParams)
    
    @inbounds for jj in 1:majorregions.n[kk]                # set up variables local to the task
        guess = [1; KRshifter[jj]]
        LB = [0; KRshifter[jj]]
        UB = [10^6; YFmax[jj] + KRshifter[jj]]
        l_guess = length(guess)

        x = Vector{Float64}(undef, l_guess)
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, LB[i] <= x[i=1:l_guess] <= UB[i], start=guess[i])
        @constraint(model, c1, x[1] - x[2] == 0) 
        @objective(model, Min, obj2(x, power[jj], shifter[jj], KFshifter[jj], KRshifter[jj], p_F, params))
        optimize!(model)

        # record the output
        P_out1 = value.(x) 

        price = Price_Solve(P_out1, shifter[jj], 1, params)[1]
        result_Dout_init[kk][jj] = P_out1[1]
        result_Yout_init[kk][jj] = P_out1[2]
        result_price_init[kk][1][1, jj] = price
    end
end

function optimize_offgrid!(result_Dout_init::Matrix{Any}, result_Yout_init::Matrix{Any}, result_price_init::Vector{Vector}, # modified variables
    majorregions::DataFrame, laboralloc::Matrix{Float64}, Lsector::Matrix{Float64}, params::StructParams, wage_init::Vector{Float64}, rP_init::Vector{Float64}, 
    result_Pout_init::Matrix{Any}, pg_init_s::Matrix{Float64}, pE_market_init::Vector{Float64}, kappa::Float64,
    regionParams::StructRWParams, KR_init_S::Matrix{Float64}, KR_init_W::Matrix{Float64}, p_F::Float64)

    kk = params.N 

    KRshifter, YFmax, power, KFshifter, shifter = second_loop(kk, majorregions, laboralloc, Lsector, params, wage_init, rP_init,
                                                                            result_Pout_init, pg_init_s, pE_market_init, kappa, 
                                                                            regionParams, KR_init_S, KR_init_W, p_F) #  311.800 μs (162 allocations: 395.20 KiB)
    solve_offgrid!(result_Dout_init, result_Yout_init, result_price_init,
        majorregions, KRshifter, YFmax, power, KFshifter, shifter, kk, p_F, params) # 1.245 s (1591029 allocations: 58.87 MiB)
    #   1.572967 seconds (1.62 M allocations: 61.364 MiB, 1.97% compilation time)
    
end

function fill_init!(p_E_init::Vector{Float64}, D_init::Vector{Float64}, YE_init::Vector{Float64}, PI_init::Vector{Float64}, # modified variables
            N::Int64, majorregions::DataFrame, result_price_init::Vector{Vector}, result_Dout_init::Matrix{Any}, 
            result_Yout_init::Matrix{Any}, L::Matrix{Float64})


    @inbounds for kk=1:(N - 1)
        ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
        p_E_init[ind] .= result_price_init[kk]
        D_init[ind].= result_Dout_init[kk]
        YE_init[ind] .= result_Yout_init[kk]
        PI_init[ind, 1] .= (sum(result_price_init[kk] .*(result_Dout_init[kk] - result_Yout_init[kk]))) .*
                            (L[ind, 1]) ./ 
                            (sum(L[ind, 1]))
    end
        
    kk = N
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    result_price_init[kk][1] .= clamp.(result_price_init[kk][1], 0.001, 0.3)	# put bounds on prices
    matrix_values = result_price_init[kk][1]
    p_E_init[ind] .= vec(matrix_values)
    D_init[ind].= vec(result_Dout_init[kk])
    YE_init[ind] .= vec(result_Yout_init[kk])
    PI_init[ind, 1] .= (sum(matrix_values.*(result_Dout_init[kk] .- result_Yout_init[kk]))) .*
                        (L[ind, 1]) ./ (sum(L[ind, 1]))
end

function market_setup!(rP_init::Vector, pg_init_s::Matrix, pE_market_init::Vector, # modified variables
    wage_init::Vector, params, p_E_init::Vector, p_F::Float64, R_LR::Float64, PC_guess_init::Matrix)
    
    # Set initial capital returns
    coef = R_LR - 1 + params.deltaP
    rP_init .= coef .* PC_guess_init #   623.256 ns (2 allocations: 64 bytes)

    # Calculate final good priceshifterupdate
    pg_init_s .= (wage_init .^ (params.Vs[:, 1]' .* ones(params.J, 1))) .* 
                (p_E_init .^ ((params.Vs[:, 2]' .* ones(params.J, 1)) .+ (params.Vs[:, 3]' .* ones(params.J, 1)))) .* 
                ((params.kappa .+ (params.kappa .* p_F ./ p_E_init) .^ (1 - params.psi)) .^ 
                (-(params.psi / (params.psi - 1)) .* params.Vs[:, 3]')) .* 
                (rP_init .^ (params.Vs[:, 4]' .* ones(params.J, 1))) ./ 
                (params.Z .* params.zsector .* params.cdc) # 1.683 ms (75 allocations: 82.73 KiB)

    # Update pE_market_init
    pE_market_init .= p_E_init

end

function optimize_region!(result_price_init::Vector, result_Dout_init::Matrix{Any}, result_Yout_init::Matrix{Any}, Lossfac_init::Matrix, # modified variables
	majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc::Matrix, Lsector::Matrix, params::StructParams,
	wage_init::Vector, rP_init::Vector, linconscount::Int, pg_init_s::Matrix, pE_market_init::Vector, kappa::Float64,
	p_F::Float64, regionParams::StructRWParams, KR_init_S::Matrix, KR_init_W::Matrix)

	Threads.@threads :static for kk in 1:(params.N - 1)
		# the local call in this section localizes the memory to each thread to reduce crossing data 
		# set up optimization problem for region kk

		local l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid = data_set_up(kk, majorregions, Linecounts, RWParams, laboralloc,
											Lsector, params, wage_init, rP_init, pg_init_s, pE_market_init, kappa, regionParams, 
											regionParams.KF, p_F, linconscount, KR_init_S, KR_init_W, "market")

		# solve the model for region kk (see RegionModel.jl)
        local P_out = solve_model(kk, l_guess, LB, UB, guess, regionParams, params, power, shifter, KFshifter, KRshifter, p_F, mid)

		result_price_init[kk] .= Price_Solve(P_out, shifter, n, params) #.MarketEquilibrium.jl
		@views result_Dout_init[kk] .= P_out[1:mid]
		@views result_Yout_init[kk] .= P_out[1+mid:end]
		local Pvec = P_out[1+mid:end] .- P_out[1:mid]
		local Losses = Pvec[2:end]' * regionParams.B[kk] * Pvec[2:end] .* params.Rweight
		@views Lossfac_init[kk] = Losses ./ sum(result_Yout_init[kk])

	end
end

function optimize_region_test!(result_price_init::Vector, result_Dout_init::Matrix{Any}, result_Yout_init::Matrix{Any}, Lossfac_init::Matrix,
	majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc::Matrix, Lsector::Matrix, params::StructParams,
	wage_init::Vector, rP_init::Vector, linconscount::Int, pg_init_s::Matrix, pE_market_init::Vector, kappa::Float64,
	p_F::Float64, regionParams::StructRWParams, KR_init_S::Matrix, KR_init_W::Matrix, power2::Float64)

	Threads.@threads :static for kk in 1:(params.N - 1)
		# the local call in this section localizes the memory to each thread to reduce crossing data 
		# set up optimization problem for region kk
		local l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid = data_set_up(kk, majorregions, Linecounts, RWParams, laboralloc,
											Lsector, params, wage_init, rP_init, pg_init_s, pE_market_init, kappa, regionParams, 
											regionParams.KF, p_F, linconscount, KR_init_S, KR_init_W, "market")

		# solve the model for region kk (see RegionModel.jl)
		local P_out = solve_model_test(kk, l_guess, LB, UB, guess, regionParams, params, power, shifter, KFshifter, KRshifter, p_F, mid, power2)

		result_price_init[kk].=Price_Solve(P_out, shifter, n, params) #.MarketEquilibrium.jl
		#local mid = length(P_out) ÷ 2
		@views result_Dout_init[kk] .= P_out[1:mid]
		@views result_Yout_init[kk] .= P_out[1+mid:end]
		local Pvec = P_out[1+mid:end] .- P_out[1:mid]
		local Losses = Pvec[2:end]' * regionParams.B[kk] * Pvec[2:end] .* params.Rweight
		@views Lossfac_init[kk] = Losses ./ sum(result_Yout_init[kk])

	end
end

function second_loop(kk::Int, majorregions::DataFrame, laboralloc::Matrix, Lsector::Matrix, params::StructParams, 
            wage_init::Vector, rP_init::Vector, result_Pout_init::Matrix, 
            pg_init_s::Matrix, pE_market_init::Vector, kappa::Float64, 
            regionParams::StructRWParams, KR_init_S::Matrix, KR_init_W::Matrix, p_F::Float64)

    YFmax = Vector{Float64}(undef, 332)
    KRshifter = Vector{Float64}(undef, 332)
    #guess = Matrix{Float64}(undef, 348, 1)
    power = Matrix{Float64}(undef, 332, 10)
    KFshifter = Vector{Float64}(undef, 332)
    shifter = Matrix{Float64}(undef, 332, 10)
    prices = Matrix{Float64}(undef, 332, 10)
    pg_s = Matrix{Float64}(undef, 332, 10)
    Lshifter = Matrix{Float64}(undef, 332, 10)
    secalloc = Matrix{Float64}(undef, 332, 10)
    Kshifter = Matrix{Float64}(undef, 332, 10)

    # set up optimization problem for region kk
    nr::Int = majorregions.rowid[kk] - majorregions.rowid2[kk] + 1
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]

    @views secalloc .= laboralloc[ind, :]
    @views Lshifter .= Lsector[ind, :]
    @views Kshifter .= Lshifter .* 
        (params.Vs[:, 4]' .* ones(majorregions.n[kk], 1)) ./ 
        (params.Vs[:, 1]' .* ones(majorregions.n[kk], 1)) .* 
        wage_init[ind] ./ 
        rP_init[ind]

    #Ltotal = sum(Lshifter, dims = 2)
    Jlength::Int = majorregions.n[kk]
    #guess .= result_Pout_init[kk]

    # define shifters for objective function
    @views pg_s .= pg_init_s[ind,:]
    @views prices .= (pE_market_init[ind] .* ones(1, params.I))
    @views power .= (params.Vs[:, 2]' .* ones(Jlength, 1))+(params.Vs[:,3]' .* ones(Jlength, 1))
    shifter .= pg_s .* 
            ((kappa .+ (prices ./ (kappa .* p_F))) .^ (params.psi - 1)) .^ 
            ((params.psi / (params.psi - 1)) .* (params.Vs[:, 3]' .* ones(majorregions.n[kk], 1))) .* 
            (1 .+ (params.Vs[:, 3]' .* ones(majorregions.n[kk], 1)) ./ (params.Vs[:, 2]' .* ones(majorregions.n[kk], 1))) .^ 
            (-(params.Vs[:, 2]' .* ones(majorregions.n[kk], 1)) .- (params.Vs[:, 2]' .* ones(majorregions.n[kk], 1))) .* 
            params.Z[ind] .* params.zsector[ind, :] .* 
            (Lshifter .^ (params.Vs[:, 1]' .* ones(majorregions.n[kk], 1))) .* 
            (Kshifter .^ (params.Vs[:, 4]' .* ones(majorregions.n[kk], 1)))

    shifter .= shifter .* secalloc .^ power

    @views KRshifter .= regionParams.thetaS[ind] .* KR_init_S[ind] +
                        regionParams.thetaW[ind] .* KR_init_W[ind]
    @views KFshifter .= regionParams.KF[ind]

    # define bounds
    @views YFmax .= regionParams.maxF[ind]

    # define shifter
    #wedge_vec = wedge[ind]
    #return KRshifter, YFmax, guess, power, KFshifter, shifter
    return KRshifter, YFmax, power, KFshifter, shifter

end

function solve_KP_init!(KP_init::Vector, Lsector::Matrix, params::StructParams, wage_init::Vector, rP_init::Vector)
    # get capital vec
    Ksector = Lsector .* 
            (params.Vs[:, 4]' .* ones(params.J, 1)) ./
            (params.Vs[:, 1]' .* ones(params.J, 1)) .* 
            (wage_init ./ rP_init)
    KP_init .= sum(Ksector, dims = 2)
end

function up_wages_productivity!(w_guess::Matrix{Float64}, Z::Matrix{Float64}, laboralloc::Matrix{Float64}, Lsector::Matrix{Float64}, # modified variables
            w_update::Matrix{Float64}, updw_w::Float64, wage_init::Vector{Float64}, L::Matrix{Float64}, Xjdashs::Matrix{Float64})

    relexp = Matrix{Float64}(undef, 2531, 10)
    relab = Matrix{Float64}(undef, 2531, 10)

    w_guess .= 0.1 .* w_update .+ (1 - 0.1) .* w_guess
    Z .= Z .* clamp.(1 .+ updw_w .* (wage_init .- w_guess) ./ wage_init, 0.95, 1.05)

    # update relative labor allocations
    laboralloc .= Lsector ./ L

    relab .= laboralloc ./ sum(laboralloc, dims=2)
    relexp .= Xjdashs ./ sum(Xjdashs, dims = 2)

    Lsector .= Lsector .* clamp.(1 .+ 0.05 .* (relexp .- relab), 0.9, 1.1)
    Lsector .= Lsector ./ sum(Lsector, dims=2) .* L
end

# ---------------------------------------------------------------------------- #
#                              Exported Functions                              #
# ---------------------------------------------------------------------------- #

function up_SShare_pKRbar_init!(SShare_init::Matrix{Float64}, p_KR_bar_init::Matrix{Float64}, # modified variables
        thetaS::Matrix{Float64}, p_KR_init_S::Float64, varrho::Float64, thetaW::Matrix{Float64}, p_KR_init_W::Float64)
    
    @inbounds for i = 1:2531
        SShare_init[i] = (thetaS[i] / p_KR_init_S) ^ varrho / ((thetaS[i] / p_KR_init_S) ^ varrho + (thetaW[i] / p_KR_init_W) ^ varrho)
        p_KR_bar_init[i] = SShare_init[i] * p_KR_init_S + (1 - SShare_init[i]) * p_KR_init_W
    end
end


mutable struct StructMarketEq
    Lsector::Matrix{Float64}
    result_price_init::Vector{Any}  # Likely Vector{Vector{Float64}} or Vector{Matrix{Float64}} depending on your data structure
    pg_init_s::Matrix{Float64}
    pE_market_init::Vector{Float64}
    Lossfac_init::Matrix{Float64}
    D_init::Vector{Float64}
    YE_init::Vector{Float64}
    diffend::Float64
    w_guess::Matrix{Float64}
    p_E_init::Vector{Float64}
    result_Pout_init::Array{Matrix{Float64}, 2}
    laboralloc::Matrix{Float64}
    PC_guess_init::Matrix{Float64}
    result_Dout_init::Array{Matrix{Float64}, 2}
    result_Yout_init::Array{Matrix{Float64}, 2}
    p_F_path_guess::Matrix{Float64}
    wedge::Matrix{Float64}
    priceshifterupdate::Matrix{Float64}
    fossilsales::Matrix{Float64}
    #P_out::Vector{Float64}
    Expenditure_init::Vector{Float64}
    rP_init::Vector{Float64}
    PI_init::Vector{Float64}
    KP_init::Vector{Float64}
    W_Real::Vector{Float64}
    PC_init::Vector{Float64}
    YF_init::Vector{Float64}
    p_F::Float64
    subsidy_US::Float64
end

function solve_initial_equilibrium(params::StructParams, wage_init::Vector{Float64}, majorregions::DataFrame,
    regionParams::StructRWParams, KR_init_S::Matrix{Float64}, KR_init_W::Matrix{Float64}, R_LR::Float64, sectoralempshares::Matrix{Union{Float64, Missing}},
    Linecounts::DataFrame, kappa::Float64, regions::DataFrame, linconscount::Int, updw_w::Float64, upw_z::Float64, RWParams::StructRWParams, G::String)

    # allocations for variables
    sizes = [727, 755, 30, 53, 13, 15, 46, 11, 26, 78, 125, 320, 332]
    result_price_init = [Vector{Float64}(undef, size) for size in sizes[1:end-1]]
    last_element = [reshape(Vector{Float64}(undef, sizes[end]), 1, :)]
    result_price_init = [result_price_init..., last_element]


    rP_init = Vector{Float64}(undef, 2531)
    pg_init_s = Matrix{Float64}(undef, 2531, 10)
    pE_market_init = Vector{Float64}(undef, 2531)
    Lossfac_init = Matrix{Float64}(undef, 1, 12)
    D_init = Vector{Float64}(undef, 2531)
    YE_init = Vector{Float64}(undef, 2531)
    FUtilization = Vector{Float64}(undef, 2531)	
    Expenditure_init = Vector{Float64}(undef, 2531)
    Lsector = Matrix{Float64}(undef, 2531, 10)
    p_F_path_guess = Matrix{Float64}(undef, 1, 501)
    W_Real = Vector{Float64}(undef, 2531)
    PC_init = Vector{Float64}(undef, 2531)

    YF_init = Vector{Float64}(undef, 2531)
    PI_init = Vector{Float64}(undef, 2531)
    KP_init = Vector{Float64}(undef, 2531)

    # load variable guesses
    p_F = 0.05 # initial fossil sales guess
    subsidy_US = 0.0

    w_guess, p_E_init, laboralloc, PC_guess_init, wedge, priceshifterupdate, fossilsales, result_Pout_init, result_Yout_init, p_F_path_guess, result_Dout_init = open_mat_var(G)
    Lsector .= laboralloc .* params.L
    
    diffend = 1

    while diffend>10^(-2)

        # optimization for each region kk
        market_setup!(rP_init, pg_init_s, pE_market_init, wage_init, params, p_E_init, p_F, R_LR, PC_guess_init) #   1.683 ms (13 allocations: 80.14 KiB)

        optimize_region!(result_price_init, result_Dout_init, result_Yout_init, Lossfac_init, majorregions, Linecounts, RWParams, laboralloc,
                            Lsector, params, wage_init, rP_init, linconscount, pg_init_s, pE_market_init, kappa, p_F, regionParams, KR_init_S,
                            KR_init_W); #   17.706 s (3528432 allocations: 1.07 GiB)

        # solve market equilibrium

        optimize_offgrid!(result_Dout_init, result_Yout_init, result_price_init, # modified variables
            majorregions, laboralloc, Lsector, params, wage_init, rP_init, result_Pout_init, pg_init_s, pE_market_init, kappa,
            regionParams, KR_init_S, KR_init_W, p_F); #    1.386 s (1591190 allocations: 59.26 MiB)

        fill_init!(p_E_init, D_init, YE_init, PI_init, # modified variables
            params.N, majorregions, result_price_init, result_Dout_init, result_Yout_init, params.L); #   22.900 μs (324 allocations: 50.86 KiB)

        # get capital vec
        solve_KP_init!(KP_init, Lsector, params, wage_init, rP_init);
        
        # solve wages and calculate productivity
        w_update, WR, Incomefactor, PC, Xjdashs = wage_update_ms(w_guess, p_E_init, p_E_init, p_F, D_init, YE_init, rP_init, KP_init, PI_init, fossilsales, params);

        W_Real .= WR
        PC_init .= PC

        up_wages_productivity!(w_guess, params.Z, laboralloc, Lsector, w_update, updw_w, wage_init, params.L, Xjdashs); #   107.000 μs (22 allocations: 455.41 KiB)

        diffend = maximum(abs.(wage_init .- w_guess) ./ wage_init)

        # calibrate sectoral params.Z to move sectoral emp closer
        params.zsector[:, :] .= params.zsector[:, :] .* clamp.(1 .+ upw_z .* (sectoralempshares[:, 2:end] .- laboralloc[:, :]), 0.99, 1.01); #   113.900 μs (22 allocations: 618.80 KiB)

        # Update consumption price guess
        PC_guess_init .= 0.2 .* PC_init .+ (1 - 0.2) .* PC_guess_init #  1.690 μs (6 allocations: 224 bytes)

        # get fossil fuel usage in the initial steady state
        @. YF_init = YE_init - regionParams.thetaS * KR_init_S - regionParams.thetaW * KR_init_W #   3.067 μs (8 allocations: 288 bytes)
        FUtilization .= YF_init ./ regionParams.maxF #  1.360 μs (2 allocations: 64 bytes)

        # calibrate efficiency wedges costs to match initial power prices across countries
        # set subsidy amount for transition
        subsidy_US = calc_subsidyUS(p_E_init, regions, majorregions)  # 108.700 μs (349 allocations: 109.13 KiB)

        fossilsales, Expenditure_init = elec_fuel_expenditure!(laboralloc, D_init, params, p_E_init, p_F, YF_init, regionParams,
                                                                wage_init, rP_init, KP_init, fossilsales, YE_init, regions, PI_init);

        println("diffend = ", diffend)
    end

    return StructMarketEq(
        Lsector,
        result_price_init,
        pg_init_s,
        pE_market_init,
        Lossfac_init,
        D_init,
        YE_init,
        diffend,
        w_guess,
        p_E_init,
        result_Pout_init,
        laboralloc,
        PC_guess_init,
        result_Dout_init,
        result_Yout_init,
        p_F_path_guess,
        wedge,
        priceshifterupdate,
        fossilsales,
        #P_out,
        Expenditure_init,
        rP_init,
        PI_init,
        KP_init,
        W_Real,
        PC_init,
        YF_init,
        p_F,
        subsidy_US
    )
end

# ---------------------------------------------------------------------------- #
#                                   Addendum                                   #
# ---------------------------------------------------------------------------- #
"""n_regions = params.N - 1
tasks = Vector{Task}(undef, n_regions)

# Spawn a task for each region kk.
for kk in 1:n_regions
    tasks[kk] = Threads.@spawn begin
        # Localize all data for region kk.
        l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid =
            data_set_up(kk, majorregions, Linecounts, RWParams, laboralloc,
                        Lsector, params, wage_init, rP_init, pg_init_s, pE_market_init,
                        kappa, regionParams, regionParams.KF, p_F, linconscount,
                        KR_init_S, KR_init_W, "market")

        # Solve the model for region kk.
        P_out = solve_model(kk, l_guess, LB, UB, guess, regionParams, params,
                            power, shifter, KFshifter, KRshifter, p_F, mid)

        local_price = Price_Solve(P_out, shifter, n, params)
        local_Dout = P_out[1:mid]
        local_Yout = P_out[1+mid:end]
        Pvec = local_Yout .- local_Dout
        Losses = Pvec[2:end]' * regionParams.B[kk] * Pvec[2:end] .* params.Rweight
        local_Lossfac = Losses ./ sum(local_Yout)

        return (local_price = local_price,
                local_Dout  = local_Dout,
                local_Yout  = local_Yout,
                local_Lossfac = local_Lossfac)
    end
end

for kk in 1:n_regions
    result = fetch(tasks[kk])
    @views result_price_init[kk] .= result.local_price
    @views result_Dout_init[kk] .= result.local_Dout
    @views result_Yout_init[kk] .= result.local_Yout
    @views Lossfac_init[kk] = result.local_Lossfac
end"""


"""for kk in 1:(params.N - 1)
    # set up optimization problem for region kk

    l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid = data_set_up(kk, majorregions, Linecounts, RWParams, laboralloc,
                                        Lsector, params, wage_init, rP_init, pg_init_s, pE_market_init, kappa, regionParams, 
                                        regionParams.KF, p_F, linconscount, KR_init_S, KR_init_W, "market")

    # solve the model for region kk (see RegionModel.jl)
    P_out = solve_model(kk, l_guess, LB, UB, guess, regionParams, params, power, shifter, KFshifter, KRshifter, p_F, mid)

    result_price_init[kk].=Price_Solve(P_out, shifter, n, params) #.MarketEquilibrium.jl
    @views result_Dout_init[kk] .= P_out[1:mid]
    @views result_Yout_init[kk] .= P_out[1+mid:end]
    Pvec = P_out[1+mid:end] .- P_out[1:mid]
    Losses = Pvec[2:end]' * regionParams.B[kk] * Pvec[2:end] .* params.Rweight
    @views Lossfac_init[kk] = Losses ./ sum(result_Yout_init[kk])
end"""

# ----------------------------- Unused Variables ----------------------------- #
#diffsec = maximum(abs.(sectoralempshares[:, 2:end] .- laboralloc[:, 1:end]), dims=1)
#diff_w2 = maximum(abs.(w_update .- w_guess) ./ w_update)
#mean(FUtilization[1:majorregions[1, :n]])
#P_out = Vector{Float64}(undef, 2)
#power2 = 1 / params.alpha1



end