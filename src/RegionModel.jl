module RegionModel

export solve_model, data_set_up, solve_model_test, data_set_up_exog

using JuMP, Ipopt
import DataFrames: DataFrame
import Random: Random
import LinearAlgebra: I
import SparseArrays: sparse, SparseMatrixCSC
import ..DataLoadsFunc: StructRWParams
import ..ParamsFunctions: StructParams

using ..MarketEquilibrium

function data_set_up(kk::Int, majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc::Matrix, Lsector::Matrix, params::StructParams,
    wage::Union{Matrix, Vector}, rP::Union{Matrix, Vector}, pg_n_s::Matrix, pE::Union{Vector, Matrix}, kappa::Float64, regionParams, KF::Matrix, p_F::Union{Int64, Float64}, 
    linconscount::Int, KR_S::Matrix, KR_W::Matrix, method::String)
    local ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    local n = majorregions.n[kk]
    local l_ind = Linecounts.rowid2[kk]:Linecounts.rowid[kk]
    local gam = RWParams.Gam[kk]
    local l_n = Linecounts.n[kk]

    local @views secalloc = laboralloc[ind, :]
    local @views Lshifter = Lsector[ind, :]
    local Kshifter = Lsector[ind, :] .* (params.Vs[:,4]' .* ones(n, 1)) ./
                    (params.Vs[:,1]' .* ones(n, 1)) .*
                    (wage[ind] ./ rP[ind])
    #Ltotal=sum(Lshifter,dims=2)
    #Jlength= n
    # all of these are varying correctly

    # define data for inequality constraints
    local linecons = copy(RWParams.Zmax[l_ind])
    local Gammatrix = hcat(zeros(size(gam, 1)), gam)

    if linconscount < l_n
        Random.seed!(1)  
        randvec = rand(l_n)  
        randvec = randvec .> (linconscount / l_n)  
        Gammatrix[findall(randvec), :] .= 0 
    end   

    local stacker = [-Matrix(I, n, n) Matrix(I, n, n)]
    local Gammatrix = sparse(Gammatrix * stacker)
    local Gammatrix = [Gammatrix; -Gammatrix]
    local linecons = [linecons; linecons]

    # define shifters for objective function
    local @views pg_s = pg_n_s[ind, :]
    local @views prices = (pE[ind] .* ones(1, params.I))
    local @views power = (params.Vs[:,2]' .* ones(n, 1)) .+ (params.Vs[:,3]' .* ones(n, 1))


    local shifter = pg_s .* (kappa .+ (prices ./ (kappa .* p_F)).^(params.psi - 1)) .^ ((params.psi / (params.psi - 1)) .* (params.Vs[:, 3]'.* ones(n, 1))) .* 
                    (1 .+ (params.Vs[:,3]'.* ones(n, 1)) ./ (params.Vs[:,2]'.* ones(n, 1))) .^ (-(params.Vs[:,2]'.* ones(n, 1)) - 
                    (params.Vs[:,2]'.* ones(n, 1))) .* 
                    params.Z[ind] .* 
                    params.zsector[ind, :] .*
                    Lshifter .^ (params.Vs[:,1]'.* ones(n, 1)) .* 
                    Kshifter .^ (params.Vs[:,4]'.* ones(n, 1))
    local shifter=shifter.*secalloc.^power

    local @views KRshifter = @. regionParams.thetaS[ind] * KR_S[ind] + 
                            regionParams.thetaW[ind] * KR_W[ind]
    local @views KFshifter=KF[ind]

    if method == "market"
        #println("using method market to set YFmax, LB, UB, guess")

        # define bounds
        local YFmax=regionParams.maxF[ind]
        local LB = [zeros(n); KRshifter]
        local UB = [fill(1000, n); YFmax + KRshifter]

        local guess = [KRshifter .- 0.001; KRshifter]
    elseif method == "steadystate"
        #println("using method steadystate to set YFmax, LB, UB, guess")
        local YFmax =KF[ind]
        local LB = [zeros(n); KRshifter]
        local UB = [fill(1000, n); YFmax .+ KRshifter .+ 1] # different from market

        local guess = [KRshifter; KRshifter .+ 0.001] # different from market

    elseif method == "steadystate_imp"
        local YFmax = KF[ind]
        local LB = [zeros(n); KRshifter]
        local UB = [fill(10^6, n); YFmax .+ KRshifter]

        local guess = [result_Dout_LR]
    end

    local l_guess = length(guess)
    local mid = l_guess ÷ 2

    return l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid, Gammatrix, linecons
end
# 1842 calls on Market.jl compiler
# lots of type inference


function data_set_up_exog(kk::Int, majorregions::DataFrame, Linecounts::DataFrame, RWParams::StructRWParams, laboralloc::Matrix, Lsector::Matrix, params,
    wage::Union{Matrix, Vector}, rP::Vector, pg_n_s::Matrix, pE::Union{Vector, Matrix}, kappa::Int, regionParams, KF::Matrix, p_F::Union{Int64, Float64}, 
    linconscount::Int, KR_S::Matrix, KR_W::Matrix, result_Dout_LR, result_Yout_LR)
    local ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    local n = majorregions.n[kk]
    local l_ind = Linecounts.rowid2[kk]:Linecounts.rowid[kk]
    local gam_kk = RWParams.Gam[kk]
    local l_n = Linecounts.n[kk]

    local @views secalloc = laboralloc[ind, :]
    local @views Lshifter = Lsector[ind, :]
    local Kshifter = Lsector[ind, :] .* (params.Vs[:,4]' .* ones(n, 1)) ./
                    (params.Vs[:,1]' .* ones(n, 1)) .*
                    (wage[ind] ./ rP[ind])
    #Ltotal=sum(Lshifter,dims=2)
    #Jlength= n
    # all of these are varying correctly

    # define data for inequality constraints
    local linecons = RWParams.Zmax[l_ind]
    local Gammatrix = hcat(zeros(n_constraints, 1), gam_kk)

    if linconscount < l_n
        Random.seed!(1)  
        randvec = rand(l_n)  
        mask = randvec .> (linconscount / l_n)  
        Gammatrix[mask, :] .= 0 
    end   

    local stacker = hcat(-I(n), I(n))
    local Gammatrix = sparse(Gammatrix * stacker)
    local Gammatrix = vcat(Gammatrix, -Gammatrix)
    local linecons = vcat(linecons, linecons)

    # define shifters for objective function
    local @views pg_s = pg_n_s[ind, :]
    local @views prices = (pE[ind] .* ones(1, params.I))
    local @views power = (params.Vs[:,2]' .* ones(n, 1)) .+ (params.Vs[:,3]' .* ones(n, 1))


    local shifter = pg_s .* (kappa .+ (prices ./ (kappa .* p_F)).^(params.psi - 1)) .^ ((params.psi / (params.psi - 1)) .* (params.Vs[:, 3]'.* ones(n, 1))) .* 
                    (1 .+ (params.Vs[:,3]'.* ones(n, 1)) ./ (params.Vs[:,2]'.* ones(n, 1))) .^ (-(params.Vs[:,2]'.* ones(n, 1)) - 
                    (params.Vs[:,2]'.* ones(n, 1))) .* 
                    params.Z[ind] .* 
                    params.zsector[ind, :] .*
                    Lshifter .^ (params.Vs[:,1]'.* ones(n, 1)) .* 
                    Kshifter .^ (params.Vs[:,4]'.* ones(n, 1))
    local shifter = shifter.*secalloc.^power

    local @views KRshifter = @. regionParams.thetaS[ind] * KR_S[ind] + 
                            regionParams.thetaW[ind] * KR_W[ind]
    local @views KFshifter=KF[ind]

    local YFmax = KF[ind]
    local LB = [zeros(n); KRshifter]
    local UB = [fill(10^6, n); YFmax .+ KRshifter]

    local guess = [result_Dout_LR[kk]; result_Yout_LR[kk]]


    local l_guess = length(guess)
    local mid = l_guess ÷ 2

    return l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid, Gammatrix, linecons
end

function add_model_variable(model::Model, LB::Vector, l_guess::Int, UB::Vector, guess::Union{Vector, Matrix})
    local x = Vector{AffExpr}(undef, l_guess)
    for i in eachindex(x)
        local x[i] = AffExpr(0.0)
    end
    return @variable(model, LB[i] <= x[i=1:l_guess] <= UB[i], start=guess[i])
end

function add_model_constraint(model::Model, regionParams::StructRWParams, params::StructParams, kk::Int, mid::Int)
    local x = model[:x]
    local Pvec = @expression(model, x[mid+1:end] .- x[1:mid])
    local sPvec = @expression(model, sum(Pvec))
    local quad_mat = @expression(model, -params.Rweight .* Pvec[2:end]' * regionParams.B[kk] * Pvec[2:end])
    add_to_expression!(quad_mat[1], sPvec-Pvec[1]^2)
    
    return @constraint(model, ceq, quad_mat[1] == 0)

end
# 418 calls on Market.jl compiler
# lots of type inference

function add_model_objective(model::Model, power::Matrix, shifter::Matrix, KFshifter::Union{Vector, SubArray}, KRshifter::Vector, p_F::Union{Float64, Vector, Int}, params::StructParams)
    local x = model[:x]
    @objective(model, Min, obj(x, power, shifter, KFshifter, KRshifter, p_F, params))
end

# 683 calls on Market.jl compiler
# lots of type inference

function add_model_objective_test(model::Model, power::Matrix, shifter::Matrix, KFshifter::Union{Vector, SubArray}, 
            KRshifter::Vector, p_F::Union{Float64, Vector, Int}, params::StructParams, mid::Int, power2::Float64)
    local x = model[:x]
    local Dsec = @expression(model, x[1:mid] .* ones(1, params.I))
    local Yvec = @expression(model, x[1+mid:end])
    local value1 = @expression(model, sum((-(Dsec .^ power) .* shifter), dims=2))
    local svalue1 = @expression(model, sum(value1))
    local value2 = @expression(model, p_F .* ((Yvec .- KRshifter) ./ KFshifter .^ params.alpha2) .^ power2)
    local svalue2 = @expression(model, sum(value2))
    local svalue1 += svalue2

    return @objective(model, Min, svalue1)
    
end

function add_gammatrix_constraint(model::Model, Gammatrix::SparseMatrixCSC, linecons::Vector{Float64})
    local x = model[:x]
    return @constraint(model, lingamcon, Gammatrix * x <= linecons)
end

function solve_model(kk::Int, l_guess::Int, LB::Vector, UB::Vector, guess::Union{Vector, Matrix}, regionParams::StructRWParams, params::StructParams, power::Matrix, 
    shifter::Matrix, KFshifter::Union{SubArray, Vector}, KRshifter::Vector, p_F::Union{Float64, Vector, Int}, mid::Int, Gammatrix::SparseMatrixCSC, linecons::Vector{Float64})
    local model = Model(Ipopt.Optimizer)
    set_silent(model)
    add_model_variable(model, LB, l_guess, UB, guess)
    add_model_constraint(model, regionParams, params, kk, mid)
    add_gammatrix_constraint(model, Gammatrix, linecons)
    add_model_objective(model, power, shifter, KFshifter, KRshifter, p_F, params)
    optimize!(model)
    return value.(model[:x])
end



end