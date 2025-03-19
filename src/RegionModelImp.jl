module RegionModelImp

export solve_model_imp, data_set_up_imp, solve_model_test

using JuMP, Ipopt
import DataFrames: DataFrame
import Random: Random
import LinearAlgebra: I
import SparseArrays: sparse
import ..DataLoadsFunc: StructRWParams
import ..ParamsFunctions: StructParams

using ..MarketEquilibrium

function data_set_up_imp(kk::Int, majorregions::DataFrame, Linecounts::DataFrame, laboralloc::Matrix, Lsector::Matrix, params,
    wage::Union{Matrix, Vector}, rP::Vector, pg_n_s::Matrix, pE::Union{Vector, Matrix}, kappa::Int, regionParams, KF::Matrix, p_F::Union{Int64, Float64}, 
    linconscount::Int, KR_S::Matrix, KR_W::Matrix, Linecounts_imp::DataFrame, RegionImp)
    local ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    local n = majorregions.n[kk]
    "local l_ind = Linecounts_imp.rowid2[kk]:Linecounts_imp.rowid[kk]
    local gam = RegionImp.Gam_imp[kk]
    local l_n = Linecounts.n[kk]"

    local @views secalloc = laboralloc[ind, :]
    local @views Lshifter = Lsector[ind, :]
    local Kshifter = Lsector[ind, :] .* (params.Vs[:,4]' .* ones(n, 1)) ./
                    (params.Vs[:,1]' .* ones(n, 1)) .*
                    (wage[ind] ./ rP[ind])

    # define data for inequality constraints
    "local linecons = copy(RegionImp.Zmax_imp[l_ind])
    local Gammatrix = hcat(zeros(size(gam, 1)), gam)
    local maxline = min(linconscount, l_n)
    local Gammatrix[maxline:end, :] .= 0
    local stacker = [-Matrix(I, n, n) Matrix(I, n, n)]
    local Gammatrix = Gammatrix * stacker"

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

    local YFmax =KF[ind]
    local LB = [zeros(n); KRshifter]
    local UB = [fill(1000, n); YFmax .+ KRshifter .+ 1] # different from market

    local guess = [KRshifter; KRshifter .+ 0.001] # different from market

    local l_guess = length(guess)
    local mid = l_guess ÷ 2

    return l_guess, LB, UB, guess, power, shifter, KFshifter, KRshifter, n, mid
end


function add_model_variable(model::Model, LB::Vector, l_guess::Int, UB::Vector, guess::Vector)
    local x = Vector{AffExpr}(undef, l_guess)
    for i in eachindex(x)
        local x[i] = AffExpr(0.0)
    end
    return @variable(model, LB[i] <= x[i=1:l_guess] <= UB[i], start=guess[i])
end

function add_model_constraint(model::Model, RegionImp, params::StructParams, kk::Int, mid::Int)
    local x = model[:x]
    local Pvec = @expression(model, x[mid+1:end] .- x[1:mid])
    local sPvec = @expression(model, sum(Pvec))
    local quad_mat = @expression(model, -params.Rweight .* Pvec[2:end]' * RegionImp.B_imp[kk] * Pvec[2:end])
    add_to_expression!(quad_mat[1], sPvec-Pvec[1]^2)
    
    return @constraint(model, ceq, quad_mat[1] ==0)

end
# 418 calls on Market.jl compiler
# lots of type inference

function add_model_objective(model::Model, power::Matrix, shifter::Matrix, KFshifter::Union{Vector, SubArray}, KRshifter::Vector, p_F::Union{Float64, Vector}, params::StructParams)
    local x = model[:x]
    @objective(model, Min, obj(x, power, shifter, KFshifter, KRshifter, p_F, params))
end

# 683 calls on Market.jl compiler
# lots of type inference

function add_model_objective_test(model::Model, power::Matrix, shifter::Matrix, KFshifter::Union{Vector, SubArray}, 
            KRshifter::Vector, p_F::Union{Float64, Vector}, params::StructParams, mid::Int, power2::Float64)
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

function solve_model_imp(kk::Int, l_guess::Int, LB::Vector, UB::Vector, guess::Vector, RegionImp, params::StructParams, power::Matrix, 
    shifter::Matrix, KFshifter::Union{SubArray, Vector}, KRshifter::Vector, p_F::Union{Float64, Vector}, mid::Int)
    local model = Model(Ipopt.Optimizer)
    set_silent(model)
    add_model_variable(model, LB, l_guess, UB, guess)
    add_model_constraint(model, RegionImp, params, kk, mid)
    add_model_objective(model, power, shifter, KFshifter, KRshifter, p_F, params)
    optimize!(model)
    return value.(model[:x])
end

function solve_model_test(kk::Int, l_guess::Int, LB::Vector, UB::Vector, guess::Vector, RegionImp, params::StructParams, power::Matrix, 
    shifter::Matrix, KFshifter::Union{SubArray, Vector}, KRshifter::Vector, p_F::Union{Float64, Vector}, mid::Int, power2::Float64)
    local model = Model(Ipopt.Optimizer)
    set_silent(model)
    add_model_variable(model, LB, l_guess, UB, guess)
    add_model_constraint(model, RegionImp, params, kk, mid)
    add_model_objective_test(model, power, shifter, KFshifter, KRshifter, p_F, params, mid, power2)
    optimize!(model)
    return value.(model[:x])
end


end