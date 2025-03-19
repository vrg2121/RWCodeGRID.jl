module TransitionSub

# load functions
using ..DataAdjustments, ..TransitionFunctions, ..MarketEquilibrium

# load packages
using Ipopt, JuMP, Interpolations
import Random: Random
import Plots: plot, plot!
import DataFrames: DataFrame
import MAT: matwrite
import SparseArrays: sparse

# import parameters, data and variables
import ..Params: params, regions, majorregions, T, Linecounts, linconscount, kappa, curtailmentswitch, decayp, hoursofstorage, pB_shifter, g
import ..ParameterizeRun: Transiter, Initialprod, st
import ..DataLoads: RWParams, regionParams, GsupplyCurves, R_LR, wage_init, KR_init_S, KR_init_W
import ..Market: p_KR_bar_init, laboralloc_init, p_KR_init_S, p_KR_init_W, p_F_int, mrkteq, subsidy_US
import ..SteadyState: sseq, interp3
import ..DataLoadsFunc: StructGsupply, StructRWParams
import ..ParamsFunctions: StructParams

export transeq, renewshare_path_region, renewshare_path_world, renewshareUS, welfare_wagechange_2040, welfare_capitalchange_2040, 
welfare_electricitychange_2040, welfare_fossilchange_2040, YUS_rel

# initialize data used in welfare 
Init_weight = Vector{Float64}(undef, 2531)
welfare_wagechange_2040 = Vector{Float64}(undef, 2531)
welfare_capitalchange_2040 = Vector{Float64}(undef, 2531)
welfare_electricitychange_2040 = Vector{Float64}(undef, 2531)
welfare_fossilchange_2040 = Vector{Float64}(undef, 2531)
renewshare_path_region = Matrix{Float64}(undef, 13, 501)
renewshare_path_region2 = Matrix{Float64}(undef, 13, 501)
TotalK = Matrix{Float64}(undef, 1, 501)
renewshareUS2 = Matrix{Float64}(undef, 1, 501)
renewshare_path_world = Matrix{Float64}(undef, 1, 501)


# ---------------------------------------------------------------------------- #
#                           Set Up Subsidy Variables                           #
# ---------------------------------------------------------------------------- #
st[1:majorregions.rowid[1], 2:11] .= subsidy_US
st[1:majorregions.rowid[1], 12] .= subsidy_US / 2

# ---------------------------------------------------------------------------- #
#                    Solve the transition market equilibrium                   #
# ---------------------------------------------------------------------------- #

transeq = solve_transition(R_LR, GsupplyCurves, decayp, T, params, sseq, KR_init_S, KR_init_W, mrkteq, Initialprod, RWParams, curtailmentswitch, 
                                p_KR_bar_init, laboralloc_init, regionParams, majorregions, Linecounts, linconscount, 
                                kappa, regions, Transiter, st, hoursofstorage, pB_shifter, g, 
                                wage_init, p_KR_init_S, p_KR_init_W, p_F_int)


# ---------------------------------------------------------------------------- #
#                                WELFARE IN 2040                               #
# ---------------------------------------------------------------------------- #


@views Init_weight .= wage_init .* params.L .+ (1 - params.beta) .* transeq.r_path[:, 1] .* transeq.PC_path_guess[:, 1] .* mrkteq.KP_init .+
                (1 - params.beta) .* (transeq.r_path[:, 1] .* transeq.KR_path[:, 1] .* transeq.p_KR_bar_path[:, 1] .+ transeq.r_path[:, 1] .* transeq.KF_path[:, 1] .* transeq.PC_path_guess[:, 1]) .+
                transeq.fossilsales_path[:, 1]

@views welfare_wagechange_2040 .= (log.(transeq.w_path_guess[:, 20] ./ transeq.PC_path_guess[:, 20]) .- log.(wage_init ./ mrkteq.PC_guess_init)) .*
                            (wage_init .* params.L ./ Init_weight)

@views welfare_capitalchange_2040 .= (log.(transeq.r_path[:, 20] .* transeq.PC_path_guess[:, 20] .* transeq.KP_path_guess[:, 20] ./ transeq.PC_path_guess[:, 20]) .-
                                log.(transeq.r_path[:, 1] .* mrkteq.PC_guess_init .* mrkteq.KP_init ./ mrkteq.PC_guess_init)) .*
                                ((1-params.beta) .* transeq.r_path[:, 1] .* transeq.PC_path_guess[:, 1] .* mrkteq.KP_init ./ Init_weight)

# add up value of capital stock
@views welfare_electricitychange_2040 .= (log.((transeq.r_path[:,20] .* transeq.KR_path[:, 20] .* transeq.p_KR_bar_path[:, 20] .* transeq.PC_path_guess[:, 20] .+ transeq.r_path[:, 20] .* transeq.KF_path[:, 20] .* transeq.PC_path_guess[:, 20]) ./ transeq.PC_path_guess[:, 20]) .-
                                    log.((transeq.r_path[:, 1] .* transeq.KR_path[:, 1] .* transeq.p_KR_bar_path[:, 1] .* transeq.PC_path_guess[:, 1] .+ transeq.r_path[:, 1] .* transeq.KF_path[:, 1] .* transeq.PC_path_guess[:, 1]))) .*
                                    ((1 - params.beta) .* (transeq.r_path[:, 1] .* transeq.KR_path[:, 1] .* transeq.p_KR_bar_path[:, 1] .* transeq.PC_path_guess[:, 1] .+ transeq.r_path[:, 1] .* transeq.KF_path[:, 1] .* transeq.PC_path_guess[:, 1]) ./ Init_weight)

@views welfare_fossilchange_2040 .= (log.(transeq.fossilsales_path[:, 20] ./ transeq.PC_path_guess[:, 20]) .- log.(transeq.fossilsales_path[:, 1] ./ mrkteq.PC_guess_init)) .* (transeq.fossilsales_path[:, 1] ./ Init_weight)

# ---------------------------------------------------------------------------- #
#                              SUPPLEMENTARY STUFF                             #
# ---------------------------------------------------------------------------- #

# save the path for the price of capital

for kk in 1:params.N
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    @views renewshare_path_region[kk, :] = (1 .- sum(transeq.YF_path[ind, :], dims=1) ./ sum(transeq.Y_path[ind, :], dims=1))'
end
    
for kk = 1:params.N
    ind = majorregions.rowid2[kk]:majorregions.rowid[kk]
    @views renewshare_path_region2[kk, :] = (1 .- sum(transeq.YR_path[ind, :], dims=1) ./ sum(transeq.Y_path[ind, :], dims=1))'  # uses power path
end

TotalK .= sum(transeq.KR_path, dims=1)
renewshareUS2 .= (1 .- (sum(transeq.YF_path[1:700, :], dims = 1) ./ sum(transeq.Y_path[1:700, :], dims=1)))
renewshare_path_world .= 1 .- (sum(transeq.YF_path, dims = 1) ./ sum(transeq.Y_path, dims = 1))
KUS = sum(transeq.KR_path[1:700, :], dims=1)

plot(1:100, KUS[1:100], label="KUS")
plot!(1:100, renewshareUS2[1:100], label="renewshareUS")

KEU = sum(transeq.KR_path[800:1500, :], dims = 1)
KCH = sum(transeq.KR_path[1614:1643, :], dims = 1)
plot(1:100, log.(KEU[1:100]), label = "KEU")
plot!(1:100, log.(KCH[1:100]), label = "KCH")

# in julia, plots automatically clear

YCHF = sum(transeq.YF_path[1493:1522, :], dims=1)
plot(1:100, log.(YCHF[1:100]), label="YCHF", legend=:topright)

YCHR = sum(transeq.YR_path[1493:1522, :], dims=1)
plot!(1:100, log.(YCHR[1:100]), label="YCHR")

x = transeq.YR_path ./ transeq.Y_path

KBR = sum(transeq.KR_path[1842:1969, :], dims=1)
KRUS = sum(transeq.KR_path[1970:2300, :], dims=1)

YUS = sum(transeq.Y_path[1:722, :], dims=1)
YUS_rel = YUS ./ YUS[1]

normalized_KR_path_S = transeq.p_KR_path_S[:, 1:20] ./ transeq.p_KR_path_S[:, 1]
normalized_KR_path_W = transeq.p_KR_path_W[:, 1:20] ./ transeq.p_KR_path_W[:, 1]

plot(1:20, normalized_KR_path_S', label="KR Path S", legend=:topright)
plot!(1:20, normalized_KR_path_W', label="KR Path W")

if config.hoursofstorage==0
    matwrite("$G/p_F_path_guess_saved.mat", Dict("p_F_path_guess" => p_F_path_guess))
end

end

