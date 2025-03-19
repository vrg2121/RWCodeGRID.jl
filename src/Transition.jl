module Transition

# load functions
using ..DataAdjustments, ..TransitionFunctions, ..MarketEquilibrium

# load packages
using Ipopt, JuMP, Interpolations
import Random: Random
import DataFrames: DataFrame
import MAT: matwrite

import ..ModelConfiguration: ModelConfig

# import parameters, data and variables

export solve_transition

"""
    solve_transition(P::NamedTuple, DL::NamedTuple, M::NamedTuple, S::NamedTuple, Subsidy::Int, config::ModelConfig, G::String)

Model of the renewable energy transition up to 2040.

## Inputs
- `P::NamedTuple` -- NamedTuple of parameters. Output of `P = setup_parameters(D, G)`
- `D::NamedTuple` -- NamedTuple of model data. Output of `DL = load_data(P, D)`
- `M::NamedTuple` -- NamedTuple of market equilibrium. Output of `M = solve_market(P, DL, config, G)`
- `S::NamedTuple` -- NamedTuple of steady state equilibrium. Output of `S = solve_steadystate(P, DL, M, config, Guesses)`
- `Subsidy::Int` -- Whether or not to calculate transition with a renewable energy subsidy.
- `config::ModelConfig` -- struct of user defined model configurations. `config = ModelConfig()`
- `G::String` -- path to Guesses folder. `G = "path/to/Guesses"`

## Outputs
Named tuple containing path of renewable energy transition across specific regions and the world; share of renewables in the US; wage changes,
    capital changes, electricity changes and fossil fuel changes until 2040. Saves price of fuel when hours of battery 
    storage is 0 in Guesses.

## Notes
Calculated with some variations when RunTransition==1, RunBatteries==1, RunExog==1, RunCurtailment==1. Not calculated when RunImprovement==1.
"""
function solve_transition(P::NamedTuple, DL::NamedTuple, M::NamedTuple, S::NamedTuple, Subsidy::Int, config::ModelConfig, G::String)
    # set st 
    st = zeros(P.params.J, P.T + 1)
    if Subsidy == 1
        st[1:P.majorregions.rowid[1], 2:11] .= M.mrkteq.subsidy_US
        st[1:P.majorregions.rowid[1], 12] .= M.mrkteq.subsidy_US ./ 2
    end

    pB_shifter = P.pB_shifter
    if config.RunBatteries == 1 && Subsidy == 0
        pB_shifter = P.pkwh_B / P.pkw_solar
    end

    curtailmentswitch = P.curtailmentswitch
    if config.RunCurtailment == 1
        curtailmentswitch = 1
    end

    # initialize data used in welfare 
    Init_weight = Vector{Float64}(undef, 2531)
    welfare_wagechange_2040 = Vector{Float64}(undef, 2531)
    welfare_capitalchange_2040 = Vector{Float64}(undef, 2531)
    welfare_electricitychange_2040 = Vector{Float64}(undef, 2531)
    welfare_fossilchange_2040 = Vector{Float64}(undef, 2531)
    renewshare_path_region = Matrix{Float64}(undef, 13, 501)
    renewshare_path_region2 = Matrix{Float64}(undef, 13, 501)
    TotalK = Matrix{Float64}(undef, 1, 501)
    renewshareUS = Matrix{Float64}(undef, 1, 501)
    renewshare_path_world = Matrix{Float64}(undef, 1, 501)

    transeq = solve_transition_eq(DL.R_LR, DL.GsupplyCurves, P.decayp, P.T, P.params, S.sseq, DL.KR_init_S, 
                                DL.KR_init_W, M.mrkteq, config.Initialprod, DL.RWParams, curtailmentswitch, 
                                M.p_KR_bar_init, M.laboralloc_init, DL.regionParams, P.majorregions, P.Linecounts, P.linconscount, 
                                P.kappa, P.regions, config.Transiter, st, config.hoursofstorage, pB_shifter, P.g, 
                                DL.wage_init, M.p_KR_init_S, M.p_KR_init_W, M.p_F_int, S.interp3)


    # ---------------------------------------------------------------------------- #
    #                                WELFARE IN 2040                               #
    # ---------------------------------------------------------------------------- #


    @views Init_weight .= DL.wage_init .* P.params.L .+ (1 - P.params.beta) .* transeq.r_path[:, 1] .* transeq.PC_path_guess[:, 1] .* M.mrkteq.KP_init .+
                    (1 - P.params.beta) .* (transeq.r_path[:, 1] .* transeq.KR_path[:, 1] .* transeq.p_KR_bar_path[:, 1] .+ transeq.r_path[:, 1] .* transeq.KF_path[:, 1] .* transeq.PC_path_guess[:, 1]) .+
                    transeq.fossilsales_path[:, 1]

    @views welfare_wagechange_2040 .= (log.(transeq.w_path_guess[:, 20] ./ transeq.PC_path_guess[:, 20]) .- log.(DL.wage_init ./ M.mrkteq.PC_guess_init)) .*
                                (DL.wage_init .* P.params.L ./ Init_weight)

    @views welfare_capitalchange_2040 .= (log.(transeq.r_path[:, 20] .* transeq.PC_path_guess[:, 20] .* transeq.KP_path_guess[:, 20] ./ transeq.PC_path_guess[:, 20]) .-
                                    log.(transeq.r_path[:, 1] .* M.mrkteq.PC_guess_init .* M.mrkteq.KP_init ./ M.mrkteq.PC_guess_init)) .*
                                    ((1-P.params.beta) .* transeq.r_path[:, 1] .* transeq.PC_path_guess[:, 1] .* M.mrkteq.KP_init ./ Init_weight)

    # add up value of capital stock
    @views welfare_electricitychange_2040 .= (log.((transeq.r_path[:,20] .* transeq.KR_path[:, 20] .* transeq.p_KR_bar_path[:, 20] .* transeq.PC_path_guess[:, 20] .+ transeq.r_path[:, 20] .* transeq.KF_path[:, 20] .* transeq.PC_path_guess[:, 20]) ./ transeq.PC_path_guess[:, 20]) .-
                                        log.((transeq.r_path[:, 1] .* transeq.KR_path[:, 1] .* transeq.p_KR_bar_path[:, 1] .* transeq.PC_path_guess[:, 1] .+ transeq.r_path[:, 1] .* transeq.KF_path[:, 1] .* transeq.PC_path_guess[:, 1]))) .*
                                        ((1 - P.params.beta) .* (transeq.r_path[:, 1] .* transeq.KR_path[:, 1] .* transeq.p_KR_bar_path[:, 1] .* transeq.PC_path_guess[:, 1] .+ transeq.r_path[:, 1] .* transeq.KF_path[:, 1] .* transeq.PC_path_guess[:, 1]) ./ Init_weight)

    @views welfare_fossilchange_2040 .= (log.(transeq.fossilsales_path[:, 20] ./ transeq.PC_path_guess[:, 20]) .- log.(transeq.fossilsales_path[:, 1] ./ M.mrkteq.PC_guess_init)) .* (transeq.fossilsales_path[:, 1] ./ Init_weight)

    # ---------------------------------------------------------------------------- #
    #                              SUPPLEMENTARY STUFF                             #
    # ---------------------------------------------------------------------------- #

    # save the path for the price of capital

    for kk in 1:P.params.N
        ind = P.majorregions.rowid2[kk]:P.majorregions.rowid[kk]
        @views renewshare_path_region[kk, :] = (1 .- sum(transeq.YF_path[ind, :], dims=1) ./ sum(transeq.Y_path[ind, :], dims=1))'
    end
        
    for kk = 1:P.params.N
        ind = P.majorregions.rowid2[kk]:P.majorregions.rowid[kk]
        @views renewshare_path_region2[kk, :] = (1 .- sum(transeq.YR_path[ind, :], dims=1) ./ sum(transeq.Y_path[ind, :], dims=1))'  # uses power path
    end

    TotalK .= sum(transeq.KR_path, dims=1)
    renewshareUS .= (1 .- (sum(transeq.YF_path[1:700, :], dims = 1) ./ sum(transeq.Y_path[1:700, :], dims=1)))
    renewshare_path_world .= 1 .- (sum(transeq.YF_path, dims = 1) ./ sum(transeq.Y_path, dims = 1))
    
    YUS = sum(transeq.Y_path[1:722, :], dims=1)
    YUS_rel = YUS ./ YUS[1]
    
    p_F_path_guess = transeq.p_F_path_guess

    if config.hoursofstorage==0
        matwrite("$G/p_F_path_guess_saved.mat", Dict("p_F_path_guess" => p_F_path_guess))
    end

    return (
        transeq = transeq,
        renewshare_path_region = renewshare_path_region,
        renewshare_path_world = renewshare_path_world,
        renewshareUS = renewshareUS,
        welfare_wagechange_2040 = welfare_wagechange_2040,
        welfare_capitalchange_2040 = welfare_capitalchange_2040,
        welfare_electricitychange_2040 = welfare_electricitychange_2040,
        welfare_fossilchange_2040 = welfare_fossilchange_2040,
        YUS_rel = YUS_rel,
        st = st
    )
end

end

