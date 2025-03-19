module TransitionExog

# load functions
using ..DataAdjustments, ..TransitionExogFunc, ..MarketEquilibrium

# load packages
using Ipopt, JuMP, Interpolations
import Random: Random
import DataFrames: DataFrame
import SparseArrays: sparse
import ..ModelConfiguration: ModelConfig

export solve_transition_exog

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
Named tuple containing path of renewable energy transition with exogenous tech across specific regions and the world; share of renewables in the US; wage changes,
    capital changes, electricity changes and fossil fuel changes until 2040.

## Notes
Calculated with some variations when RunTransition==1, RunBatteries==1, RunExog==1, RunCurtailment==1. Not calculated when RunImprovement==1.
"""
function solve_transition_exog(P::NamedTuple, DL::NamedTuple, M::NamedTuple, S::NamedTuple, config::ModelConfig, exogindex::Int64)

    # set st 
    st = zeros(P.params.J, P.T + 1)

    # initialize data used in welfare 
    Init_weight = Vector{Float64}(undef, 2531)
    welfare_wagechange_2040 = Vector{Float64}(undef, 2531)
    welfare_capitalchange_2040 = Vector{Float64}(undef, 2531)
    welfare_electricitychange_2040 = Vector{Float64}(undef, 2531)
    welfare_fossilchange_2040 = Vector{Float64}(undef, 2531)
    renewshareUS = Matrix{Float64}(undef, 1, 501)
    renewshare_path = Matrix{Float64}(undef, 13, 501)
    renewshare_path_region = Matrix{Float64}(undef, 13, 501)
    renewshare_path_world = Matrix{Float64}(undef, 1, 501)


    transeq = solve_transitioneq_exog(DL.R_LR, DL.GsupplyCurves, P.decayp, P.T, P.params, S.sseqE, DL.KR_init_S, 
                                DL.KR_init_W, M.mrkteq, config.Initialprod, DL.RWParams, 
                                M.p_KR_bar_init, M.laboralloc_init, DL.regionParams, P.majorregions, P.Linecounts, P.linconscount, 
                                P.kappa, P.regions, config.Transiter, st, config.hoursofstorage, P.pB_shifter, P.g, 
                                DL.wage_init, M.p_KR_init_S, M.p_KR_init_W, M.p_F_int, exogindex, DL.projectionssolar, DL.projectionswind)


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
        @views renewshare_path[kk, :] .= (sum(transeq.YR_path[ind, :], dims=1) ./ sum(transeq.Y_path[ind, :], dims=1))'  # uses power path
    end

    renewshareUS .= (1 .- (sum(transeq.YF_path[1:700, :], dims = 1) ./ sum(transeq.Y_path[1:700, :], dims=1)))
    renewshare_path_world .= 1 .- (sum(transeq.YF_path, dims = 1) ./ sum(transeq.Y_path, dims = 1))

    return (
        transeq = transeq,
        renewshare_path = renewshare_path,
        renewshareUS = renewshareUS,
        renewshare_path_region = renewshare_path_region,
        renewshare_path_world = renewshare_path_world,
        welfare_wagechange_2040 = welfare_wagechange_2040,
        welfare_capitalchange_2040 = welfare_capitalchange_2040,
        welfare_electricitychange_2040 = welfare_electricitychange_2040,
        welfare_fossilchange_2040 = welfare_fossilchange_2040,
        st = st
    )

end

end

