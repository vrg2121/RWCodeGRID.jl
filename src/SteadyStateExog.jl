module SteadyStateExog
using ..SteadyStateExogFunc, ..DataAdjustments, ..MarketEquilibrium

# import parameters, data and variables
import ..ModelConfiguration: ModelConfig

# export variables
export solve_steadystate_exog

"""
    solve_steadystate_exog(P::NamedTuple, DL::NamedTuple, M::NamedTuple, config::ModelConfig, exogindex::Int64, G::String)

Solve the steadystate equilibrium for wind and solar in the energy grid.

## Inputs
- `P::NamedTuple` -- NamedTuple of parameters. Output of `P = setup_parameters(D, G)`
- `DL::NamedTuple` -- NamedTuple of model data. Output of `DL = load_data(P, D)`
- `M::NamedTuple` -- NamedTuple of market equilibrium. Output of `M = solve_market(P, DL, config, G)`
- `exogindex::Int64` -- The exogenous tech index: 1, 2, 3
- `config::ModelConfig` -- struct of user defined model configurations.
- `G::String` -- path to Guesses folder. `G = "path/to/Guesses"`

## Outputs
Named tuple containing steadystate with exogenous techs levels of GDP, wages, labor, capital, electricity, fossil fuels, etc.

## Notes
Calculated for only when RunExog==1.
"""
function solve_steadystate_exog(P::NamedTuple, DL::NamedTuple, M::NamedTuple, config::ModelConfig, exogindex::Int64, G::String)
    # ---------------------------------------------------------------------------- #
    #                             Step 0: Initial Guess                            #
    # ---------------------------------------------------------------------------- #
    pB_shifter = 0.0

    # ------------------ Solve Power Output within given capital ----------------- #
    

    sseqE = solve_power_output_exog(DL.RWParams, P.params, config.RunBatteries,
                config.Initialprod, DL.R_LR, P.majorregions, P.Linecounts, P.linconscount,
                DL.regionParams, pB_shifter, P.T, M.mrkteq, DL.projectionssolar, DL.projectionswind, 
                config, exogindex, M.p_KR_init_S, M.p_KR_init_W, P.kappa, G);

    println("Steady State diffK= ", sseqE.diffK)

    # Get fossil fuel usage in the initial steady state
    FUtilization = sseqE.YF_LR ./ sseqE.KF_LR

    # Compute electricity and fossil fuel usage in industry and electricity sectors
    e2_LR = M.mrkteq.laboralloc .* repeat(sseqE.D_LR, 1, P.params.I) .* (repeat(P.params.Vs[:,2]', P.params.J, 1) ./ (repeat(P.params.Vs[:,2]', P.params.J, 1) + repeat(P.params.Vs[:,3]', P.params.J, 1)))
    fusage_ind_LR = sum(e2_LR, dims=2) .* (sseqE.p_E_LR / sseqE.p_F_LR) .^ P.params.psi
    fusage_power_LR = (sseqE.YF_LR ./ sseqE.KF_LR .^ P.params.alpha2) .^ (1 / P.params.alpha1) # issue
    fusage_total_LR = sum(fusage_power_LR) + sum(fusage_ind_LR)

    # compute fossil usage as a share of GDP
    GDP = sum(sseqE.w_LR .* P.params.L .+ sseqE.p_E_LR .* sseqE.D_LR .+ sseqE.rP_LR .* sseqE.KP_LR .+ sseqE.p_F_LR .* fusage_ind_LR)
    Fossshare = sum(sseqE.p_F_LR .* fusage_ind_LR) / GDP

    M.wageresults[:,2] = sseqE.w_real
    wr = (M.wageresults[:, 2] ./ M.wageresults[:, 1]) .- 1
    wagechange = [wr P.regions.csr_id]

    priceresults_LR = sseqE.p_E_LR

    # Get changes in welfare from the different components
    welfare_wagechange = (log.(sseqE.w_LR ./ sseqE.PC_guess_LR) .- log.(DL.wage_init ./ M.mrkteq.PC_guess_init)) .* (DL.wage_init .* P.params.L ./ M.mrkteq.Expenditure_init)
    welfare_capitalchange = (log.(sseqE.KP_LR ./ sseqE.PC_guess_LR) .- log.(M.mrkteq.KP_init ./ M.mrkteq.PC_guess_init)) .* (M.mrkteq.rP_init .* M.mrkteq.KP_init ./ M.mrkteq.Expenditure_init)

    welfare_electricitychange = (log.((DL.R_LR .* sseqE.KR_LR .* sseqE.p_KR_bar_LR .* sseqE.PC_LR + DL.R_LR .* sseqE.KF_LR .* sseqE.PC_LR) ./ sseqE.PC_LR) .- 
        log.((DL.R_LR .* (DL.KR_init_W + DL.KR_init_S) .* M.p_KR_bar_init .* M.mrkteq.PC_init + DL.R_LR .* M.KF_init .* M.mrkteq.PC_init))) .* 
        ((1 - P.params.beta) * (DL.R_LR .* (DL.KR_init_W + DL.KR_init_S) .* M.p_KR_bar_init .* M.mrkteq.PC_init + DL.R_LR .* M.KF_init .* M.mrkteq.PC_init) ./ M.mrkteq.Expenditure_init)

    welfare_fossilchange = -M.mrkteq.fossilsales ./ M.mrkteq.Expenditure_init

    return (
        sseqE = sseqE,
        GDP = GDP,
        wr = wr,
        wagechange = wagechange,
        welfare_wagechange = welfare_wagechange,
        welfare_capitalchange = welfare_capitalchange,
        welfare_electricitychange = welfare_electricitychange,
        welfare_fossilchange = welfare_fossilchange
    )

end

end