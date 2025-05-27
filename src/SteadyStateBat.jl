module SteadyStateBat

# load functions
using ..SteadyStateFunctions, ..DataAdjustments, ..MarketEquilibrium
 
import MAT: matwrite
import Interpolations: interpolate, Gridded, Linear

# import parameters, data and variables
import ..ModelConfiguration: ModelConfig
import DrawGammas: StructAllParams
import ..DataLoads: StructAllData
import ..Market: StructMarketOutput
import ..SteadyState: StructSteadyState

# export variables
export solve_steadystate_bat, StructSteadyStateBat

mutable struct StructSteadyStateBat
    sseq::StructPowerOutput                      
    interp3::Any                 
    GDP::Float64
    wr::Vector{Float64}
    wagechange::Matrix{Float64}
    welfare_wagechange::Matrix{Float64}
    welfare_capitalchange::Matrix{Float64}
    welfare_electricitychange::Matrix{Float64}
    welfare_fossilchange::Matrix{Float64}
end

function solve_steadystate_bat(S, P::StructAllParams, D::StructAllData, M::StructMarketOutput, config::ModelConfig, G::String)
    
    pB_shifter = P.pB_shifter
    if config.RunBatteries == 1
        pB_shifter = P.pkwh_B / P.pkw_solar
    end

    curtailmentswitch = P.curtailmentswitch
    if config.RunCurtailment == 1
        curtailmentswitch = 1
    end

    x = range(start = 0.0, stop = 1.0, step = 0.05) 
    y = range(start = 0.0, stop = 1.0, step = 0.05) 
    z = range(start = 0.0, stop = 12.0, step = 6.0)

    interp3 = interpolate((x, y, z), D.curtmat, Gridded(Linear()))


    # ------------------ Solve Power Output within given capital ----------------- #

    sseq = solve_power_output_bat(D.RWParams, P.params, config.RunBatteries, config.RunCurtailment,
                                                config.Initialprod, D.R_LR, P.majorregions, P.Linecounts, P.linconscount,
                                                D.regionParams, curtailmentswitch, interp3,
                                                P.T, P.kappa, M.mrkteq, config, pB_shifter, S.sseq, G);

    println("Steady State diffK= ", sseq.diffK)
    println("Steady State diffp= ", sseq.diffp)

    # -------------------------- Pre-allocate Variables -------------------------- #
    wr = Vector{Float64}(undef, 2531)
    e2_LR = Matrix{Float64}(undef, 2531, 10)
    fusage_ind_LR = Matrix{Float64}(undef, 2531, 1)
    wagechange = Matrix{Float64}(undef, 2531, 2)
    welfare_fossilchange = Matrix{Float64}(undef, 2531, 1)


    # Compute electricity and fossil fuel usage in industry and electricity sectors
    up_e2_LR!(e2_LR, P.params.Vs, M.mrkteq.laboralloc, sseq.D_LR) # 57.200 μs (22 allocations: 198.88 KiB)
    fusage_ind_LR .= sum(e2_LR, dims=2) .* sseq.p_E_LR .^ P.params.psi #    53.600 μs (11 allocations: 20.08 KiB)
   
    # compute fossil usage as a share of GDP
    GDP = sum(sseq.w_LR .* P.params.L .+ sseq.p_E_LR .* sseq.D_LR .+ sseq.rP_LR .* sseq.KP_LR .+ sseq.p_F_LR .* fusage_ind_LR) #   7.360 μs (18 allocations: 20.52 KiB)

    M.wageresults[:,2] = sseq.w_real #   2.300 μs (1 allocation: 16 bytes)    
    fill_wr!(wr, M.wageresults) #   3.925 μs (0 allocations: 0 bytes)

    wagechange[:, 1] .= wr
    wagechange[:, 2] .= P.regions.csr_id


    # Get changes in welfare from the different components
    welfare_wagechange = (log.(sseq.w_LR ./ sseq.PC_guess_LR) .- log.(D.wage_init ./ M.mrkteq.PC_guess_init)) .* (D.wage_init .* P.params.L ./ M.mrkteq.Expenditure_init) #   50.200 μs (18 allocations: 20.45 KiB)
    welfare_capitalchange = (log.(sseq.KP_LR ./ sseq.PC_guess_LR) .- log.(M.mrkteq.KP_init ./ M.mrkteq.PC_guess_init)) .* (M.mrkteq.rP_init .* M.mrkteq.KP_init ./ M.mrkteq.Expenditure_init) #   48.400 μs (18 allocations: 20.45 KiB)

    welfare_electricitychange = (log.((D.R_LR .* sseq.KR_LR .* sseq.p_KR_bar_LR .* sseq.PC_LR + D.R_LR .* sseq.KF_LR .* sseq.PC_LR) ./ sseq.PC_LR) .- 
        log.((D.R_LR .* (D.KR_init_W + D.KR_init_S) .* M.p_KR_bar_init .* M.mrkteq.PC_init + D.R_LR .* M.KF_init .* M.mrkteq.PC_init))) .* 
        ((1 - P.params.beta) * (D.R_LR .* (D.KR_init_W + D.KR_init_S) .* M.p_KR_bar_init .* M.mrkteq.PC_init + D.R_LR .* M.KF_init .* M.mrkteq.PC_init) ./ M.mrkteq.Expenditure_init) #   93.000 μs (76 allocations: 259.70 KiB)

    welfare_fossilchange .= -M.mrkteq.fossilsales ./ M.mrkteq.Expenditure_init #   8.800 μs (6 allocations: 39.78 KiB)

    return StructSteadyState(
    sseq,
    interp3,
    GDP,
    wr,
    wagechange,
    welfare_wagechange,
    welfare_capitalchange,
    welfare_electricitychange,
    welfare_fossilchange
    )

end

end