module SteadyState

# load functions
using ..SteadyStateFunctions, ..DataAdjustments, ..MarketEquilibrium
 
import MAT: matwrite
import Interpolations: interpolate, Gridded, Linear

# import parameters, data and variables
import ..ModelConfiguration: ModelConfig

# export variables
export solve_steadystate


# ---------------------------------------------------------------------------- #
#                             Step 0: Initial Guess                            #
# ---------------------------------------------------------------------------- #


"""
NOTES ON THE TRANSLATION FROM JULIA TO MATLAB

Manually Solving NLP Optimization Problem using Ipopt.jl
user defined gradient:
- Ipopt optimizer is unstable when it comes to multithreading and throws errors with large numbers of threads
- the first multithreaded loop keeps ending up with a negative number in x, which is technically impossible
    - checking the error suggests that there is instability in the model that is difficult to resolve
    - the error only crops up after the while loops have run 6-22 times
- A user defined gradient slows down the model because you cannot multithread it without creating errors
- results from model with user defined gradient the same as results from automatic solve

user defined hessian:
- There is not a good way to extract the lagrangian as the model is being solved in order to manually calculate the hessian



Calculating a 3D Interpolation (MATLAB interp3()) using Interpolations.jl:
- Interpolation in Julia does not require an entire meshgrid, just a range of values defined by starting point, end point and step size
- In Julia, set up the interpolation function to align with the matlab interp3()
- Key difference between MATLAB and Julia is that to get to the same valure, you have to flip the x and y inputs in Julia 
- example:
    interp3 = interpolate((x, y, z), mat_data, Gridded(Linear()))
    X, Y, Z = points of interest
    interp_value = interp3(Y, X, Z)

- I am sure there is a specific mathematical reason for this, but the documentation in Interpolations.jl is light on details. I derived
  this by comparing graphs of the interpolation grids between MATLAB and Julia.
"""


"""
    solve_steadystate(P::NamedTuple, D::NamedTuple, M::NamedTuple, config::ModelConfig, G::String)

Solve the steady-state equilibrium for wind and solar in the energy grid.

# Arguments
- `P::NamedTuple`: NamedTuple of parameters, output of `P = setup_parameters(D, G)`
- `D::NamedTuple`: NamedTuple of model data, output of `DL = load_data(P, D)`
- `M::NamedTuple`: NamedTuple of market equilibrium, output of `M = solve_market(P, DL, config, G)`
- `config::ModelConfig`: User defined model configurations.
- `G::String`: Path to the Guesses folder, e.g., `G = "path/to/Guesses"`

# Returns
NamedTuple containing steady-state levels of GDP, wages, labor, capital, electricity, fossil fuels, etc.

# Notes
Updates some guesses when hours of storage = 0.
"""
function solve_steadystate(P::NamedTuple, D::NamedTuple, M::NamedTuple, config::ModelConfig, G::String)

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

    sseq = solve_power_output(D.RWParams, P.params, config.RunBatteries, config.RunCurtailment,
                                                config.Initialprod, D.R_LR, P.majorregions, P.Linecounts, P.linconscount,
                                                D.regionParams, curtailmentswitch, interp3,
                                                P.T, P.kappa, M.mrkteq, config, pB_shifter, G);

    println("Steady State diffK= ", sseq.diffK)
    println("Steady State diffp= ", sseq.diffp)



    # Get fossil fuel usage in the initial steady state
    #YF_LR = max.(sseq.YE_LR .- D.regionParams.theta .* sseq.KR_LR, 0)
    FUtilization = sseq.YF_LR ./ sseq.KF_LR
    #mean(FUtilization[1:P.majorregions.n[1]]))

    # Compute electricity and fossil fuel usage in industry and electricity sectors
    e2_LR = M.mrkteq.laboralloc .* repeat(sseq.D_LR, 1, P.params.I) .* (repeat(P.params.Vs[:,2]', P.params.J, 1) ./ (repeat(P.params.Vs[:,2]', P.params.J, 1) + repeat(P.params.Vs[:,3]', P.params.J, 1)))
    fusage_ind_LR = sum(e2_LR, dims=2) .* (sseq.p_E_LR / sseq.p_F_LR) .^ P.params.psi
    fusage_power_LR = (sseq.YF_LR ./ sseq.KF_LR .^ P.params.alpha2) .^ (1 / P.params.alpha1)
    fusage_total_LR = sum(fusage_power_LR) + sum(fusage_ind_LR)

    # compute fossil usage as a share of GDP
    GDP = sum(sseq.w_LR .* P.params.L .+ sseq.p_E_LR .* sseq.D_LR .+ sseq.rP_LR .* sseq.KP_LR .+ sseq.p_F_LR .* fusage_ind_LR)
    Fossshare = sum(sseq.p_F_LR .* fusage_ind_LR) / GDP

    # Save long run variables for transition
    I_tota_S_LR = copy(sseq.Depreciation_LR_S)
    I_tota_W_LR = copy(sseq.Depreciation_LR_W)

    KR_LR_S = sseq.KR_LR_S
    KR_LR_W = sseq.KR_LR_W
    p_E_LR = sseq.p_E_LR
    w_LR = sseq.w_LR
    result_Dout_LR = sseq.result_Dout_LR
    result_Yout_LR = sseq.result_Yout_LR
    PC_guess_LR = sseq.PC_guess_LR
    laboralloc_LR = sseq.laboralloc_LR

    if config.hoursofstorage == 0
        matwrite("$G/alloc_LR_guess.mat", Dict("laboralloc_LR" => laboralloc_LR))
        matwrite("$G/KR_LR_S_guess.mat", Dict("KR_LR_S" => KR_LR_S))
        matwrite("$G/KR_LR_W_guess.mat", Dict("KR_LR_W" => KR_LR_W))
        matwrite("$G/p_E_LR_guess.mat", Dict("p_E_LR" => p_E_LR))
        matwrite("$G/w_LR_guess.mat", Dict("w_LR" => w_LR))
        matwrite("$G/Dout_guess_LR.mat", Dict("result_Dout_LR" => result_Dout_LR))
        matwrite("$G/Yout_guess_LR.mat", Dict("result_Yout_LR" => result_Yout_LR))
        matwrite("$G/PC_guess_LR.mat", Dict("PC_guess_LR" => PC_guess_LR))
    end


    M.wageresults[:,2] = sseq.w_real
    wr = (M.wageresults[:, 2] ./ M.wageresults[:, 1]) .- 1
    wagechange = [wr P.regions.csr_id]

    priceresults_LR = sseq.p_E_LR

    # Get changes in welfare from the different components
    welfare_wagechange = (log.(sseq.w_LR ./ sseq.PC_guess_LR) .- log.(D.wage_init ./ M.mrkteq.PC_guess_init)) .* (D.wage_init .* P.params.L ./ M.mrkteq.Expenditure_init)
    welfare_capitalchange = (log.(sseq.KP_LR ./ sseq.PC_guess_LR) .- log.(M.mrkteq.KP_init ./ M.mrkteq.PC_guess_init)) .* (M.mrkteq.rP_init .* M.mrkteq.KP_init ./ M.mrkteq.Expenditure_init)

    welfare_electricitychange = (log.((D.R_LR .* sseq.KR_LR .* sseq.p_KR_bar_LR .* sseq.PC_LR + D.R_LR .* sseq.KF_LR .* sseq.PC_LR) ./ sseq.PC_LR) .- 
        log.((D.R_LR .* (D.KR_init_W + D.KR_init_S) .* M.p_KR_bar_init .* M.mrkteq.PC_init + D.R_LR .* M.KF_init .* M.mrkteq.PC_init))) .* 
        ((1 - P.params.beta) * (D.R_LR .* (D.KR_init_W + D.KR_init_S) .* M.p_KR_bar_init .* M.mrkteq.PC_init + D.R_LR .* M.KF_init .* M.mrkteq.PC_init) ./ M.mrkteq.Expenditure_init)

    welfare_fossilchange = -M.mrkteq.fossilsales ./ M.mrkteq.Expenditure_init

    # Get changes in manufacturing share
    comparativeadvantagechange = M.laboralloc_init ./ sseq.laboralloc_LR

    w_real_LR = sseq.w_LR ./ sseq.PC_guess_LR
    GDP_ind_LR = (sseq.w_LR .* P.params.L .+ sseq.p_E_LR .* sseq.D_LR .+ sseq.rP_LR .* sseq.KP_LR .+ sseq.p_F_LR .* fusage_ind_LR) ./ sseq.PC_guess_LR

    return (
    sseq = sseq,
    interp3 = interp3,
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