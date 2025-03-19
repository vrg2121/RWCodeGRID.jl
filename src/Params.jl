# Params.jl
# working directory: TradeAndGrowth/Code/RWCodeJulia

module Params
export setup_parameters
# VARIABLES INITIATED, NOT USED FOR EXPORT: updps, updw_alpha, upw_zmax, updw_l, updw_D, updw_k

# import functions
using ..ParamsFunctions

"""
    setup_parameters(Data::String, Guesses::String)

Set parameters for the model run.

## Inputs
- `Data` -- full path to Data folder.
- `Guesses` -- full path to Guesses folder.

## Output
Model parameters wrapped in a named tuple. The named tuple 
    contains constants and structs like `params`.

## Notes
Descriptions for each parameter are in the submodule `ParamsFunctions.jl` comments.

## Example
```julia-repl
julia> D = "path/to/Data"
julia> G = "path/to/Guesses"
julia> P = setup_parameters(D, G) # the named tuple is saved in variable P
```
"""
function setup_parameters(D::String, G::String)
    # preallocate parameters
    thetaS = Vector{Float64}(undef, 2531)
    thetaW = Vector{Float64}(undef, 2531)

    # ---------------------------------------------------------------------------- #
    #                               Create Parameters                              #
    # ---------------------------------------------------------------------------- #

    regions, majorregions, Linecounts = load_parameters_csv(D)

    params = fill_params(regions, majorregions, Linecounts, D, G)

    # renewable potential
    capacityfactorS = 1


    thetaS = tS!(thetaS, capacityfactorS, regions)

    capacityfactorW = 1

    thetaW = tW!(thetaW, capacityfactorW, regions)
    theta = max.(thetaS, thetaW)

    # update weights for iterations
    updw_k = 0.01
    updw_w = 0.1
    updw_D = 0.01
    updw_l = 0.00001
    upw_z = 0.001
    updwk = 0.5
    updwF = 0.001
    upw_zmax = 0.001
    updw_alpha = 0.2
    upda = 0.001
    updps = 0.1

    # adjust for population costs
    popelas = 0.2

    # speed up for line constraints to ignore
    linconscount = 200

    # relative advantage of fossil fuels initially
    kappa = 1

    # exogenous growth rate in extraction productivity
    g = 0.01

    # battery initialisation
    pkwh_B = 412.37   #Initial price of batteries
    pkw_solar = 2000
    hoursofstorage = 0 #initialisation
    pB_shifter = 0.0

    #curtailment intialisation
    curtailmentswitch = 0

    # Transition Parameters
    T = 500
    decayp = -0.05

    return (
        params = params,
        thetaS = thetaS,
        theta = theta,
        thetaW = thetaW,
        regions = regions,
        majorregions = majorregions,
        popelas = popelas,
        T = T,
        Linecounts = Linecounts,
        linconscount = linconscount,
        kappa = kappa,
        updw_w = updw_w,
        upw_z = upw_z,
        curtailmentswitch = curtailmentswitch,
        decayp = decayp,
        hoursofstorage = hoursofstorage,
        pB_shifter = pB_shifter,
        pkw_solar = pkw_solar,
        pkwh_B = pkwh_B,
        g = g,
        upda = upda,
        updwF = updwF,
        updwk = updwk
    )

end


end