# Market.jl
module Market

# export variables
export solve_market, StructMarketOutput

# load functions
using ..DataAdjustments
using ..MarketFunctions
using ..MarketEquilibrium
using ..RegionModel
import ..ModelConfiguration: ModelConfig
import DrawGammas: StructAllParams, StructParams
import ..DataLoads: StructAllData

# load packages
using JuMP, Ipopt
import Random: Random
import LinearAlgebra: I
import MAT: matwrite
import DataFrames: DataFrame

mutable struct StructMarketOutput
    wageresults::Matrix{Float64}
    p_KR_bar_init::Matrix{Float64}  # need to double check
    KF_init::Vector{Float64}
    laboralloc_init::Matrix{Float64}
    p_KR_init_S::Float64 # need to double check
    p_KR_init_W::Float64 # need to double check
    renewshareUS::Float64 # need to double check
    p_F_int::Float64
    mrkteq::StructMarketEq 
    priceresults::Vector{Float64}
end

"""
    solve_market(P::StructAllParams, DL::NamedTuple, config::ModelConfig, G::String)

Solve the initial market equilibrium using parameters, data and guesses. The initial market equilibrium is 
    solved identically for all model configurations.

## Inputs
- `P::StructAllParams` -- NamedTuple containing all model parameters. Output of `P = setup_parameters(D, G)`
- `DL::NamedTuple` -- NamedTuple containing all model data. Output of `DL = load_data(P, Data)`
- `config::ModelConfig` -- model configuration set by the user. Output of `config = ModelConfig()` 
- `G::String` -- path to Guesses folder. `G = "path/to/Guesses"`

## Outputs
Outputs of the market equilibrium and updates to wages, labor, prices, etc.
"""
function solve_market(P::StructAllParams, DL::StructAllData, config::ModelConfig, G::String)
# ---------------------------------------------------------------------------- #
#                           Solve Market Equilibrium                           #
# ---------------------------------------------------------------------------- #

    mrkteq = solve_initial_equilibrium(P.params, DL.wage_init, P.majorregions,
                                        DL.regionParams, DL.KR_init_S, DL.KR_init_W, DL.R_LR, DL.sectoralempshares,
                                        P.Linecounts, P.kappa, P.regions, P.linconscount, P.updw_w, P.upw_z, DL.RWParams, G);

    println("Initial Calibration= ", mrkteq.diffend)

    wageresults = Matrix{Float64}(undef, 2531, 2)
    priceresults = Vector{Float64}(undef, 2531)
    PCresults = Vector{Float64}(undef, 2531)
    laboralloc_init = Matrix{Float64}(undef, 2531, 10)
    KF_init = Vector{Float64}(undef, 2531)

    SShare_init = Matrix{Float64}(undef, 2531, 1)
    p_KR_bar_init = Matrix{Float64}(undef, 2531, 1)

    # set Q initial to be the solar already installed
    Qtotal_init_S = sum(DL.KR_init_S)
    Qtotal_init_W = sum(DL.KR_init_W)
    p_KR_init_S = (config.Initialprod + Qtotal_init_S) ^ (-P.params.gammaS)   
    p_KR_init_W = (config.Initialprod + Qtotal_init_W) ^ (-P.params.gammaW)

    up_SShare_pKRbar_init!(SShare_init, p_KR_bar_init, DL.regionParams.thetaS, p_KR_init_S, P.params.varrho, DL.regionParams.thetaW, p_KR_init_W) #   73.200 μs (1 allocation: 16 bytes)

    wageresults .= copy(mrkteq.W_Real)
    priceresults .= copy(mrkteq.p_E_init)
    laboralloc_init .= copy(mrkteq.laboralloc)
    PCresults .= copy(mrkteq.p_E_init)

    #get renewable shares
    renewshareUS = 1 - (sum(mrkteq.YF_init[1:P.majorregions.n[1],:]) ./ sum(mrkteq.YE_init[1:P.majorregions.n[1]]));

    p_F_int = copy(mrkteq.p_F)
    KF_init .= copy(DL.regionParams.KF)

    return StructMarketOutput(
        wageresults,
        p_KR_bar_init, # check var
        KF_init, # check var
        laboralloc_init, # check var
        p_KR_init_S,
        p_KR_init_W,
        renewshareUS,
        p_F_int, # check var
        mrkteq,
        priceresults
)

end

function solve_market_batteries(P::StructAllParams, DL::StructAllData, )
    mrkteq = solve_initial_equilibrium(P.params, DL.wage_init, P.majorregions,
                                        DL.regionParams, DL.KR_init_S, DL.KR_init_W, DL.R_LR, DL.sectoralempshares,
                                        P.Linecounts, P.kappa, P.regions, P.linconscount, P.updw_w, P.upw_z, DL.RWParams, G);

    println("Initial Calibration= ", mrkteq.diffend)

    wageresults = Matrix{Float64}(undef, 2531, 2)
    priceresults = Vector{Float64}(undef, 2531)
    PCresults = Vector{Float64}(undef, 2531)
    laboralloc_init = Matrix{Float64}(undef, 2531, 10)
    KF_init = Vector{Float64}(undef, 2531)

    SShare_init = Matrix{Float64}(undef, 2531, 1)
    p_KR_bar_init = Matrix{Float64}(undef, 2531, 1)

    # set Q initial to be the solar already installed
    Qtotal_init_S = sum(DL.KR_init_S)
    Qtotal_init_W = sum(DL.KR_init_W)
    p_KR_init_S = (config.Initialprod + Qtotal_init_S) ^ (-P.params.gammaS)   
    p_KR_init_W = (config.Initialprod + Qtotal_init_W) ^ (-P.params.gammaW)

    up_SShare_pKRbar_init!(SShare_init, p_KR_bar_init, DL.regionParams.thetaS, p_KR_init_S, P.params.varrho, DL.regionParams.thetaW, p_KR_init_W) #   73.200 μs (1 allocation: 16 bytes)

    wageresults .= copy(mrkteq.W_Real)
    priceresults .= copy(mrkteq.p_E_init)
    laboralloc_init .= copy(mrkteq.laboralloc)
    PCresults .= copy(mrkteq.p_E_init)

end

# ---------------------------------------------------------------------------- #
#                                   Addendum                                   #
# ---------------------------------------------------------------------------- #

# ----------------------------- unused variables ----------------------------- #
#subsidy_US = mrkteq.subsidy_US
#thetabar_init = DL.regionParams.thetaS .* SShare_init + DL.regionParams.thetaW .* (1 .- SShare_init)
#pE_FE_init=(p_KR_bar_init-p_KR_bar_init*(1-P.params.deltaR)./DL.R_LR).*DL.regionParams.costshifter./thetabar_init
#renewshareEU=1-(sum(mrkteq.YF_init[P.majorregions.rowid[1]+1:P.majorregions.rowid[2]]))./sum(mrkteq.YE_init[P.majorregions.rowid[1]+1:P.majorregions.rowid[2]])


## set initial power output vector
#P_out_init = mrkteq.P_out

#Z = P.params.Z
#Zsec = P.params.zsector
#w_guess = mrkteq.w_guess
#result_Dout_init = mrkteq.result_Dout_init
#result_Yout_init = mrkteq.result_Yout_init
#PC_guess_init = mrkteq.PC_guess_init
#wedge = mrkteq.wedge
#priceshifterupdate = mrkteq.priceshifterupdate
#fossilsales = mrkteq.fossilsales
#p_E_init = mrkteq.p_E_init
#laboralloc = mrkteq.laboralloc

# this will conflict with other GRID run data; and Market is only solved once so it makes no sense
"""matwrite("G/w_guess_mat.mat", Dict("w_guess" => w_guess))
matwrite("G/p_E_guessmat.mat", Dict("p_E_init" => p_E_init))
matwrite("G/Dout_guess_init.mat", Dict("result_Dout_init" => result_Dout_init))
matwrite("G/Yout_guess_init.mat", Dict("result_Yout_init" => result_Yout_init))
matwrite("G/PC_guess_init.mat", Dict("PC_guess_init" => PC_guess_init))
matwrite("G/laboralloc_guess.mat", Dict("laboralloc" => laboralloc))
matwrite("G/z_mat.mat", Dict("Z" => Z))
matwrite("G/z_sec_mat.mat", Dict("Zsec" => Zsec))
matwrite("G/wedge_vec.mat", Dict("wedge" => wedge))
matwrite("G/priceshifterupdate_vec.mat", Dict("priceshifterupdate" => priceshifterupdate))
matwrite("G/fossilsales_guess.mat", Dict("fossilsales" => fossilsales))"""

end