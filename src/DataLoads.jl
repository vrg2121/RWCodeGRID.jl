module DataLoads

# export data
export load_data

# load functions
using ..DataLoadsFunc, ..DataAdjustments

# load functions from packages
import CSV: CSV
import DataFrames: DataFrame


function load_data(P::NamedTuple, D::String)
    # initialize data
    wage_init = Vector{Float64}(undef, 2531)
    secshares = Matrix{Float64}(undef, 10, 2)
    sectoralempshares = Matrix{Union{Float64, Missing}}(undef, 2531, 11)
    curtmatmx = Matrix{Float64}(undef, 21, 21)
    curtmat = zeros(21, 21, 3)
    batteryrequirements = Matrix{Float64}(undef, 15, 2)


    # ---------------------------------------------------------------------------- #
    #                                   Load Data                                  #
    # ---------------------------------------------------------------------------- #

    # load global csv data

    regions_all, Linedata, majorregions_all, Fossilcapital, Renewablecapital, Electricityregionprices = load_csv_data(D)

    # initiate wage
    wage_init = w_i!(wage_init, P.regions)

    # long run interest rate
    R_LR = 1/P.params.beta

    rP_LR = R_LR - 1 + P.params.deltaB    # long run return on production capital
    # this variable is calculated in SteadyState.jl differently. This one is only used locally to this file

    # create RWParams
    RWParams = fill_RWParams(majorregions_all, P.majorregions, P.regions, Linedata, P.params, wage_init, P.thetaS, P.thetaW, P.popelas, rP_LR, D);
    ## right now RWParams is stored in the heap and accessed by a chain of addresses. 
    ## Alternative method to store in stack using immutable struct is in rwparams_notes.jl 

    # load in sectoral shares

    secshares, sectoralempshares = sec_shares!(secshares, sectoralempshares, D)

    # ---------------------------------------------------------------------------- #
    #                             Load Fossil Fuel Data                            #
    # ---------------------------------------------------------------------------- #

    # FFsupplyCurves
    FFsupplyCurves = fill_FFsupply(D)

    # GsupplyCurves
    GsupplyCurves = fill_Gsupply(D)

    P.regions.reserves .= P.regions.reserves ./ sum(P.regions.reserves)

    # ---------------------------------------------------------------------------- #
    #                            Convert to Model Inputs                           #
    # ---------------------------------------------------------------------------- #

    # regionparams
    regionParams = deepcopy(RWParams)

    # initial renewable capital
    KR_init_S = RWParams.KR ./ 2
    KR_init_W = RWParams.KR ./ 2

    # ---------------------------------------------------------------------------- #
    #                        LOAD EXOGENOUS TECH PROJECTIONS                       #
    # ---------------------------------------------------------------------------- #

    projectionswind = DataFrame(CSV.File("$D/ModelDataset/Windprojectionsdata.csv")) |> Matrix
    projectionssolar = DataFrame(CSV.File("$D/ModelDataset/Solarprojectionsdata.csv")) |> Matrix

    # ---------------------------------------------------------------------------- #
    #                             LOAD CURTAILMENT DATA                            #
    # ---------------------------------------------------------------------------- #

    # linear map
    # storagey, storagex = linear_map()
    # have not seen either variable called anwhere else

    # create curtmat

    curtmat, samplepointssolar, samplepointswind, samplepointsbat = create_curtmat!(curtmatmx, curtmatmx, curtmatmx, curtmat, D);

    # import battery requirements
    batteryrequirements = battery_req!(batteryrequirements, D)

    return(
        RWParams = RWParams,
        regionParams = regionParams,
        FFsupplyCurves = FFsupplyCurves,
        GsupplyCurves = GsupplyCurves,
        projectionswind = projectionswind,
        projectionssolar = projectionssolar,
        curtmat = curtmat,
        batteryrequirements = batteryrequirements,
        sectoralempshares = sectoralempshares,
        samplepointssolar = samplepointssolar,
        samplepointswind = samplepointswind,
        samplepointsbat = samplepointsbat,
        R_LR = R_LR, 
        wage_init = wage_init, 
        KR_init_S = KR_init_S, 
        KR_init_W = KR_init_W
        )
end
end