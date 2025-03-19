module ParamsFunctions

export load_parameters_csv, create_params!, fill_params, tS!, tW!

# export structs for type stability when using in functions
export StructParams

# import functions from packages
import CSV: CSV
import DataFrames: DataFrame
import MAT: matopen
import Tables: Tables
import Distributions: rand, Normal

function load_parameters_csv(D::String)
    # load CSV files
    regions = CSV.File("$D/ModelDataset/Regions_sorted.csv", header=true) |> DataFrame
    replace!(regions.solar_mean, "NA" => "0.0")
    regions.solar_mean = parse.(Float64, regions.solar_mean)

    majorregions = CSV.File("$D/ModelDataset/majorregions.csv") |> DataFrame
    majorregions.rowid2 = [1; majorregions.rowid[1:end-1] .+ 1]

    Linecounts = CSV.File("$D/ModelDataset/Linecountsregion.csv") |> DataFrame
    Linecounts.rowid2 = [1; Linecounts.rowid[1:end-1] .+ 1]
    return regions, majorregions, Linecounts
end

# create a struct for params
mutable struct StructParams
    Z::Matrix{Float64}
    zsector::Matrix{Float64}
    I::Int
    tau::Vector{Matrix{Float64}}
    tau1s::Matrix{Float64} 
    Vs::Matrix{Float64}
    betaS::Matrix{Float64}
    tol::Float64
    J::Int
    sig::Int
    L::Matrix{Float64}
    N::Int
    alpha1::Float64
    alpha2::Float64
    vE::Float64
    vF::Float64
    vL::Float64
    vK::Float64
    cdc::Float64
    kappa::Matrix{Float64}
    psi::Float64
    r_K::Matrix{Float64}
    T::Int
    beta::Float64
    iota::Float64
    gammaS::Float64
    gammaW::Float64
    gammaB::Float64
    delta::Float64
    deltaR::Float64
    deltaP::Float64
    deltaB::Float64
    varrho::Float64
    Rweight::Float64
    K::Int
    renewablecostscaler::Int


    # Inner Constructor
    StructParams() = new(
        Matrix{Float64}(undef, 2531, 1),                     # Pre-allocated Z
        Matrix{Float64}(undef, 2531, 10),                    # Pre-allocated Zsector
        10,                                                  # Set I
        [Matrix{Float64}(undef, 2531, 2531) for _ in 1:10],  # Pre-allocated tau vector
        Matrix{Float64}(undef, 2531, 2531),                  # Pre-allocated tau1s
        Matrix{Float64}(undef, 10, 4),                       # pre-allocated Vs
        Matrix{Float64}(undef, 2531, 10),                    # pre-allocated betaS
        0.0,                                                 # tol
        0,                                                   # J
        0,                                                   # sig
        Matrix{Float64}(undef, 2531, 1),                     # pre-allocated L
        0,                                                   # N
        0.0,                                                 # alpha1
        0.0,                                                 # alpha2
        0.0,                                                 # vE
        0.0,                                                 # vF
        0.0,                                                 # vL
        0.0,                                                 # vK
        0.0,                                                 # cdc
        Matrix{Float64}(undef, 2531, 1),                     # kappa
        0.0,                                                 # psi
        Matrix{Float64}(undef, 2531, 1),                     # pre-allocated r_K   
        0,                                                   # T
        0.0,                                                 # beta
        0.0,                                                 # iota
        0.0,                                                 # gammaS
        0.0,                                                 # gammaW
        0.0,                                                 # gammaB
        0.0,                                                 # delta
        0.0,                                                 # deltaR
        0.0,                                                 # deltaP
        0.0,                                                 # deltaB
        0.0,                                                 # varrho
        0.0,                                                 # Rweight
        0,                                                   # K
        0                                                    # renewablecostscaler
    )
end

# create a function to fill params
function create_params!(params::StructParams, z_path::String, zsector_path::String, tau_paths::Vector{String}, 
                        regions::DataFrame, majorregions::DataFrame, Linecounts::DataFrame, D::String)
    # Load Z from its file and fill the pre-allocated matrix
    z_file = matopen(z_path)
    params.Z .= read(z_file, "Z")::Matrix{Float64}
    close(z_file)
    

    # Load Zsector from its file and fill the pre-allocated matrix
    zsector_file = matopen(zsector_path)
    params.zsector .= read(zsector_file, "Zsec")::Matrix{Float64}
    close(zsector_file)
    
    # initiate I
    params.I = 10           # industries

    # Load each tau matrix from its respective file into the pre-allocated matrices
    for i in 1:10
        tau_file = matopen(tau_paths[i])
        params.tau[i] .= read(tau_file, "tau_export")::Matrix{Float64}
        close(tau_file)
    end
    # only way to improve the speed of this query is to use a different file type (HDF5)

    params.tau1s .= params.tau[end]

    # load in sectoral shares
    secshares = CSV.File("$D/ModelDataset/secshares.csv", select=[2, 3]) |> DataFrame
    for i in 1:10
        params.Vs[i, 2] = secshares[i, 1] - 1e-15                                    # vE
        params.Vs[i, 3] = secshares[i, 2] - 1e-15                                    # vF
        params.Vs[i, 4] = 0.3
        params.Vs[i, 1] = 1 - (secshares[i, 1] + secshares[i, 2]) - params.Vs[i, 4]  # vL
    end

    # expenditure shares
    #sectoralconshares = CSV.File("$D/ModelDataset/expshare.csv", select=3:12) |> Tables.matrix
    #params.betaS .= sectoralconshares
    params.betaS .= CSV.File("$D/ModelDataset/expshare.csv", select=3:12) |> Tables.matrix

    ## programer Parameters
    params.tol = 1e-4

    ## baseline parameters

    # regions and consumer / workers
    params.J = size(regions, 1)
    params.sig = 5
    params.L .= regions[!,"pop_sum"] ./ regions[!, "pop_sum"][1]
    params.N = size(majorregions, 1)

    # fossil fuel production
    params.alpha1 = 0.3
    params.alpha2 = 0.7    # capital share

    # final good production input shares
    params.vE = 0.03 - 1e-15
    params.vF = 0.03 - 1e-15
    params.vL = 0.94 - 1e-15
    params.vK = 1 - params.vL - params.vF - params.vE + 3e-15
    params.cdc = params.vL^params.vL * params.vE^params.vE * params.vF^params.vF
    params.kappa .= 2.0   # parameter on electricity usefulness in direct fossil fuel
    params.psi = 1.4     # elasticity between electricity and fossil fuels substitution
    params.r_K .= 1.0     # rent paid by production capital

    # growth parameters
    params.T = 10          # number of periods
    params.beta = 0.95     # discount factor
    params.iota = 0.99     # forgetting rate

    params.gammaS = 0.35   # learning rate of learning-by-doing formulation for solar
    params.gammaW = 0.2    # learning rate of learning-by-doing formulation for wind
    params.gammaB = 0.35   # learning rate of learning-by-doing formulation for batteries


    params.delta = 0.03    # depreciation rate on fossil capital
    params.deltaR = 0.03   # depreciation rate on renewable capital
    params.deltaP = 0.05   # depreciation rate on production capital
    params.deltaB = 0.05   # depreciation rate on batteries

    # renewable potential
    params.varrho = 0.7

    # resistance weight for B Matrix
    params.Rweight = 1.5
    
    # line counts
    params.K = Linecounts[end, end-1]

    # renewable cost scalar
    params.renewablecostscaler = 6


end


# fill params
function fill_params(regions::DataFrame, majorregions::DataFrame, Linecounts::DataFrame, D::String, G::String)
    z_path = "$G/z_mat.mat"
    zsector_path = "$G/z_sec_mat.mat"
    tau_paths = ["$D/ModelDataset/tau/tau_ind1.mat", "$D/ModelDataset/tau/tau_ind2.mat",
                "$D/ModelDataset/tau/tau_ind3.mat", "$D/ModelDataset/tau/tau_ind4.mat",
                "$D/ModelDataset/tau/tau_ind5.mat", "$D/ModelDataset/tau/tau_ind6.mat", 
                "$D/ModelDataset/tau/tau_ind7.mat", "$D/ModelDataset/tau/tau_ind8.mat",
                "$D/ModelDataset/tau/tau_ind9.mat", "$D/ModelDataset/tau/tau_ind10.mat"]
    params = StructParams()
    create_params!(params, z_path, zsector_path, tau_paths, regions, majorregions, Linecounts, D)
    return params
end

function tS!(thetaS::Vector, capacityfactorS::Int, regions::DataFrame)
    thetaS .= regions.solar_mean ./ regions.solar_mean[1]
    thetaS .= capacityfactorS .* thetaS
    thetaS .= coalesce.(thetaS, 0.2)
end

function tW!(thetaW::Vector, capacityfactorW::Int, regions::DataFrame)
    thetaW .= regions.wind_mean ./ regions.solar_mean[1] # is this supposed to be divied by solar_mean?
    thetaW .= capacityfactorW .* thetaW
end 

end