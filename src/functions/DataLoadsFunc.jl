module DataLoadsFunc

# import functions from packages
import CSV: CSV
import DataFrames: DataFrame
import LinearAlgebra: Diagonal
import Tables: Tables


# load functions
using ..DataAdjustments


export load_csv_data, w_i!, create_RWParams!, fill_RWParams, sec_shares!,
        create_FFsupplyCurves, fill_FFsupply, create_StructGsupply!,
        fill_Gsupply, linear_map, create_curtmat!, battery_req!

# export structs for type stability when used in functions
export StructRWParams

# load functions to process data
mutable struct StructRWParams
    Kmatrix::Matrix{Float64}
    A::Vector{Matrix{Float64}}
    Adash::Vector{Matrix{Float64}}
    O::Vector{Matrix{Float64}}
    Gam::Vector{Matrix{Float64}}
    Gam2::Vector{Matrix{Float64}}
    Gam3::Vector{Matrix{Float64}}
    B::Vector{Matrix{Float64}}
    KF::Matrix{Float64}
    maxF::Matrix{Float64}
    KR::Matrix{Float64}
    Zmax::Matrix{Float64}
    thetaS::Matrix{Float64}
    thetaW::Matrix{Float64}
    scalar::Int
    Countryshifter::Matrix{Float64}
    costshifter::Matrix{Float64}
    KP::Matrix{Float64}
    z::Matrix{Float64}
    zS::Matrix{Float64}

    # Inner Constructor
    function StructRWParams()
        s1 = [(1466, 726), (1432, 754), (59, 29), (114, 52), 
        (14, 12), (23, 14), (76, 45), (10, 10), 
        (42, 25), (157, 77), (247, 124), (738, 319)]

        s2 = [(1466, 1466), (1432, 1432), (59, 59), (114, 114), 
        (14, 14), (23, 23), (76, 76), (10, 10), 
        (42, 42), (157, 157), (247, 247), (738, 738)]

        s3 = [(726, 726), (754, 754), (29, 29), (52, 52), 
        (12, 12), (14, 14), (45, 45), (10, 10), 
        (25, 25), (77, 77), (124, 124), (319, 319)]
        
        new(
        Matrix{Float64}(undef, 4382, 2531),             # KMatrix
        [Matrix{Float64}(undef, size...) for size in s1], # A
        [Matrix{Float64}(undef, size...) for size in s1], # Adash
        [Matrix{Float64}(undef, size...) for size in s2], # O
        [Matrix{Float64}(undef, size...) for size in s1], # Gam
        [Matrix{Float64}(undef, size...) for size in s1], # Gam2
        [Matrix{Float64}(undef, size...) for size in s1], # Gam3
        [Matrix{Float64}(undef, size...) for size in s3], # B
        Matrix{Float64}(undef, 2531, 1),                  # KF
        Matrix{Float64}(undef, 2531, 1),                  # maxF
        Matrix{Float64}(undef, 2531, 1),                  # KR
        Matrix{Float64}(undef, 4382, 1),                  # Zmax
        Matrix{Float64}(undef, 2531, 1),                  # thetaS
        Matrix{Float64}(undef, 2531, 1),                  # thetaW
        0,                                                # scalar
        Matrix{Float64}(undef, 2531, 1),                  # Countryshifter
        Matrix{Float64}(undef, 2531, 1),                  # costshifter
        Matrix{Float64}(undef, 2531, 1),                  # KP
        Matrix{Float64}(undef, 2531, 1),                  # z 
        Matrix{Float64}(undef, 2531, 10)                  # zS
        )
    end
end

struct StructFFsupplyCurves
    Q::Matrix{Float64}
    P::Matrix{Float64}

    function StructFFsupplyCurves()
        new(
            # initiate P and Q with zeroes
            zeros(Float64, 15204, 16), 
            zeros(Float64, 15204, 16)
        )
    end

end

mutable struct StructGsupply
    Q::Matrix{Float64}
    P::Matrix{Float64}

    function StructGsupply()
        new(
            Matrix{Float64}(undef, 67518, 1),
            Matrix{Float64}(undef, 67518, 1)
        )
    end
end

function load_csv_data(D::String)
    regions_all = CSV.File("$D/ModelDataset/Regions_sorted.csv", header=true) |> DataFrame
    Linedata = CSV.File("$D/ModelDataset/Linedata.csv") |> DataFrame
    majorregions_all = CSV.File("$D/ModelDataset/majorregions.csv") |> DataFrame
    Fossilcapital = CSV.File("$D/ModelDataset/aggcap_ffl.csv")  |> DataFrame
    Renewablecapital = CSV.File("$D/ModelDataset/aggcap_rnw.csv") |> DataFrame
    Electricityregionprices = CSV.File("$D/ModelDataset/Region_prices.csv") |> DataFrame
    return regions_all, Linedata, majorregions_all, Fossilcapital, Renewablecapital, Electricityregionprices
end

# initiate functions to calculate variables
function w_i!(wage_init::Vector{Float64}, regions::DataFrame)
    wage_init .= regions.wages ./ (regions.wages[1])
    coalesce.(wage_init, 1.0)
    clamp!(wage_init, -Inf, 2.0)
    #wage_init[wage_init .> 2] .= 2.0
    return wage_init
end

function create_RWParams!(RWParams::StructRWParams, majorregions_all::DataFrame, majorregions::DataFrame, regions::DataFrame, Linedata::DataFrame, params, wage_init::Vector{Float64},
    thetaS::Vector{Float64}, thetaW::Vector{Float64}, popelas::Float64, rP_LR, D::String)
    # create kmatrix
    Kmatrix = Matrix(CSV.File("$D/ModelDataset/Kmatrix.csv", drop=[1]) |> DataFrame)
    RWParams.Kmatrix .= Matshifter(Kmatrix)

    # create A
    Kmx = Matrix(CSV.File("$D/ModelDataset/Kmatrix_1.csv") |> DataFrame)[:, 2:majorregions_all.rowid[1]]
    RWParams.A[1] .= Matshifter(Kmx)
    RWParams.Adash[1] .= Matshifter(Kmx)

    # create Omatrix
    Omatrix = CSV.File("$D/ModelDataset/Omatrix_1.csv", drop=[1]) |> DataFrame
    Omatrix = Vector(Omatrix[!,1])
    RWParams.O[1] .= Diagonal(Omatrix)

    # create Gam
    RWParams.Gam[1] .= RWParams.O[1]^(-1) * RWParams.A[1] * inv(RWParams.A[1]' * RWParams.O[1]^(-1) * RWParams.A[1])
    RWParams.Gam2[1] .= RWParams.Gam[1]

    row_maxima = vec(maximum(RWParams.Gam[1], dims=2))
    num_repeats = majorregions_all.rowid[1] - 1
    indmax = repeat(row_maxima, 1, num_repeats)

    RWParams.Gam2[1][RWParams.Gam2[1] .< indmax] .= 0

    RWParams.Gam3[1] .= RWParams.Gam[1]
    row_min = vec(minimum(RWParams.Gam[1], dims=2))
    indmin = repeat(row_min, 1, num_repeats)
    RWParams.Gam3[1][RWParams.Gam3[1] .> indmin] .= 0

    # fill the rest of A, Adash, O in a for loop
    for jj in 2:(size(majorregions_all, 1)-1)
    #stringer = "$D/ModelDataset/Kmatrix_$(jj).csv"
    #stringer2 = "$D/ModelDataset/Omatrix_$(jj).csv"
    #Kmatrix = Matrix(CSV.File(stringer) |> DataFrame)
    Kmatrix = CSV.File("$D/ModelDataset/Kmatrix_$(jj).csv") |> Tables.matrix
    Kmatrix = Kmatrix[:, majorregions_all.rowid[jj-1] + 2 : majorregions_all.rowid[jj]]

    #Omatrix = Matrix(CSV.File(stringer2, drop=[1]) |> DataFrame)
    Omatrix = CSV.File("$D/ModelDataset/Omatrix_$(jj).csv", drop=[1]) |> Tables.matrix
    RWParams.A[jj] .= Matshifter(Kmatrix)
    RWParams.Adash[jj] .= Matshifter(Kmatrix)
    Omatrix = Vector(Omatrix[:,1])
    RWParams.O[jj] .= Diagonal(Omatrix)
    RWParams.Gam[jj] .= RWParams.O[jj]^(-1) * RWParams.A[jj] * inv(RWParams.A[jj]' * RWParams.O[jj]^(-1) * RWParams.A[jj])


    indmax =  repeat(maximum(RWParams.Gam[jj], dims=2), 1, majorregions_all.n[jj] - 1)
    RWParams.Gam2[jj] .= RWParams.Gam[jj]
    RWParams.Gam2[jj][RWParams.Gam[jj] .< indmax] .= 0


    indmin = repeat(minimum(RWParams.Gam[jj], dims=2), 1, majorregions_all.n[jj] - 1)
    RWParams.Gam3[jj] .= RWParams.Gam[jj]
    RWParams.Gam3[jj][RWParams.Gam3[jj] .> indmin] .= 0   
    end

    R = size(majorregions, 1) - 1   # regions
    I = 10                          # industries

    # create B 
    for jj in 1:size(majorregions, 1) - 1
    RWParams.B[jj] .= inv(RWParams.A[jj]' * RWParams.O[jj] * RWParams.A[jj])
    end

    # fossil fuel capital
    RWParams.KF .= regions.ffl_capacity_mw ./ regions.ffl_capacity_mw[1]
    RWParams.KF .+= 10.0^(-1)

    # set max fossil fuel use to capacity
    RWParams.maxF .= RWParams.KF

    # renewable capital
    RWParams.KR .= regions.rnw_capacity_mw ./ regions.ffl_capacity_mw[1]

    # max line flows
    RWParams.Zmax .= Linedata.lcap ./ regions.ffl_capacity_mw[1]

    # renewable potential
    RWParams.thetaS .= coalesce.(thetaS, 0.2) # set places without sunlight data to very low
    replace!(RWParams.thetaS, 0 => 0.2)
    RWParams.thetaW .= coalesce.(thetaW, 0.2) # set places without wind data to very low

    #shift potential so that urban regions don't install lots of capital, then
    #scale by relative country costshifters
    RWParams.scalar = params.renewablecostscaler
    RWParams.Countryshifter .= regions[!, :costrel]

    # define costshifter
    for kk in 1:params.N
    range = majorregions[!, :rowid2][kk]:majorregions[!, :rowid][kk]

    RWParams.costshifter[range] = (
    (regions[!, :pop_dens][range] .^ popelas) ./ 
    (regions[!, :pop_dens][majorregions[!, :rowid2][kk]] .^ popelas)
    ) .* RWParams.scalar .* RWParams.Countryshifter[range]
    end

    # production capital
    RWParams.KP .= regions[!, :capitalperworker]
    R_normaliser = wage_init ./ (RWParams.KP .* rP_LR)
    RWParams.KP .= RWParams.KP .* R_normaliser

    RWParams.z .= params.Z
    RWParams.zS .= repeat(params.Z, 1, 10)

end

function fill_RWParams(majorregions_all::DataFrame, majorregions::DataFrame, regions::DataFrame, Linedata::DataFrame, params, wage_init::Vector{Float64}, thetaS::Vector{Float64}, 
    thetaW::Vector{Float64}, popelas::Float64, rP_LR::Float64, D::String)
    RWParams = StructRWParams()
    create_RWParams!(RWParams, majorregions_all, majorregions, regions, Linedata, params, wage_init, thetaS, thetaW, popelas, rP_LR, D)
    return RWParams
end

function sec_shares!(secshares::Matrix, sectoralempshares::Matrix{Union{Missing, Float64}}, D::String)
    secshares_df = CSV.File("$D/ModelDataset/secshares.csv") |> DataFrame
    secshares .= Matrix{Float64}(secshares_df[:, 2:3])

    sectoralempshares_df = CSV.read("$D/ModelDataset/locationempshares.csv", DataFrame)
    sectoralempshares .= Matrix{Union{Missing, Float64}}(sectoralempshares_df)
    sectoralempshares .= coalesce.(sectoralempshares, 0.0)
    return secshares, sectoralempshares
end

function create_FFsupplyCurves(FFsupplyCurves::StructFFsupplyCurves, D::String)

    countriesCurves = CSV.File("$D/FossilFuels/country_curves.csv") |> DataFrame
    countries = unique(countriesCurves[:, :region_name])
    totalCountries = length(countries)
    maxPoints = 15202

    Q = zeros(15204, 16)
    P = zeros(15204, 16)

    for country = 1:totalCountries
        country_name = countries[country]
        newQ = countriesCurves.Q[countriesCurves.region_name .== country_name]
        newP = countriesCurves.P_smooth[countriesCurves.region_name .== country_name]

        if length(newQ) < maxPoints
            newQ = vcat(newQ, fill(newQ[end], maxPoints - length(newQ)))
            newP = vcat(newP, fill(newP[end], maxPoints - length(newP)))
        end

        Q[2:end-1, country] .= newQ
        P[2:end-1, country] .= newP

    end

    P[end, :] .= P[end-1, :] .* (1000)
    P .= P ./ 100

    Q[end, :] .= Q[end-1,:] .* 1.001
    Q .= Q .* 100000

    FFsupplyCurves.Q .= Q
    FFsupplyCurves.P .= P

    return FFsupplyCurves

end

function fill_FFsupply(D::String)
    FFsupplyCurves = StructFFsupplyCurves()
    create_FFsupplyCurves(FFsupplyCurves, D)
    return FFsupplyCurves
end

function create_StructGsupply!(GsupplyCurves::StructGsupply, D::String)

    globalCurve = CSV.File("$D/FossilFuels/global_curve.csv") |> DataFrame

    GsupplyCurves.Q[1:67517, 1] .= Vector{Float64}(globalCurve[:, :Q]) .* 100000
    GsupplyCurves.P[1:67517, 1] .= Vector{Float64}(globalCurve[:, :P_smooth]) ./ 200

    GsupplyCurves.P[67518, 1] = GsupplyCurves.P[67517, 1] * 1000
    GsupplyCurves.Q[67518, 1] = GsupplyCurves.Q[67517, 1] * 1.001

end

function fill_Gsupply(D::String)
    GsupplyCurves = StructGsupply()
    create_StructGsupply!(GsupplyCurves, D)
    return GsupplyCurves
end

function linear_map()
    maxstorage=12
    storepoints=100
    storagey = range(0, stop=maxstorage, length=storepoints)
    storagex = range(0, stop=1, length=storepoints)
    return storagey, storagex
end

function create_curtmat!(curtmatno::Matrix, curtmat4::Matrix, curtmat12::Matrix, curtmat::Array, D::String)
    curtmatno .= DataFrame(CSV.File("$D/CurtailmentUS/heatmap_us_mat_nostorage.csv", header = false)) |> Matrix
    curtmat4 .= DataFrame(CSV.File("$D/CurtailmentUS/heatmap_us_mat_4hour.csv", header = false)) |> Matrix
    curtmat12 .= DataFrame(CSV.File("$D/CurtailmentUS/heatmap_us_mat_12hour.csv", header = false)) |> Matrix

    n = size(curtmatno, 1)
    x = range(start = 0.0, stop = 1.0, length=n)  # Change these to reflect your actual coordinate grids
    y = range(start = 0.0, stop = 1.0, length=n) 
    z = range(start = 0.0, stop = 12.0, length=3)

    samplepointssolar = [xi for yj in y, xi in x, zk in z]
    samplepointswind = [yj for yj in y, xi in x, zk in z]
    samplepointsbat = [zk for yj in y, xi in x, zk in z]
    #(samplepointssolar, samplepointswind, samplepointsbat) = ndgrid(x, y, z)

    # fill the NaN border cells with the same value

    for i = 2:size(curtmatno, 1)
        curtmatno[size(curtmatno,1)-i+2, i] = curtmatno[size(curtmatno,1)-i+1, i]
        curtmat4[size(curtmat4,1)-i+2, i] = curtmat4[size(curtmat4,1)-i+1, i]
        curtmat12[size(curtmat12,1)-i+2, i] = curtmat12[size(curtmat12,1)-i+1, i]
    end

    curtmat = cat(curtmatno, curtmat4, curtmat12, dims = 3)
    return curtmat, samplepointssolar, samplepointswind, samplepointsbat
end

function battery_req!(batteryrequirements::Matrix, D::String)
    batteryrequirements_df = DataFrame(CSV.File("$D/CurtailmentUS/Curtailment_vs_Battery.csv")) |> Matrix
    batteryrequirements[1:end-1, :] .= batteryrequirements_df[:, [1, end]]
    batteryrequirements[15, 1] = 12
    batteryrequirements[15, 2] = 100
    broadcast!(x -> x / 100, batteryrequirements[:, 2], batteryrequirements[:, 2])
    #batteryrequirements[:, 2] .= batteryrequirements[:, 2] ./ 100
    return batteryrequirements
end


end