module WriteDataBattery

# import package 
import DelimitedFiles: writedlm
import ..ModelConfiguration: ModelConfig

# import data from model
export writedata_battery

"""
    writedata_battery(P::NamedTuple, M::NamedTuple, S::NamedTuple, T::NamedTuple, config::ModelConfig, R::String)

Writes data outputs to .csv files for analysis. All outputs are in the Results folder

## Inputs
- `P::NamedTuple` -- NamedTuple of parameters. Output of `P = setup_parameters(D, G)`
- `D::NamedTuple` -- NamedTuple of model data. Output of `DL = load_data(P, D)`
- `M::NamedTuple` -- NamedTuple of market equilibrium. Output of `M = solve_market(P, DL, config, G)`
- `S::NamedTuple` -- NamedTuple of steady state equilibrium. Output of `S = solve_steadystate(P, DL, M, config, Guesses)`
- `T::NamedTuple` -- NamedTuple of transition outputs. Output of `T = solve_transition(P, DL, M, S, Subsidy, config, Guesses)`
- `config::ModelConfig` -- struct of user defined model configurations. `config = ModelConfig()`
- `R::String` -- path to Results folder. `R = "path/to/Results"`

## Outputs
Model results for capital and battery price falls, renewable shares, US GDP outcomes, capital investment results,
    price results, fossil fuel usage and prices, welfare changes in 2040. Clearly labeled .csv files in Results.

## Notes
This function writes data when RunBattery==1 or RunCurtailment==1.
"""
function writedata_battery(P::NamedTuple, M::NamedTuple, S::NamedTuple, T::NamedTuple, config::ModelConfig, R::String)

    curtailmentswitch = P.curtailmentswitch
    hoursofstorage = config.hoursofstorage
    
    if config.RunCurtailment == 1
        curtailmentswitch = 1
    end

    # initialize data
    yearindex_cap = Vector{Int64}(undef, 20)
    yearindex_share = Vector{Int64}(undef, 30)
    yearindex_subsidy = Vector{Int64}(undef, 12)
    capitalpricefall = Vector{Float64}(undef, 20)
    solarpricefall = Matrix{Float64}(undef, 20, 1)
    windpricefall = Matrix{Float64}(undef, 20, 1)
    capprice = Matrix{Float64}(undef, 20, 4)
    pricecsv = Matrix{Float64}(undef, 2531, 2)
    GDPUS = Matrix{Float64}(undef, 1, 501)
    G = Matrix{Float64}(undef, 2531, 501)
    sharepath = Matrix{Float64}(undef, 30, 16)

    labeller = "hours$(lpad(hoursofstorage, 2, '0'))_curt_$(lpad(curtailmentswitch, 2, '0'))"

    # initialize year indices
    yearindex_cap .= collect(1:20) .+ 2020
    yearindex_share .= collect(1:30) .+ 2020
    yearindex_subsidy .= collect(1:12) .+ 2021


    # capital price falls 
    capitalpricefall .= (T.transeq.p_KR_bar_path[1, 1:20] ./ T.transeq.p_KR_bar_path[1, 1]) .* 100
    solarpricefall .=  (T.transeq.p_KR_path_S[:, 1:20]' ./ T.transeq.p_KR_path_S[:, 1]) .* 100
    windpricefall .= (T.transeq.p_KR_path_W[:, 1:20]' ./ T.transeq.p_KR_path_W[:,1]) .* 100
    capprice[:, 1] .= yearindex_cap
    capprice[:, 2] .= capitalpricefall
    capprice[:, 3] .= solarpricefall
    capprice[:, 4] .= windpricefall
    writedlm("$R/Capital_prices$(labeller).csv", capprice, ",")

    # battery price falls
    batterypricefall = (T.transeq.p_B_path_guess[:, 1:20]' ./ T.transeq.p_B_path_guess[:, 1]) .* 100
    batprice = [yearindex_cap batterypricefall]
    writedlm("$R/Battery_prices$(labeller).csv", batprice, ",") 

    # renewable shares
    sharepath[:, 1] .= yearindex_share
    sharepath[:, 2:14] .= 100 .* T.renewshare_path_region[:, 1:30]'
    sharepath[:, 15] .= 100 .* T.renewshareUS[1:30]
    sharepath[:, 16] .= 100 .* T.renewshare_path_world[:, 1:30]'
    writedlm("$R/Renewable_share$(labeller).csv", sharepath, ",")

    # write price results
    pricecsv .= [M.priceresults P.regions.csr_id]
    writedlm("$R/pricecsv$(labeller).csv", pricecsv, ",")

    # write GDP results
    G .= T.transeq.w_path_guess .* P.params.L ./ T.transeq.PC_path_guess
    GDPUS .= sum(G[1:743, :], dims = 1)
    GDPUS .= GDPUS ./ GDPUS[1]
    writedlm("$R/GDPUS$(labeller).csv", GDPUS, ",")

    # write capital investment results
    capitalinvestment = Matrix{Float64}(undef, 2531, 502)
    capitalinvestment .= [P.regions.csr_id T.transeq.KR_path]
    writedlm("$R/capitalinvestment$(labeller).csv", capitalinvestment, ",")

    # write price results
    pricepath = Matrix{Float64}(undef, 2531, 502)
    pricepath .= [P.regions.csr_id T.transeq.p_E_path_guess] 
    writedlm("$R/pricepath$(labeller).csv", pricepath, ",")

    # write fossil fuel usage
    fosspath = Matrix{Float64}(undef, 30, 2)
    fosspath .=[yearindex_share T.transeq.fusage_total_path[1:30]]
    writedlm("$R/Fossil_usage$(labeller).csv", fosspath, ",")

    # write fossil fuel price
    fosspath .= [yearindex_share T.transeq.p_F_path_guess[1:30]]
    writedlm("$R/Fossil_price$(labeller).csv", fosspath, ",")

    # write welfare changes
    welfare = Matrix{Float64}(undef, 2531, 5)
    welfare .= [P.regions.csr_id S.welfare_wagechange S.welfare_capitalchange S.welfare_electricitychange S.welfare_fossilchange]
    writedlm("$R/welfare.csv", welfare, ",")

    welfare_2040 = Matrix{Float64}(undef, 2531, 5)
    welfare_2040 .= [P.regions.csr_id T.welfare_wagechange_2040 T.welfare_capitalchange_2040 T.welfare_electricitychange_2040 T.welfare_fossilchange_2040]
    writedlm("$R/welfare_2040$(labeller).csv", welfare_2040, ",")
end

end