module WriteDataExog
import DelimitedFiles: writedlm
using Printf
using CSV
using DataFrames

"""
    writedata_exog(TE::NamedTuple, exogindex::Int, R::String)

Writes data outputs to .csv files for analysis. All outputs are in the Results folder

## Inputs
- `TE::NamedTuple` -- NamedTuple of transition solved with exogenous tech outputs. 
    Output of `T = solve_transition(P, DL, M, S, Subsidy, config, Guesses)`
- `exogindex::Int64` -- The exogenous tech index: 1, 2, 3
- `R::String` -- path to Results folder. `R = "path/to/Results"`

## Outputs
Model results (with and without subsidy) for renewable shares when there is exogenous technology.

## Notes
This function writes data only when RunExog==1.
"""
function writedata_exog(TE::NamedTuple, exogindex::Int, R::String)
    # Constructing a label string
    labeller = "exog" * @sprintf("%02d", exogindex)

    # Generating year indices for different categories
    yearindex_share = Vector{Int64}(undef, 30)
    yearindex_share .= collect(1:30) .+ 2020

    # Creating the share path matrix
    sharepath = hcat(
        yearindex_share,
        100 .* TE.renewshare_path_region[:, 1:30]',
        100 .* TE.renewshareUS[1:30],
        100 .* TE.renewshare_path_world[:, 1:30]'
    )

    writedlm("$R/Renewable_share$(labeller).csv", sharepath, ",")

    
end

end