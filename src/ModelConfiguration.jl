# Parameterize Run
module ModelConfiguration

export ModelConfig

function prompt_for_bin(prompt, default=nothing, binary=false)
    while true
        println(prompt)
        val = readline()
        if isempty(val) && default !== nothing
            return default
        else
            try
                parsed_val = parse(Int, val)
                if binary && !(parsed_val in (0, 1))
                    println("Error: Only 0 or 1 is allowed. Please try again.")
                else
                    return parsed_val
                end
            catch
                println("Invalid input. Please enter a valid integer.")
            end
        end
    end
end

function prompt_for_int(prompt, default=nothing)
    println(prompt)
    val = readline()
    return isempty(val) && default !== nothing ? default : parse(Int, val)
end

function prompt_for_vector(prompt, default=nothing)
    println(prompt)
    val = readline()
    if isempty(val) && default !== nothing
        return default
    else
        #val = map(x -> parse(Int64, x), val)
        return parse.(Float64, split(val, ","))
    end
end


"""
    ModelConfig()

This function guides the user through creating a mutable struct with the model configurations. The outputs for this function should be stored in a variable, e.g.
    config = ModelConfig(). The first 5 inputs take binary arguments: 1, 0. The next 3 configurations take integers and the last configuration, hoursvec
    takes a list of Int64 values, e.g. 1,2,3,4,....

- RunTransition: When set to 1, the model will run a 20 year transition from 2020-2040 as wind and solar are incorporated into the global energy grid.
    The model also runs the transition with a subsidy included in the prices for the US.

- RunBatteries: When set to 1, the model will run the 20 year transition with the hours of storage for batteries set by the user in hoursvec.

- RunExog: When set to 1, the model assumes that technology is exogenous. Steady state and the transition assume that technology does not change.
    The exogenous values impact the guesses for renewable capital prices in both the long run and the transition, setting the guesses on 
    projections for wind / solar respectively.

- RunCurtailment: When set to 1, there is no subsidy during transition and battery hours of storage is assumed to be 12. This problem demonstrates
    the effects of renewable intermittency without storage.

- Transiter: This is the number of iterations the transition model will run. For academic purposes, the transition model runs for 100 iterations.
    To simply check that the code is working, set Transiter = 2.

- Initialprod: Initial level of renewables costs

- hoursofstorage: Hours of battery storage. This value set by the model in most cases, unless RunBatteries==1. Then it is set by hoursvec.

- hoursvec: When RunBatteries==1, the hoursvec is a vector of battery storage hours used as inputs for the model. This will output results for the change
    in battery prices over the transition given that price. It is useful for comparing different hours of storage.

# Examples
```jl-repl
julia> config = ModelConfig()
Enter RunTransition (0 or 1, default = 1):
0
Enter RunBatteries (0 or 1, default=0):
1
Enter RunExog (0 or 1, default=0):
0
Enter RunCurtailment (0 or 1, default=0):
0
Enter the Number of Transition Iterations (recommend 0-100 iterations, default=2):
2
Enter Initial Production (default = 100):
0
Enter hoursofstorage (default=0):
0
Enter hoursvec (comma-separated, default = 2,4,6):
1,3,5
ModelConfig(0, 1, 0, 0, 2, 0, 0, [1, 3, 5])
```
"""
mutable struct ModelConfig
    RunTransition::Int64
    RunBatteries::Int64
    RunExog::Int64
    RunCurtailment::Int64
    Transiter::Int64
    Initialprod::Int64
    hoursofstorage::Int64
    hoursvec::Vector{Int64}

    function ModelConfig()
        RunTransition = prompt_for_bin("Enter RunTransition (0 or 1, default = 1):", 1, true)
        RunBatteries = prompt_for_bin("Enter RunBatteries (0 or 1, default=0):", 0, true)
        RunExog = prompt_for_bin("Enter RunExog (0 or 1, default=0):",0, true)
        RunCurtailment = prompt_for_bin("Enter RunCurtailment (0 or 1, default=0):",0, true)
        Transiter = prompt_for_int("Enter the Number of Transition Iterations (recommend 0-100 iterations, default=2):", 2)
        Initialprod = prompt_for_int("Enter Initial Production (default = 100):", 100)
        hoursofstorage = prompt_for_int("Enter hoursofstorage (default=0):", 0)
        hoursvec = prompt_for_vector("Enter hoursvec (comma-separated, default = 2,4,6):", [2.0, 4.0, 6.0])
        
        new(RunTransition, RunBatteries, RunExog, RunCurtailment, Transiter, Initialprod, hoursofstorage, hoursvec)
    end
    
end


end