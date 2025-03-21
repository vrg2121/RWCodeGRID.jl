# Parameterize Run
module ModelConfiguration

export ModelConfig

struct ModelConfig
    RunTransition::Int64
    RunBatteries::Int64
    RunExog::Int64
    RunCurtailment::Int64
    Transiter::Int64
    Initialprod::Int64
    hoursofstorage::Int64
    hoursvec::Vector{Int64}
    
end


end