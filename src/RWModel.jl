module RWModel
export run_model, ModelConfig

# ---------------------------------------------------------------------------- #
#                         Import all Relevant Functions                        #
# ---------------------------------------------------------------------------- #

# load struct for model configurations
include("./ModelConfiguration.jl")
import .ModelConfiguration: ModelConfig

# load struct for model Params
using DrawGammas #StructAllParams, StructParams

# Load Data
include("./functions/DataAdjustments.jl")
include("./functions/DataLoadsFunc.jl")
include("./DataLoads.jl")
import .DataLoads: load_data

# Initial Equilibrium
include("./functions/MarketEquilibrium.jl")
include("./RegionModel.jl")
include("./functions/MarketFunctions.jl")
include("./Market.jl")
import .Market: solve_market

# Long Run Equilibrium
include("./functions/SteadyStateFunctions.jl")
include("./SteadyState.jl")
import .SteadyState: solve_steadystate, StructSteadyState

# Run Transitional Dynamics
include("./functions/TransitionFunctions.jl")
include("./Transition.jl")
import .Transition: solve_transition

# Write Data
include("./WriteData.jl")
import .WriteData: writedata

# -------------------------- Run Battery Transition -------------------------- #

include("./SteadyStateBat.jl")
import .SteadyStateBat: solve_steadystate_bat, StructSteadyStateBat

include("./TransitionBat.jl")
import .TransitionBat: StructTransOutputBat, solve_transition_bat

# Write Data with Battery configurations
include("./WriteDataBattery.jl")
import .WriteDataBattery: writedata_battery

# -------------------- Run Exogenous Technology Equilibria ------------------- #

# Long Run Equilibrium with Exogenous Tech
include("./functions/SteadyStateExogFunc.jl")
include("./SteadyStateExog.jl")
import .SteadyStateExog: solve_steadystate_exog, StructSteadyStateExog

# Transitional Dynamics with Exogenous Tech
include("./functions/TransitionExogFunc.jl")
include("./TransitionExog.jl")
import .TransitionExog: solve_transition_exog

# Data Outputs with Exogenous Tech
include("./WriteDataExog.jl")
import .WriteDataExog: writedata_exog


function run_model(config::ModelConfig, D::String, G::String, R::String, P::StructAllParams)

    # ---------------------------------------------------------------------------- #
    #                                  Data Loads                                  #
    # ---------------------------------------------------------------------------- #
    
    println("Loading data inputs...")
    DL = load_data(P, D); #DataLoads.jl
    # Base.summarysize(DL) = 475.74 MB

    # ---------------------------------------------------------------------------- #
    #                              Initial Equilibrium                             #
    # ---------------------------------------------------------------------------- #

    println("Solving initial equilibrium...")    
    M = solve_market(P, DL, config, G);
    
    
    # ---------------------------------------------------------------------------- #
    #                             Long Run Equilibrium                             #
    # ---------------------------------------------------------------------------- #

    println("Solving initial long run equilibrium...")
    S = solve_steadystate(P, DL, M, config, G);

    # ---------------------------------------------------------------------------- #
    #                           Run Transitional Dynamics                          #
    # ---------------------------------------------------------------------------- #
    if config.RunTransition == 1

        # ---------------------- Run Transition without Subsidy ---------------------- #
        println("Solving transitional dynamics without Subsidy...")
        Subsidy = 0
        T = solve_transition(P, DL, M, S, Subsidy, config, G)
        # transeq.p_F_path_guess updated if config.hoursofstorage == 0, input no where

        println("Writing outputs of transitional dynamics without Subsidy...")
        writedata(P, DL, M, S, T, Subsidy, config, R)
        #writedata(P, DL, M, S, T, Subsidy, config, path)

        # ------------------------ Run Transition with Subsidy ----------------------- #
        println("Solving transitional dynamics with Subsidy...")
        Subsidy = 1
        TS = solve_transition(P, DL, M, S, Subsidy, config, G)

        println("Writing outputs of transitional dynamics with Subsidy...")
        writedata(P, DL, M, S, TS, Subsidy, config, R)
        #writedata(P, DL, M, S, TS, Subsidy, config, path)

    end

    # ---------------------------------------------------------------------------- #
    #                              Battery Robustness                              #
    # ---------------------------------------------------------------------------- #
    """
    If you run config.RunTransition == 1, data will be written using writedata without Subsidy and then with Subsidy.
    If you run config.RunTransition == 1 && config.RunBatteries == 1, data will be written 
        using writedata (with subsidy, without subsidy) and then written using writedatabattery.
    """
    
    if config.RunBatteries == 1

        for bb in 1:length(config.hoursvec)         
            config.hoursofstorage = config.hoursvec[bb]
            Subsidy = 0

            println("Solving long run equilibrium when battery storage hours = $(config.hoursofstorage)...")
            SB = solve_steadystate_bat(S, P, DL, M, config, G)

            println("Solving transitional dynamics when battery storage hours = $(config.hoursofstorage)...")
            TB = solve_transition_bat(P, DL, M, SB, Subsidy, config, G)
            
            println("Writing outputs when battery storage hours = $(config.hoursofstorage)...")
            writedata_battery(P, M, SB, TB, config, R)
        end
    end

    # ---------------------------------------------------------------------------- #
    #                                Exogenous Tech                                #
    # ---------------------------------------------------------------------------- #
    if config.RunExog == 1
        config.hoursofstorage = 0
        for exogindex in 3:-1:2
            # exogindex == 1 does not converge
            println("Solving long run equilibrium with exogenous tech when exog index = $exogindex...")
            SE = SteadyStateExog.solve_steadystate_exog(P, DL, M, config, exogindex, G)


            println("Solving transitional dynamics with exogenous tech when exog index = $exogindex...")
            TE = solve_transition_exog(P, DL, M, SE, config, exogindex)
            
            println("Writing outputs with exogenous tech when exog index = $exogindex...")
            writedata_exog(TE, exogindex, R)
        end
            
    end

    # ---------------------------------------------------------------------------- #
    #                                  Curtailment                                 #
    # ---------------------------------------------------------------------------- #
    """
    User defined hours of storage is overwritten when RunCurtailment == 1.
    """
    if config.RunCurtailment == 1
        Subsidy = 0
        config.hoursofstorage = 12

        println("Solving long run equilibrium with curtailment (hours of battery storage = 12)...")
        SC = solve_steadystate(P, DL, M, config, G)

        println("Solving transitional dynamics with curtailment (hours of battery storage = 12)...")
        TC = solve_transition(P, DL, M, SC, Subsidy, config, G)

        println("Writing output for curtailment (hours of battery storage = 12)...")
        writedata_battery(P, M, SC, TC, config, R)
    end


end

# ---------------------------------------------------------------------------- #
#                         Testing on personal computer                         #
# ---------------------------------------------------------------------------- #
#using JLD2, DataFrames
#import DrawGammas: StructAllParams, StructParams
#wd = pwd()
#D = "$wd/Data"
#G = "$wd/Guesses"
#R = "$wd/Results"

# load the parameters for this task
#SGE_TASK_ID = "1"
#P = jldopen("$D/Params/P_$SGE_TASK_ID.jld2", "r")["P"];

#println("Number of threads on task $SGE_TASK_ID = ", Threads.nthreads())
#config = ModelConfig(1, 1, 0, 0, 2, 100, 0, [2.0, 4.0, 6.0])

#run_model(config, D, G, R, P)

# must change SGE_TASK_ID in WriteData and WriteDataBattery
# 6/24/2025 - everything works perfectly on PC
# ---------------------------------------------------------------------------- #
#                                  Testing End                                 #
# ---------------------------------------------------------------------------- #


end # module RWModel
