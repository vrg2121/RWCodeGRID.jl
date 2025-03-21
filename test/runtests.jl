using Test, RWModel

config = ModelConfig(1, 0, 0, 0, 2, 100, 0, [2.0, 4.0, 6.0])

@test println("The model configuration is: ", config)