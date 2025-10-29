module HMMRBM

using HiddenMarkovModels
using StatsAPI
using DensityInterface
using Random
using LinearAlgebra
using StatsFuns
using ArgCheck
using Adapt

include("HMM/utils.jl")
include("HMM/AbstractDistributions.jl")
export Distribution
export DistributionFamily
export distribution
export family
export parameter
export baum_value_gradient_hessian

include("HMM/MultiSeqHMM.jl")
export MultiSeqHMM
export SingleSeqHMM
export inits
export loginits
export transitions
export logtransitions
export emissions
export emission_parameters
export hyperparameters
export parent
export sequence_index

include("HMM/inference.jl")

include("RBMDistribution/utils.jl")
include("RBMDistribution/RBMDistribution.jl")
export RBMEmission
export RBMEmissionFamily

include("hmm.jl")
export hmm_rbm

end
