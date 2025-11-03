module HMMRBM

using HiddenMarkovModels
using StatsAPI
using DensityInterface
using Random
using LinearAlgebra
using StatsFuns
using ArgCheck
using Adapt
using RestrictedBoltzmannMachines
using Distributions

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
include("RBMDistribution/RBMMultiDistribution.jl")
export RBMMultiEmission
export RBMMultiEmissionFamily
include("RBMDistribution/constructors.jl")

include("hmm.jl")
export hmm_rbm

export MultiSeqHMM_to_group!
export MultiSeqHMM_from_group
export save_hmm
export load_hmm

"""
    MultiSeqHMM_to_group!(group, hmm; kwargs...)

Persist a `MultiSeqHMM` into an HDF5 group and optionally store metadata in a dedicated
`info` subgroup.
"""
function MultiSeqHMM_to_group! end
"""
    MultiSeqHMM_from_group(group; emissions)

Rebuild a `MultiSeqHMM` from datasets stored in an HDF5 group. The caller supplies the
emission families that match the stored parameters.
"""
function MultiSeqHMM_from_group end
"""
    save_hmm(hmm, filename; kwargs...)

Write a `MultiSeqHMM` to an HDF5 file, forwarding keyword arguments to
`MultiSeqHMM_to_group!`.
"""
function save_hmm end
"""
    load_hmm(filename; emissions)

Load a `MultiSeqHMM` from an HDF5 file by reading the saved arrays and pairing them with
the provided emission families.
"""
function load_hmm end

end
