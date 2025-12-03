module HMMRBMHDF5Ext

using HMMRBM
using HDF5

const _HDF5Parent = Union{HDF5.File,HDF5.Group}

# Save a MultiSeqHMM to an HDF5 group

function HMMRBM.MultiSeqHMM_to_group!(group::_HDF5Parent, hmm::HMMRBM.MultiSeqHMM; kwargs...)
    inits_group = create_group(group, "inits")
    for (i, init) in enumerate(HMMRBM.inits(hmm))
        write(inits_group, "init_$i", init)
    end

    transitions_group = create_group(group, "transitions")
    for (i, trans) in enumerate(HMMRBM.transitions(hmm))
        write(transitions_group, "trans_$i", trans)
    end

    write(group, "emission_parameters", HMMRBM.emission_parameters(hmm))

    hyperparameters_group = create_group(group, "hyperparameters")
    for (key, value) in pairs(HMMRBM.hyperparameters(hmm))
        write(hyperparameters_group, string(key), value)
    end

    info_group = create_group(group, "info")
    for (key, value) in kwargs
        write(info_group, string(key), value)
    end

    return group
end

function HMMRBM.MultiSeqHMM_from_group(group::_HDF5Parent; emissions)
    inits_group = open_group(group, "inits")
    inits = [read(inits_group, "init_$i") for i in 1:length(inits_group)]

    transitions_group = open_group(group, "transitions")
    transitions = [read(transitions_group, "trans_$i") for i in 1:length(transitions_group)]

    θ = read(group, "emission_parameters")

    hyperparameters_group = open_group(group, "hyperparameters")
    hyperparameters = (; [Symbol(name) => read(hyperparameters_group, name) for name in keys(hyperparameters_group)]...)

    return HMMRBM.MultiSeqHMM(inits, transitions, emissions, θ, hyperparameters)
end

function HMMRBM.save_hmm(hmm::HMMRBM.MultiSeqHMM, filename::String; kwargs...)
    h5open(filename, "w") do file
        HMMRBM.MultiSeqHMM_to_group!(file, hmm; kwargs...)
    end
end

function HMMRBM.load_hmm(filename::String; emissions)
    h5open(filename, "r") do file
        HMMRBM.MultiSeqHMM_from_group(file; emissions)
    end
end

# Compatibility with constrained transition types

function HDF5.write(parent::_HDF5Parent, key::AbstractString, mat::HMMRBM.ConstrainedTransitions)
    write(parent, key, Matrix(mat))
end

end
