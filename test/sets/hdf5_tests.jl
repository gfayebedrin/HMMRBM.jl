module HDF5Tests

using Random
using Test
using HDF5

using HiddenMarkovModels
using RestrictedBoltzmannMachines: RBM, Binary

using HMMRBM

@testset "HDF5 MultiSeqHMM round-trip" begin
    rng = MersenneTwister(5)
    n_states, hidden_dim = 2, 3
    rbm = RBM(
        Binary(randn(rng, n_states)'),
        Binary(randn(rng, hidden_dim)'),
        randn(rng, n_states, hidden_dim),
    )
    l2 = 0.15

    inits = [fill(0.5, n_states)]
    transitions = [Float64[0.6 0.4; 0.2 0.8]]
    hiddens = randn(rng, n_states, hidden_dim)

    hmm = HMMRBM.MultiSeqHMM(
        inits=inits,
        transitions=transitions,
        rbms=[rbm],
        hiddens=hiddens,
        l2=l2,
    )

    mktemp() do path, io
        close(io)

        HMMRBM.save_hmm(hmm, path; version="1.0.0", note="roundtrip test")

        h5open(path, "r") do file
            @test Set(keys(file)) == Set(["inits", "transitions", "emission_parameters", "hyperparameters", "info"])
            inits_group = file["inits"]
            @test length(inits_group) == length(inits)
            @test read(inits_group["init_1"]) == first(inits)

            info_group = file["info"]
            @test read(info_group["version"]) == "1.0.0"
            @test read(info_group["note"]) == "roundtrip test"
        end

        hmm_loaded = HMMRBM.load_hmm(path; emissions=HMMRBM.emissions(hmm))

        @test HMMRBM.inits(hmm_loaded) == inits
        @test HMMRBM.transitions(hmm_loaded) == transitions
        @test HMMRBM.emission_parameters(hmm_loaded) == HMMRBM.emission_parameters(hmm)
        @test HMMRBM.hyperparameters(hmm_loaded) == HMMRBM.hyperparameters(hmm)
        @test HMMRBM.emissions(hmm_loaded) == HMMRBM.emissions(hmm)
        @test HiddenMarkovModels.valid_hmm(hmm_loaded)
    end
end

end
