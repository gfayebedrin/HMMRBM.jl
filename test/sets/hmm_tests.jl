module HMMTests

using Random
using Test

using DensityInterface: logdensityof
using HiddenMarkovModels
using RestrictedBoltzmannMachines: RBM, Binary

using HMMRBM

@testset "hmm_rbm constructor" begin
    rng = MersenneTwister(5)
    N, M = 3, 2
    rbms = [
        RBM(Binary(randn(rng, N)'), Binary(randn(rng, M)'), randn(rng, N, M)),
        RBM(Binary(randn(rng, N)'), Binary(randn(rng, M)'), randn(rng, N, M)),
    ]

    Random.seed!(1234)
    n_states = 3
    l2 = 0.05
    hmm = HMMRBM.hmm_rbm(rbms, n_states; l2)

    @test hmm isa HMMRBM.MultiSeqHMM
    @test length(hmm) == length(rbms)
    @test all(isapprox(sum(init), 1; atol=1e-12) for init in HMMRBM.inits(hmm))
    @test all(all(isapprox(sum(row), 1; atol=1e-12) for row in eachrow(trans)) for trans in HMMRBM.transitions(hmm))

    emissions = HMMRBM.emissions(hmm)
    @test all(em -> em isa HMMRBM.RBMEmissionFamily, emissions)
    @test all(i -> HMMRBM.rbm(emissions[i]) === rbms[i], eachindex(rbms))
    @test all(em -> HMMRBM.l2(em) == l2, emissions)

    hidden_sizes = unique(size.(getfield.(rbms, :hidden)))
    @test length(hidden_sizes) == 1
    @test size(HMMRBM.emission_parameters(hmm)) == (n_states, only(hidden_sizes)...)
    @test all(iszero, HMMRBM.emission_parameters(hmm))

    @test HMMRBM.hyperparameters(hmm).l2 == l2
end

@testset "MultiSeqHMM wrapper" begin
    rng = MersenneTwister(3)
    N, M = 2, 2
    rbm = RBM(Binary(randn(rng, N)'), Binary(randn(rng, M)'), randn(rng, N, M))
    l2 = 0.2

    inits = [fill(0.5, 2)]
    transitions = [Float64[0.7 0.3; 0.4 0.6]]
    hiddens = randn(rng, 2, M)

    hmm = HMMRBM.MultiSeqHMM(
        inits=inits,
        transitions=transitions,
        rbms=[rbm],
        hiddens=hiddens,
        l2=l2,
    )

    @test length(hmm) == 1
    seq = hmm[1]
    @test HMMRBM.sequence_index(seq) == 1
    @test HMMRBM.parent(seq) === hmm

    emissions = HiddenMarkovModels.obs_distributions(seq)
    @test length(emissions) == size(hiddens, 1)
    @test all(d -> d isa HMMRBM.RBMEmission, emissions)

    penalty = logdensityof(seq)
    @test penalty <= 0
end

@testset "Baum-Welch convergence helper" begin
    history = Float64[]
    @test !HMMRBM.baum_welch_has_converged(history; atol=1e-4, loglikelihood_increasing=true)

    history = [0.0, 5e-5]
    @test HMMRBM.baum_welch_has_converged(history; atol=1e-4, loglikelihood_increasing=true)

    history = [0.0, -1e-3]
    @test_throws ErrorException HMMRBM.baum_welch_has_converged(history; atol=1e-4, loglikelihood_increasing=true)
end

@testset "Baum-Welch training" begin
    rng = MersenneTwister(4)
    N, M = 3, 2
    rbm = RBM(Binary(randn(rng, N)'), Binary(randn(rng, M)'), randn(rng, N, M))
    l2 = 0.1

    inits = [fill(0.5, 2)]
    transitions = [Float64[0.6 0.4; 0.3 0.7]]
    hiddens = randn(rng, 2, M)

    hmm_guess = HMMRBM.MultiSeqHMM(
        inits=inits,
        transitions=transitions,
        rbms=[rbm],
        hiddens=hiddens,
        l2=l2,
    )

    obs_sequences = [[randn(rng, N) for _ in 1:3]]

    hmm_est, logL = HiddenMarkovModels.baum_welch(
        hmm_guess,
        obs_sequences;
        max_iterations=1,
        loglikelihood_increasing=false,
        atol=-Inf,
    )

    @test length(logL) == 1
    @test size(HMMRBM.emission_parameters(hmm_est)) == size(hiddens)
    @test HiddenMarkovModels.valid_hmm(hmm_est)
end

end # module
