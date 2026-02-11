module AnnealedTransitionsTests

using Random
using Test

using HiddenMarkovModels
using RestrictedBoltzmannMachines: RBM, Binary

using HMMRBM
using HMMRBM:
    AnnealedTransitions,
    SparseTransitions,
    baum_welch_transition_update!,
    state_count,
    transitions

@testset "AnnealedTransitions constructors and validation" begin
    λ = 0.2
    trans = AnnealedTransitions(3, λ)

    @test state_count(trans) == 3
    @test size(trans) == (3, 3)
    @test eltype(trans) == Float64
    @test all(==(1 / 3), trans.transitions)
    @test trans.λ == λ

    typed = AnnealedTransitions(Float32, 2, λ)
    @test eltype(typed) == Float32
    @test typed.transitions isa Matrix{Float32}

    explicit = AnnealedTransitions([0.6 0.4; 0.25 0.75], λ)
    @test explicit[1, 2] == 0.4
    @test explicit[2, 1] == 0.25
end

@testset "AnnealedTransitions broadcast and copy" begin
    base = AnnealedTransitions([0.2 0.8; 0.3 0.7], 0.1)

    doubled = broadcast(x -> 2x, base)
    @test doubled.transitions ≈  2 .* base.transitions
    @test doubled.λ == base.λ

    copied = copy(base)
    @test copied !== base
    @test copied.transitions == base.transitions
    @test copied.λ == base.λ
end

@testset "AnnealedTransitions Baum-Welch update" begin
    λ = 2.0
    trans = AnnealedTransitions([0.5 0.5; 0.5 0.5], λ)
    logtrans = AnnealedTransitions([0.5 0.5; 0.5 0.5], λ)
    expected = [
        6.0 1.0;
        2.0 3.0
    ]

    baum_welch_transition_update!(trans, logtrans, expected)

    @test all(isapprox(sum(trans.transitions[i, :]), 1.0; atol=1e-12) for i in axes(trans.transitions, 1))
    @test trans.transitions[1, 1] ≈ 36/37
    @test trans.transitions[1, 2] ≈ 1/37
    @test trans.transitions[2, 1] ≈ 4/13
    @test trans.transitions[2, 2] ≈ 9/13

    @test logtrans.transitions ≈ log.(trans.transitions)
end

@testset "AnnealedTransitions integration with training" begin
    rng = MersenneTwister(21)
    n_states = 3
    N, M = 3, 2
    rbm = RBM(
        Binary(randn(rng, N)'),
        Binary(randn(rng, M)'),
        randn(rng, N, M),
    )

    hmm_guess = HMMRBM.MultiSeqHMM(
        inits=[fill(1 / n_states, n_states)],
        transitions=[AnnealedTransitions(n_states, 1.2)],
        rbms=[rbm],
        hiddens=randn(rng, n_states, M),
        l2=0.05,
    )

    obs_sequences = [[randn(rng, N) for _ in 1:5]]
    hmm_est, logL = HiddenMarkovModels.baum_welch(
        hmm_guess,
        obs_sequences;
        max_iterations=2,
        atol=-Inf,
        loglikelihood_increasing=false,
    )

    @test length(logL) == 2
    @test transitions(hmm_est)[1] isa AnnealedTransitions
    dense = transitions(hmm_est)[1].transitions
    @test all(isapprox(sum(dense[i, :]), 1.0; atol=1e-10) for i in 1:size(dense, 1))
    @test all(dense .>= 0)
    @test HiddenMarkovModels.valid_hmm(hmm_est)
end

end # module
