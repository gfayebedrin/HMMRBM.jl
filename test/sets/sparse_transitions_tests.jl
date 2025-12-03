module SparseTransitionsTests

using Random
using Test

using HiddenMarkovModels
using RestrictedBoltzmannMachines: RBM, Binary

using HMMRBM
using HMMRBM:
    SparseTransitions,
    baum_welch_transition_update!,
    state_count,
    transitions

@testset "SparseTransitions constructors and validation" begin
    λ = 0.2
    trans = SparseTransitions(3, λ)

    @test state_count(trans) == 3
    @test size(trans) == (3, 3)
    @test eltype(trans) == Float64
    @test all(==(1 / 3), trans.transitions)
    @test trans.λ == λ

    typed = SparseTransitions(Float32, 2, λ)
    @test eltype(typed) == Float32
    @test typed.transitions isa Matrix{Float32}

    explicit = SparseTransitions([0.6 0.4; 0.25 0.75], λ)
    @test explicit[1, 2] == 0.4
    @test explicit[2, 1] == 0.25
end

@testset "SparseTransitions broadcast and copy" begin
    base = SparseTransitions([0.2 0.8; 0.3 0.7], 0.1)

    doubled = broadcast(x -> 2x, base)
    @test doubled.transitions ≈  2 .* base.transitions
    @test doubled.λ == base.λ

    copied = copy(base)
    @test copied !== base
    @test copied.transitions == base.transitions
    @test copied.λ == base.λ
end

@testset "SparseTransitions Baum-Welch update" begin
    λ = 1.5
    trans = SparseTransitions([0.5 0.5; 0.5 0.5], λ)
    logtrans = SparseTransitions([0.5 0.5; 0.5 0.5], λ)
    expected = [
        6 1;
        2 3
    ]

    baum_welch_transition_update!(trans, logtrans, expected)

    @test all(isapprox(sum(trans.transitions[i, :]), 1.0; atol=1e-12) for i in axes(trans.transitions, 1))
    @test trans.transitions[1, 1] ≈ 1.0 atol=1e-12
    @test trans.transitions[1, 2] ≈ 0.0 atol=1e-12
    @test trans.transitions[2, 1] ≈ 1/7 atol=1e-12
    @test trans.transitions[2, 2] ≈ 6/7 atol=1e-12

    @test logtrans.transitions ≈ log.(trans.transitions)
end

@testset "SparseTransitions integration with training" begin
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
        transitions=[SparseTransitions(n_states, 0.05)],
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
    @test transitions(hmm_est)[1] isa SparseTransitions
    dense = transitions(hmm_est)[1].transitions
    @test all(isapprox(sum(dense[i, :]), 1.0; atol=1e-10) for i in 1:size(dense, 1))
    @test all(dense .>= 0)
    @test HiddenMarkovModels.valid_hmm(hmm_est)
end

end # module
