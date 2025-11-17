module GroupedTransitionsTests

using Random
using Test

using HiddenMarkovModels
using RestrictedBoltzmannMachines: RBM, Binary

using HMMRBM
using HMMRBM:
    GroupedTransitions,
    baum_welch_transition_update!,
    group_count,
    state_count,
    states_per_group,
    transitions

function dense_transition_matrix(A::GroupedTransitions)
    n = state_count(A)
    mat = Array{eltype(A)}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        mat[i, j] = A[i, j]
    end
    mat
end

@testset "GroupedTransitions constructors and indexing" begin
    trans = GroupedTransitions(2, 3)
    val = 1 / (group_count(trans) * states_per_group(trans))

    @test group_count(trans) == 2
    @test states_per_group(trans) == 3
    @test state_count(trans) == 6
    @test size(trans) == (6, 6)
    @test eltype(trans) == Float64
    @test all(==(val), trans.within)
    @test all(==(val), trans.between)
    @test trans[2, 3] == val

    custom_within = reshape(Float64.(1:8), 2, 2, 2)
    custom_between = [10.0 11.0; 12.0 13.0]
    grouped = GroupedTransitions(custom_within, custom_between)

    @test grouped[1, 2] == custom_within[1, 1, 2]
    @test grouped[2, 4] == custom_between[1, 2]
    @test grouped[3, 1] == custom_between[2, 1]

    typed = GroupedTransitions(Float32, 2, 2)
    @test eltype(typed) == Float32

    typed_array = GroupedTransitions(Array{Float32,3}, Matrix{Float32}, 2, 2)
    @test typed_array.within isa Array{Float32,3}
    @test typed_array.between isa Matrix{Float32}
end

@testset "GroupedTransitions broadcast" begin
    custom_within = reshape(collect(1:8), 2, 2, 2)
    custom_between = [10 11; 12 13]
    trans = GroupedTransitions(custom_within, custom_between)

    doubled = broadcast(x -> 2x, trans)

    @test doubled.within == 2 .* custom_within
    @test doubled.between[1, 2] == 2 * custom_between[1, 2]
    @test doubled.between[2, 1] == 2 * custom_between[2, 1]
end

@testset "GroupedTransitions Baum-Welch update" begin
    P, K = 2, 2
    trans = GroupedTransitions(P, K)
    logtrans = GroupedTransitions(P, K)
    expected = [
        0.2  0.1  0.4  0.3;
        0.05 0.15 0.4  0.4;
        0.2  0.4  0.3  0.1;
        0.1  0.4  0.25 0.25
    ]

    baum_welch_transition_update!(trans, logtrans, expected)

    dense = dense_transition_matrix(trans)
    @test all(isapprox(sum(dense[i, :]), 1.0; atol=1e-12) for i in axes(dense, 1))

    @test isapprox(trans.within[1, 1, 1], 1 / 6; atol=1e-12)
    @test isapprox(trans.within[1, 2, 1], 1 / 16; atol=1e-12)
    @test isapprox(trans.between[1, 2], 0.375; atol=1e-12)
    @test isapprox(trans.between[2, 1], 0.275; atol=1e-12)

    @test logtrans.within ≈ log.(trans.within)
    @test logtrans.between ≈ log.(trans.between)
end

@testset "GroupedTransitions integration with training" begin
    rng = MersenneTwister(11)
    group_count = 2
    states_per_group = 2
    n_states = group_count * states_per_group
    N, M = 3, 2
    rbm = RBM(
        Binary(randn(rng, N)'),
        Binary(randn(rng, M)'),
        randn(rng, N, M),
    )

    hmm_guess = HMMRBM.MultiSeqHMM(
        inits=[fill(1 / n_states, n_states)],
        transitions=[GroupedTransitions(group_count, states_per_group)],
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
    @test transitions(hmm_est)[1] isa GroupedTransitions
    dense = dense_transition_matrix(transitions(hmm_est)[1])
    @test all(isapprox(sum(dense[i, :]), 1.0; atol=1e-10) for i in 1:size(dense, 1))
    @test HiddenMarkovModels.valid_hmm(hmm_est)
end

end # module
