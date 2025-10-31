using Random
using Test

using DensityInterface: logdensityof
using HiddenMarkovModels
using RestrictedBoltzmannMachines: RBM, Binary

using LinearAlgebra: Diagonal

using HMMRBM

@testset "RBM utils" begin
    rng = MersenneTwister(0)
    v = rand(rng, Float32, 2)
    mat = rand(rng, Float32, 2, 3)

    θ = HMMRBM.stack_vector_matrix(v, mat)
    v2, mat2 = HMMRBM.unstack_vector_matrix(θ)

    @test v2 == v
    @test mat2 == mat
    @test_throws DimensionMismatch HMMRBM.stack_vector_matrix(rand(rng, Float32, 3), mat)
end

@testset "RBM emission" begin
    rng = MersenneTwister(1)
    N, M = 3, 2
    rbm = RBM(Binary(randn(rng, N)'), Binary(randn(rng, M)'), randn(rng, N, M))
    l2 = 0.1
    family = HMMRBM.RBMEmissionFamily(rbm, l2)
    hidden = randn(rng, M)
    dist = HMMRBM.RBMEmission(rbm, hidden, l2)

    @test HMMRBM.family(dist) isa HMMRBM.RBMEmissionFamily
    @test HMMRBM.parameter(dist) === hidden

    obs = randn(rng, N)
    @test isfinite(logdensityof(dist, obs))

    sample_obs = rand(rng, dist)
    @test size(sample_obs) == size(obs)

    obs_seq = [obs, obs .+ 0.1]
    γ = [0.6, 0.4]
    value_grad = HMMRBM.baum_value_gradient_hessian(family, obs_seq, γ)

    value, grad, hess = value_grad(copy(hidden))
    @test isfinite(value)
    @test length(grad) == M
    @test hess isa Diagonal
end

@testset "RBM multi emission" begin
    rng = MersenneTwister(2)
    N, M, K = 4, 3, 2
    rbm = RBM(Binary(randn(rng, N)'), Binary(randn(rng, M)'), randn(rng, N, M))
    l2 = 0.05
    logits = randn(rng, K)
    hiddens = randn(rng, K, M)
    dist = HMMRBM.RBMMultiEmission(rbm, hiddens, logits, l2)
    family = HMMRBM.RBMMultiEmissionFamily(rbm, l2)

    obs = randn(rng, N)
    @test isfinite(logdensityof(dist, obs))

    sampled = rand(rng, dist)
    @test size(sampled) == size(obs)

    obs_seq = [obs, obs .+ 0.2, obs .- 0.1]
    γ = fill(1 / length(obs_seq), length(obs_seq))
    baum = HMMRBM.baum_value_gradient_hessian(family, obs_seq, γ)
    θ_vec = copy(HMMRBM.parameter(dist)[:])
    value, grad, hess = baum(θ_vec)

    @test isfinite(value)
    @test length(grad) == length(θ_vec)
    @test hess === nothing
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
