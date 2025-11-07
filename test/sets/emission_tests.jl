module EmissionTests

using Random
using Test

using DensityInterface: logdensityof
using RestrictedBoltzmannMachines: RBM, Binary

using HMMRBM

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
    f, grad! = HMMRBM.baum_value_gradient(family, obs_seq, γ)

    value = f(copy(hidden))
    @test value isa Real
    @test isfinite(value)
    grad = similar(hidden)
    grad!(grad, copy(hidden))
    @test grad isa AbstractVector
    @test length(grad) == M
    @test all(isfinite, grad)
end

@testset "RBM multi emission" begin
    rng = MersenneTwister(2)
    N, M, K = 5, 3, 2
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
    f, grad! = HMMRBM.baum_value_gradient(family, obs_seq, γ)
    θ_vec = copy(HMMRBM.parameter(dist)[:])
    value = f(copy(θ_vec))

    @test value isa Real
    @test isfinite(value)
    grad = similar(θ_vec)
    grad!(grad, copy(θ_vec))
    @test grad isa AbstractVector
    @test length(grad) == length(θ_vec)
    @test all(isfinite, grad)
end

end # module
