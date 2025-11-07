module RBMGradientTests

using LinearAlgebra
using Random
using Test

using RestrictedBoltzmannMachines: RBM, Binary, energy

using HMMRBM

@testset "RBM gradient helpers" begin
    rng = MersenneTwister(10)
    N, M = 4, 3
    rbm = RBM(Binary(randn(rng, N)'), Binary(randn(rng, M)'), randn(rng, N, M))
    obs_seq = [randn(rng, N) for _ in 1:3]
    γ = rand(rng, length(obs_seq))
    family = HMMRBM.RBMEmissionFamily(rbm, 0.05)
    f, grad! = HMMRBM.baum_value_gradient(family, obs_seq, γ)

    h = randn(rng, M)
    grad_buf = similar(h)
    grad!(grad_buf, copy(h))
    base_value = f(copy(h))
    step = 1e-4
    value_forward = f(copy(h .+ step .* grad_buf))
    value_backward = f(copy(h .- step .* grad_buf))
    @test value_forward ≥ base_value - 1e-8
    @test value_backward ≤ base_value + 1e-8

    hess_diag = diag(HMMRBM.∂ₕ²log_P_v_given_h(rbm, h))
    @test all(isfinite, hess_diag)
    @test all(hess_diag .<= 0)
end

end # module
