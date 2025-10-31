module CUDATests

using Random
using Test

using CUDA
using Adapt
using HiddenMarkovModels
import LinearAlgebra
using RestrictedBoltzmannMachines: RBM, Binary

using HMMRBM

@testset "CUDA integration" begin
    if !CUDA.has_cuda()
        @info "CUDA unavailable; skipping CUDA integration tests"
        @test true
    else
        rng = MersenneTwister(5)
        N, M = 3, 2

        visible_cpu = randn(rng, Float32, N)
        hidden_cpu = randn(rng, Float32, M)
        weights_cpu = randn(rng, Float32, N, M)

        rbm_gpu = RBM(
            Binary(cu(visible_cpu)'),
            Binary(cu(hidden_cpu)'),
            cu(weights_cpu)
        )

        inits = [fill(0.5f0, 2)]
        transitions = [Float32[0.6 0.4; 0.3 0.7]]
        hiddens = randn(rng, Float32, 2, M)
        obs_sequences = [[randn(rng, Float32, N) for _ in 1:3]]
        obs_sequences_gpu = [[cu(obs) for obs in seq] for seq in obs_sequences]

        inits_gpu = [cu(v) for v in inits]
        transitions_gpu = [cu(T) for T in transitions]
        hiddens_gpu = cu(hiddens)

        hmm_gpu = HMMRBM.MultiSeqHMM(
            inits=inits_gpu,
            transitions=transitions_gpu,
            rbms=[rbm_gpu],
            hiddens=hiddens_gpu,
            l2=0.1,
        )

        @test all(x -> x isa CUDA.CuArray, HMMRBM.inits(hmm_gpu))
        @test all(x -> x isa CUDA.CuArray, HMMRBM.transitions(hmm_gpu))
        @test HMMRBM.emission_parameters(hmm_gpu) isa CUDA.CuArray
        hmm_roundtrip = adapt(Array, hmm_gpu)
        @test all(isapprox.(HMMRBM.inits(hmm_roundtrip), inits))
        @test all(isapprox.(HMMRBM.transitions(hmm_roundtrip), transitions))
        @test isapprox(HMMRBM.emission_parameters(hmm_roundtrip), hiddens)
        @test HiddenMarkovModels.valid_hmm(hmm_roundtrip)

        hmm_est_gpu, logL_gpu = HiddenMarkovModels.baum_welch(
            hmm_gpu,
            obs_sequences_gpu;
            max_iterations=1,
            loglikelihood_increasing=false,
            atol=-Inf,
        )

        @test length(logL_gpu) == 1
        @test HiddenMarkovModels.valid_hmm(adapt(Array, hmm_est_gpu))
        @test HMMRBM.emission_parameters(hmm_est_gpu) isa CUDA.CuArray
    end
end

end # module
