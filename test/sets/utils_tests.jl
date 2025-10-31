module UtilsTests

using Random
using Test

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

end # module
