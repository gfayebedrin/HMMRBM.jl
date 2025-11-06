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

    A = rand(rng, Float32, 4, 3)
    x = rand(rng, Float32, 3)
    M = rand(rng, Float32, 3, 5)
    T = rand(rng, Float32, 3, 5, 2)

    @test HMMRBM.my_mult(A, x) ≈ A * x
    @test HMMRBM.my_mult(A, M) ≈ A * M

    expected = reshape(A * reshape(T, size(T, 1), :), size(A, 1), size(T)[2:end]...)
    result = HMMRBM.my_mult(A, T)

    @test size(result) == size(expected)
    @test result ≈ expected
    @test eltype(result) == Float32
end

end # module
