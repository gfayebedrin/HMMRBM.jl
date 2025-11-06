σ(z) = logistic(z)
σ_prime(z) = σ(z) * (one(z) - σ(z))


"""
    my_mult(A, B)

Contract matrix `A` with array `B` over the last dimension of `A` and the first
dimension of `B`.
"""
my_mult(A::AbstractMatrix, B::AbstractVector) = A * B
my_mult(A::AbstractMatrix, B::AbstractMatrix) = A * B
function my_mult(A::AbstractMatrix, B::AbstractArray{T, N}) where {T, N}
    @argcheck size(A, 2) == size(B, 1) DimensionMismatch(
        "Incompatible dimensions: size(A) = $(size(A)), size(B) = $(size(B))"
    )

    tail_dims = Base.tail(size(B))
    result_eltype = promote_type(eltype(A), T)
    out = similar(B, result_eltype, (size(A, 1), tail_dims...))

    @views begin
        flat_B = reshape(B, size(B, 1), :)
        flat_out = reshape(out, size(out, 1), :)
        mul!(flat_out, A, flat_B)
    end

    out
end


"""
    stack_vector_matrix(v, mat)

Stack a vector `v` and a matrix `mat` into a single matrix where the first column
is `v` and the remaining columns are `mat`. The first dimensions of `v` must match
all but the last dimension of `mat`.
"""
function stack_vector_matrix(v::AbstractArray{T, N}, mat::AbstractArray{T, Np1}) where {T,N,Np1}
    (K..., M) = size(mat)
    @argcheck size(v) == K DimensionMismatch("Vector size must match matrix size except for last dimension")
    @argcheck Np1 == N + 1 DimensionMismatch("Matrix must have one more dimension than vector")

    θ = similar(mat, K..., M+1)
    selectdim(θ, ndims(θ), 1) .= v
    selectdim(θ, ndims(θ), 2:M+1) .= mat

    θ
end

"""
    unstack_vector_matrix(θ)

Unstack a matrix `θ` into a vector and a matrix where the first column of `θ`
is the vector and the remaining columns are the matrix. The returned vector has
all but the last dimension of `θ`, and the returned matrix has the same first
dimensions as `θ` but with the last dimension reduced by one.
"""
function unstack_vector_matrix(θ::AbstractArray{T, N}) where {T,N}
    M_plus_1 = size(θ, N)

    v = selectdim(θ, ndims(θ), 1)
    mat = selectdim(θ, ndims(θ), 2:M_plus_1)

    v, mat
end
