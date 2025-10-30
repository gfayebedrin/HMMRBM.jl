σ(z) = logistic(z)
σ_prime(z) = σ(z) * (one(z) - σ(z))


function stack_vector_matrix(v::AbstractArray{T, N}, mat::AbstractArray{T, N+1}) where {T,N}
    (K..., M) = size(mat)
    @argcheck size(v) == K DimensionMismatch("Vector size must match matrix size except for last dimension")

    θ = similar(mat, K..., M+1)
    selectdim(θ, ndims(θ), 1) .= v
    selectdim(θ, ndims(θ), 2:M+1) .= mat

    θ
end

function unstack_vector_matrix(θ::AbstractArray{T, N}) where {T,N}
    M_plus_1 = size(θ, N)
    
    v = selectdim(θ, ndims(θ), 1)
    mat = selectdim(θ, ndims(θ), 2:M_plus_1)

    v, mat
end