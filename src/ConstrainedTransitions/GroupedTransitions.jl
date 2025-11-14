# ----------------------------------------------------------------------
#  STRUCT DEFINITION (generic within/between types)
# ----------------------------------------------------------------------

struct GroupedTransitions{T, W<:AbstractArray{T,3}, B<:AbstractMatrix{T}} <: AbstractMatrix{T}
    within::W             # P × K × K
    between::B            # P × P
    # inner constructor enforces consistency
    function GroupedTransitions{T,W,B}(within::W, between::B) where {T,W<:AbstractArray{T,3},B<:AbstractMatrix{T}}
        P, K1, K2 = size(within)
        @assert K1 == K2 "Within blocks must be K×K"
        @assert size(between) == (P, P) "Between must be P×P"
        new{T,W,B}(within, between)
    end
end

# External convenience constructor
function GroupedTransitions(within::AbstractArray{T,3}, between::AbstractMatrix{T}) where {T}
    return GroupedTransitions{T, typeof(within), typeof(between)}(within, between)
end

# Derived sizes (no stored P, K, N)
Base.size(A::GroupedTransitions) = (size(A.within,1)*size(A.within,2), size(A.within,1)*size(A.within,2))
Base.eltype(::Type{GroupedTransitions{T}}) where {T} = T
Base.IndexStyle(::Type{<:GroupedTransitions}) = IndexCartesian()

# convenience accessors
groups(A::GroupedTransitions) = size(A.within,1)
K(A::GroupedTransitions) = size(A.within,2)
N(A::GroupedTransitions) = size(A.within,1) * size(A.within,2)

# map global index → (group, local_index)
@generated function _split_index(i::Int, K::Int)
    quote
        g = (i - 1) ÷ K + 1
        k = (i - 1) % K + 1
        return g, k
    end
end

function Base.getindex(A::GroupedTransitions{T}, i::Int, j::Int) where {T}
    K = size(A.within,2)
    gi, ki = _split_index(i, K)
    gj, kj = _split_index(j, K)
    if gi == gj
        return A.within[gi, ki, kj]
    else
        return A.between[gi, gj]
    end
end

# ----------------------------------------------------------------------
#  BROADCASTING (structure-preserving)
# ----------------------------------------------------------------------

Base.BroadcastStyle(::Type{<:GroupedTransitions}) = Base.Broadcast.ArrayStyle{GroupedTransitions}()

function Base.broadcast(f, A::GroupedTransitions)
    T = eltype(A)
    W = similar(A.within)
    B = similar(A.between)

    @inbounds for g in 1:size(A.within,1), ki in 1:size(A.within,2), kj in 1:size(A.within,3)
        W[g,ki,kj] = f(A.within[g,ki,kj])
    end

    @inbounds for g in 1:size(A.between,1), h in 1:size(A.between,2)
        B[g,h] = g == h ? zero(T) : f(A.between[g,h])
    end

    return GroupedTransitions(W, B)
end

# ----------------------------------------------------------------------
#  BAUM–WELCH UPDATE
# ----------------------------------------------------------------------

function baumwelch_update!(A::GroupedTransitions{T}, Nξ::AbstractMatrix{T}) where {T}
    P = size(A.within,1)
    K = size(A.within,2)
    N = P*K

    N_out = zeros(T, P)
    N_in  = zeros(T, P)
    N_gh  = zeros(T, P, P)
    N_blk = zeros(T, P, K, K)

    @inbounds for i in 1:N
        gi, ki = _split_index(i, K)
        for j in 1:N
            ξij = Nξ[i,j]
            gj, kj = _split_index(j, K)
            N_out[gi] += ξij
            if gi == gj
                N_in[gi] += ξij
                N_blk[gi,ki,kj] += ξij
            else
                N_gh[gi,gj] += ξij
            end
        end
    end

    epsT = eps(T)

    @inbounds for g in 1:P
        rg = N_in[g] / (N_out[g] + epsT)
        for ki in 1:K
            denom = sum(N_blk[g,ki,:])
            if denom == 0
                for kj in 1:K
                    A.within[g,ki,kj] = T(1)/K
                end
            else
                for kj in 1:K
                    A.within[g,ki,kj] = rg * (N_blk[g,ki,kj] / denom)
                end
            end
        end
    end

    @inbounds for g in 1:P, h in 1:P
        if g != h
            A.between[g,h] = N_gh[g,h] / (K * (N_out[g] + epsT))
        end
    end

    return A
end

# ----------------------------------------------------------------------
#  LOG-VIEW (READ-ONLY)
# ----------------------------------------------------------------------

struct LogView{T,GT<:GroupedTransitions{T}} <: AbstractMatrix{T}
    parent::GT
end

Base.size(L::LogView) = size(L.parent)
Base.eltype(::Type{LogView{T}}) where {T} = T
Base.IndexStyle(::Type{<:LogView}) = IndexCartesian()

function Base.getindex(L::LogView{T}, i::Int, j::Int) where {T}
    return log(L.parent[i,j])
end

"""
    logview(A::GroupedTransitions)

Return a read-only matrix-like object where each entry is log(A[i,j]).
Useful to avoid materializing a dense log matrix.
"""
logview(A::GroupedTransitions) = LogView(A)
