# ----------------------------------------------------------------------
#  STRUCT DEFINITION
# ----------------------------------------------------------------------

"""
    GroupedTransitions(groups, within, between)

Transition model for HMMs with P groups of K states each.

- groups[i] = group index of state i (1..P)
- within[i,j] = transition probability from i→j when both in same group
- between[g,h] = shared transition probability for any i∈g, j∈h (g≠h)

This object behaves like an AbstractMatrix (read-only) with size N×N.
"""
struct GroupedTransitions{T} <: AbstractMatrix{T}
    groups::Vector{Int}    # length N
    P::Int                 # number of groups
    K::Int                 # group size
    within::Matrix{T}      # N×N (only diagonal blocks used)
    between::Matrix{T}     # P×P, off-diagonal entries used
    N::Int
end


# ----------------------------------------------------------------------
#  CONSTRUCTOR
# ----------------------------------------------------------------------

function GroupedTransitions(groups::Vector{Int},
                            within::Matrix{T},
                            between::Matrix{T}) where {T}

    N = length(groups)
    P = maximum(groups)
    K = N ÷ P

    @assert size(within) == (N, N) "within must be N×N"
    @assert size(between) == (P, P) "between must be P×P"

    new{T}(groups, P, K, within, between, N)
end


# ----------------------------------------------------------------------
#  ABSTRACTMATRIX INTERFACE
# ----------------------------------------------------------------------

Base.size(A::GroupedTransitions) = (A.N, A.N)
Base.eltype(::Type{GroupedTransitions{T}}) where {T} = T
Base.IndexStyle(::Type{<:GroupedTransitions}) = IndexCartesian()

"""
    A[i,j]

Matrix-like access: if states i and j share a group, return within-block
transition; otherwise return between-block transition.
"""
function Base.getindex(A::GroupedTransitions{T}, i::Int, j::Int) where {T}
    gi = A.groups[i]
    gj = A.groups[j]

    if gi == gj
        return A.within[i,j]
    else
        return A.between[gi, gj]
    end
end


# ----------------------------------------------------------------------
#  BROADCASTING (LEVEL 2 STRUCTURE-PRESERVING)
# ----------------------------------------------------------------------

Base.BroadcastStyle(::Type{<:GroupedTransitions}) =
    Base.Broadcast.ArrayStyle{GroupedTransitions}()

"""
    broadcast(f, A::GroupedTransitions)

Apply an elementwise function f to all transition probabilities,
preserving the grouped structure. Broadcasting returns a new
GroupedTransitions object.
"""
function Base.broadcast(f, A::GroupedTransitions)
    within_new  = similar(A.within)
    between_new = similar(A.between)

    # apply f to within-block
    @inbounds for i in 1:A.N
        gi = A.groups[i]
        for j in 1:A.N
            if A.groups[j] == gi
                within_new[i,j] = f(A.within[i,j])
            else
                within_new[i,j] = zero(eltype(A))  # unused, keep clean
            end
        end
    end

    # apply f to between-block
    @inbounds for g in 1:A.P, h in 1:A.P
        if g == h
            between_new[g,h] = zero(eltype(A))
        else
            between_new[g,h] = f(A.between[g,h])
        end
    end

    return GroupedTransitions(A.groups, within_new, between_new)
end


# ----------------------------------------------------------------------
#  BAUM–WELCH UPDATE
# ----------------------------------------------------------------------

"""
    baumwelch_update!(A::GroupedTransitions, Nξ)

Update the grouped transition parameters in-place given expected counts Nξ.
Implements the block-structured closed-form EM updates:

- within-block rows follow:
      a_ij = r_g * (N_ij / sum_j N_ij)
  with r_g = N_{g,in} / N_{g,out}

- between-block:
      c_{g→h} = N_{g→h} / (K * N_{g,out})
"""
function baumwelch_update!(A::GroupedTransitions{T},
                           Nξ::AbstractMatrix{T}) where {T}

    groups = A.groups
    P, K, N = A.P, A.K, A.N

    # group totals
    N_in  = zeros(T, P)
    N_out = zeros(T, P)
    N_gh  = zeros(T, P, P)

    # accumulate counts
    @inbounds for i in 1:N
        gi = groups[i]
        for j in 1:N
            ξij = Nξ[i,j]
            gj = groups[j]
            N_out[gi] += ξij
            if gi == gj
                N_in[gi] += ξij
            else
                N_gh[gi,gj] += ξij
            end
        end
    end

    epsT = eps(T)

    # update within/between
    @inbounds for g in 1:P
        rg = N_in[g] / (N_out[g] + epsT)

        # update within-block for all i in group g
        for i in 1:N
            gi = groups[i]
            gi == g || continue

            # denominator: transitions from i to states in g
            denom = zero(T)
            for j in 1:N
                if groups[j] == g
                    denom += Nξ[i,j]
                end
            end

            for j in 1:N
                if groups[j] == g
                    if denom == 0
                        A.within[i,j] = T(1)/K
                    else
                        A.within[i,j] = rg * (Nξ[i,j] / denom)
                    end
                end
            end
        end

        # update between-block
        for h in 1:P
            if h != g
                A.between[g,h] = N_gh[g,h] / (K * (N_out[g] + epsT))
            end
        end
    end

    return A
end