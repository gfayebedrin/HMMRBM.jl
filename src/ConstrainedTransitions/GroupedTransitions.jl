"""
    struct GroupedTransitions{T, W<:AbstractArray{T,3}, B<:AbstractMatrix{T}} <: AbstractMatrix{T}

A transition matrix with grouped structure. The states are divided into `P` groups, each containing `K` states.
Transitions within the same group are free, while transitions between different groups share the same probability.
"""
struct GroupedTransitions{T,W<:AbstractArray{T,3},B<:AbstractMatrix{T}} <: AbstractMatrix{T}
    within::W             # P × K × K
    between::B            # P × P

    function GroupedTransitions(within::W, between::B) where {T,W<:AbstractArray{T,3},B<:AbstractMatrix{T}}
        P, K1, K2 = size(within)
        @assert K1 == K2 "Within blocks must be K×K"
        @assert size(between) == (P, P) "Between must be P×P"
        new{T,W,B}(within, between)
    end
end

# Constructors

"""
    GroupedTransitions(W, B, group_count, states_per_group)
    GroupedTransitions(T, group_count, states_per_group)
    GroupedTransitions(group_count, states_per_group)

Construct a `GroupedTransitions` instance with `group_count` groups and `states_per_group` states per group.
Type `T` specifies the element type (default `Float64`).
Types `W` and `B` specify the array types (default `Array{T,3}` and `Matrix{T}`).
"""
function GroupedTransitions(
    W::Type{<:AbstractArray{T,3}},
    B::Type{<:AbstractMatrix{T}},
    group_count::Integer,
    states_per_group::Integer
) where {T}
    P = group_count
    K = states_per_group
    val = one(T) / (P * K)

    within = W(fill(val, P, K, K))
    between = B(fill(val, P, P))

    return GroupedTransitions(within, between)
end

function GroupedTransitions(
    T::Type{<:Real},
    group_count::Integer,
    states_per_group::Integer
)
    return GroupedTransitions(Array{T,3}, Matrix{T}, group_count, states_per_group)
end

function GroupedTransitions(
    group_count::Integer,
    states_per_group::Integer
)
    return GroupedTransitions(Float64, group_count, states_per_group)
end

# Accessors
"""
    group_count(A::GroupedTransitions)

Get the number of groups represented by the grouped transition matrix.
"""
group_count(A::GroupedTransitions) = size(A.within, 1)

"""
    states_per_group(A::GroupedTransitions)

Get the number of states per group represented by the grouped transition matrix.
"""
states_per_group(A::GroupedTransitions) = size(A.within, 2)

"""
    state_count(A::GroupedTransitions)

Get the total number of states represented by the grouped transition matrix.
"""
state_count(A::GroupedTransitions) = size(A.within, 1) * size(A.within, 2)

# Derived sizes
Base.size(A::GroupedTransitions) = (state_count(A), state_count(A))
Base.eltype(::Type{GroupedTransitions{T}}) where {T} = T
Base.IndexStyle(::Type{<:GroupedTransitions}) = IndexCartesian()

# map global index → (group, local_index)
function _split_index(i::Int, K::Int)
    g = (i - 1) ÷ K + 1
    k = (i - 1) % K + 1
    return g, k
end

function Base.getindex(A::GroupedTransitions, i::Int, j::Int)
    K = states_per_group(A)
    gi, ki = _split_index(i, K)
    gj, kj = _split_index(j, K)
    if gi == gj
        return A.within[gi, ki, kj]
    else
        return A.between[gi, gj]
    end
end

# Structure preserving broadcast

Base.BroadcastStyle(::Type{<:GroupedTransitions}) = Base.Broadcast.ArrayStyle{GroupedTransitions}()

function _broadcast(f, A::GroupedTransitions)
    T = eltype(A)
    W = similar(A.within)
    B = similar(A.between)

    @inbounds for g in 1:size(A.within, 1), ki in 1:size(A.within, 2), kj in 1:size(A.within, 3)
        W[g, ki, kj] = f(A.within[g, ki, kj])
    end

    @inbounds for g in 1:size(A.between, 1), h in 1:size(A.between, 2)
        B[g, h] = g == h ? zero(T) : f(A.between[g, h])
    end

    return GroupedTransitions(W, B)
end

Base.broadcast(f, A::GroupedTransitions) = _broadcast(f, A)

function Base.copy(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{GroupedTransitions}})
    @argcheck length(bc.args) == 1 "broadcast over GroupedTransitions only supports unary functions"
    arg = first(bc.args)
    A = arg isa GroupedTransitions ? arg :
        arg isa Base.Broadcast.Extruded ? arg.value :
        throw(ArgumentError("Unsupported broadcast argument type $(typeof(arg))"))
    return _broadcast(bc.f, A)
end

# Baum-Welch update

function baum_welch_transition_update!(trans::GroupedTransitions{T}, logtrans::GroupedTransitions, expected::AbstractMatrix) where {T}
    P = group_count(trans)
    K = states_per_group(trans)
    N = state_count(trans)

    expected_out = zeros(T, P)
    expected_in = zeros(T, P)
    expected_between = zeros(T, P, P)
    expected_within = zeros(T, P, K, K)

    @inbounds for i in 1:N
        groupᵢ, localᵢ = _split_index(i, K)
        for j in 1:N
            ξᵢⱼ = expected[i, j]
            groupⱼ, localⱼ = _split_index(j, K)
            expected_out[groupᵢ] += ξᵢⱼ
            if groupᵢ == groupⱼ
                expected_in[groupᵢ] += ξᵢⱼ
                expected_within[groupᵢ, localᵢ, localⱼ] += ξᵢⱼ
            else
                expected_between[groupᵢ, groupⱼ] += ξᵢⱼ
            end
        end
    end

    epsT = eps(T)

    # Within-group transitions
    @inbounds for g in 1:P
        group_persistence = expected_in[g] / (expected_out[g] + epsT)
        for ki in 1:K
            denom = sum(expected_within[g, ki, :])
            if denom == 0
                for kj in 1:K
                    trans.within[g, ki, kj] = 1 / K
                end
            else
                for kj in 1:K
                    trans.within[g, ki, kj] = group_persistence * (expected_within[g, ki, kj] / denom)
                end
            end
        end
    end

    # Between-group transitions
    @inbounds for g in 1:P, h in 1:P
        if g != h
            trans.between[g, h] = expected_between[g, h] / (K * (expected_out[g] + epsT))
        end
    end

    logtrans.within .= log.(trans.within)
    logtrans.between .= log.(trans.between)

    return nothing
end
