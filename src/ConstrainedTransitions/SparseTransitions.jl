"""
    struct SparseTransitions{T, M} where {T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}

    SparseTransitions(transition, λ)
    SparseTransitions(::Type{<:AbstractMatrix}, state_count, λ)
    SparseTransitions(::Type{<:Real}, state_count, λ)
    SparseTransitions(state_count, λ)

A transition matrix with a penalty on non diagonal elements. During Baum-Welch updates the
off-diagonals are shrunk by `λ` before rows are renormalised, promoting self-transitions
and sparsity away from the diagonal while keeping the matrix row-stochastic.
"""
struct SparseTransitions{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    transitions::M
    λ::Real
end

# Constructors

function SparseTransitions(M::Type{<:AbstractMatrix{T}}, state_count::Integer, λ::Real) where {T}
    transitions = M(fill(one(T) / state_count, state_count, state_count))
    return SparseTransitions(transitions, λ)
end

function SparseTransitions(T::Type{<:Real}, state_count::Integer, λ::Real)
    return SparseTransitions(Matrix{T}, state_count, λ)
end

function SparseTransitions(state_count::Integer, λ::Real)
    return SparseTransitions(Float64, state_count, λ)
end

# Accessors

"""
    state_count(A::SparseTransitions)

Get the number of states represented by the sparse transition matrix.
"""
state_count(A::SparseTransitions) = size(A.transitions, 1)

Base.size(A::SparseTransitions) = size(A.transitions)
Base.eltype(::Type{SparseTransitions{T}}) where {T} = T
Base.IndexStyle(::Type{<:SparseTransitions{T,M}}) where {T,M} = IndexStyle(M)
Base.getindex(A::SparseTransitions, i::Int, j::Int) = getindex(A.transitions, i, j)
Base.getindex(A::SparseTransitions, i::Int) = getindex(A.transitions, i)
Base.iterate(A::SparseTransitions) = iterate(A.transitions)
Base.iterate(A::SparseTransitions, state) = iterate(A.transitions, state)
Base.copy(A::SparseTransitions) = SparseTransitions(copy(A.transitions), A.λ)

function _broadcast(f, A::SparseTransitions)
    return SparseTransitions(f.(A.transitions), A.λ)
end

Base.BroadcastStyle(::Type{<:SparseTransitions}) = Base.Broadcast.ArrayStyle{SparseTransitions}()

function Base.copy(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{SparseTransitions}})
    @argcheck length(bc.args) == 1 "broadcast over SparseTransitions only supports unary functions"
    arg = first(bc.args)
    A = arg isa SparseTransitions ? arg :
        arg isa Base.Broadcast.Extruded ? arg.value :
        throw(ArgumentError("Unsupported broadcast argument type $(typeof(arg))"))
    return _broadcast(bc.f, A)
end


# Baum-Welch update

function baum_welch_transition_update!(trans::SparseTransitions{T}, logtrans::SparseTransitions, expected::AbstractMatrix) where {T}
    @argcheck size(trans) == size(expected) DimensionMismatch
    @argcheck all(expected .>= zero(T)) ArgumentError("expected counts must be non-negative")

    rows, cols = size(expected)
    for j in 1:cols, i in 1:rows
        val = expected[i, j]
        if i != j
            # Penalize off-diagonals
            val = max(eps(T), val - trans.λ)
        end
        trans.transitions[i, j] = val
    end
    for row in eachrow(trans.transitions)
        if sum(row) == zero(T)
            row .= one(T)
        end
        sum_to_one!(row)
    end

    logtrans.transitions .= log.(trans.transitions)
    return nothing
end
