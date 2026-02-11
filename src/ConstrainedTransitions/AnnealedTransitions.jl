"""
    struct AnnealedTransitions{T, M} where {T, M<:AbstractMatrix{T}} <: SparseTransitions{T,M}

    AnnealedTransitions(transition, β)
    AnnealedTransitions(::Type{<:AbstractMatrix}, state_count, β)
    AnnealedTransitions(::Type{<:Real}, state_count, β)
    AnnealedTransitions(state_count, β)

A variant of `SparseTransitions` where the penalty is applied as a power to the usual Baum-Welch update rather than a subtraction.
In other words, this changes the temperature of the transition update, with `β > 1` promoting sparsity and `β < 1` promoting more uniform transitions.
"""
struct AnnealedTransitions{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    transitions::M
    β::Real
end

# Constructors

function AnnealedTransitions(M::Type{<:AbstractMatrix{T}}, state_count::Integer, β::Real) where {T}
    transitions = M(fill(one(T) / state_count, state_count, state_count))
    return AnnealedTransitions(transitions, β)
end

function AnnealedTransitions(T::Type{<:Real}, state_count::Integer, β::Real)
    return AnnealedTransitions(Matrix{T}, state_count, β)
end

function AnnealedTransitions(state_count::Integer, β::Real)
    return AnnealedTransitions(Float64, state_count, β)
end

# Accessors

"""
    state_count(A::AnnealedTransitions)

Get the number of states represented by the annealed transition matrix.
"""
state_count(A::AnnealedTransitions) = size(A.transitions, 1)

Base.size(A::AnnealedTransitions) = size(A.transitions)
Base.eltype(::Type{AnnealedTransitions{T}}) where {T} = T
Base.IndexStyle(::Type{<:AnnealedTransitions{T,M}}) where {T,M} = IndexStyle(M)
Base.getindex(A::AnnealedTransitions, i::Int, j::Int) = getindex(A.transitions, i, j)
Base.getindex(A::AnnealedTransitions, i::Int) = getindex(A.transitions, i)
Base.iterate(A::AnnealedTransitions) = iterate(A.transitions)
Base.iterate(A::AnnealedTransitions, state) = iterate(A.transitions, state)
Base.copy(A::AnnealedTransitions) = AnnealedTransitions(copy(A.transitions), A.β)

function _broadcast(f, A::AnnealedTransitions)
    return AnnealedTransitions(f.(A.transitions), A.β)
end

Base.BroadcastStyle(::Type{<:AnnealedTransitions}) = Base.Broadcast.ArrayStyle{AnnealedTransitions}()

function Base.copy(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{AnnealedTransitions}})
    @argcheck length(bc.args) == 1 "broadcast over AnnealedTransitions only supports unary functions"
    arg = first(bc.args)
    A = arg isa AnnealedTransitions ? arg :
        arg isa Base.Broadcast.Extruded ? arg.value :
        throw(ArgumentError("Unsupported broadcast argument type $(typeof(arg))"))
    return _broadcast(bc.f, A)
end


function baum_welch_transition_update!(trans::AnnealedTransitions{T}, logtrans::AnnealedTransitions, expected::AbstractMatrix) where {T}
    @argcheck size(trans) == size(expected) DimensionMismatch
    @argcheck all(expected .>= zero(T)) ArgumentError("expected counts must be non-negative")

    usual_update = expected ./ sum(expected, dims=2)
    penalized_update = usual_update .^ trans.β
    sum_to_one!.(eachrow(penalized_update))

    trans.transitions .= penalized_update

    logtrans.transitions .= log.(trans.transitions)
    return nothing
end
