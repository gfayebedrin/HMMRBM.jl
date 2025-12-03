const ConstrainedTransitions = Union{GroupedTransitions, SparseTransitions}

function _broadcast end

Base.BroadcastStyle(::Type{T}) where {T<:ConstrainedTransitions}= Base.Broadcast.ArrayStyle{T}()

Base.broadcast(f, A::ConstrainedTransitions) = _broadcast(f, A)

function Base.copy(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{T}}) where {T<:ConstrainedTransitions}
    @argcheck length(bc.args) == 1 "broadcast over SparseTransitions only supports unary functions"
    arg = first(bc.args)
    A = arg isa T ? arg :
        arg isa Base.Broadcast.Extruded ? arg.value :
        throw(ArgumentError("Unsupported broadcast argument type $(typeof(arg))"))
    return _broadcast(bc.f, A)
end
