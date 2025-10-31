# --- MultiSeqHMM convenience constructors ---

function MultiSeqHMM(;
    inits,
    transitions,
    rbms::AbstractVector,
    hiddens,
    l2::Real=0.0,
    logits::Union{Nothing,AbstractMatrix}=nothing,
)
    if logits === nothing
        @argcheck hiddens isa AbstractMatrix "hiddens must be a matrix when logits are not supplied"
        families = RBMEmissionFamily.(rbms, l2)
        return MultiSeqHMM(inits, transitions, families, hiddens, (; l2))
    else
        @argcheck hiddens isa AbstractArray{<:Any,3} "hiddens must be a 3D array when logits are supplied"
        @argcheck size(logits) == size(hiddens)[1:end-1] "logits and hiddens must agree on the number of states and mixture components"
        families = RBMMultiEmissionFamily.(rbms, l2)
        θ = stack_vector_matrix(logits, hiddens)
        return MultiSeqHMM(inits, transitions, families, θ, (; l2))
    end
end
