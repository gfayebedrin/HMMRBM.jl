# --- MultiSeqHMM convenience constructors ---


"""
    MultiSeqHMM(; inits, transitions, rbms, hiddens, l2=0.0, logits=nothing)

Construct a multi-sequence hidden Markov model whose emission distributions are driven by
Restricted Boltzmann Machine parameters. The keyword arguments fill the per-sequence
initial distributions and transition matrices, select one RBM per sequence, and provide
the shared hidden-state parameters. When `logits` are omitted, a single hidden vector per
state is assumed; supplying logits activates mixture emissions. The `l2` keyword stores
the regularisation strength in the resulting model.
"""
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
