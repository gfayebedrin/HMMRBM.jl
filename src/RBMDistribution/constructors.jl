# --- MultiSeqHMM convenience constructors ---


"""
    MultiSeqHMM(; inits, transitions, rbms, hiddens, l2=0.0, logits=nothing)

Construct a `MultiSeqHMM` whose emission distributions are parameterised by a set of
Restricted Boltzmann Machines (RBMs).

# Arguments
- `inits`: vector of initial state probability vectors, one per observed sequence.
- `transitions`: vector of square transition matrices matching `inits`.
- `rbms`: vector of RBM models (from `RestrictedBoltzmannMachines.jl`), one per sequence.
- `hiddens`: state-parameter arrays shared across sequences. Provide a matrix of size
  `(num_states, hidden_dim)` when `logits === nothing`, or a 3D array of size
  `(num_states, num_components, hidden_dim)` when supplying mixture logits.
- `l2`: non-negative ℓ2 penalty applied to hidden-state parameters during training (stored
  in the model hyperparameters).
- `logits`: optional matrix of unnormalised mixture weights `(num_states, num_components)` that
  activates the multi-component emission family.

All collections must have the same length and consistent element types. When `logits` are
omitted the constructor builds `RBMEmission` distributions; otherwise it builds
`RBMMultiEmission` distributions and stacks the logits/hidden states into the shared
parameter tensor.

# Returns
A `MultiSeqHMM` ready to be passed to routines such as `HiddenMarkovModels.baum_welch`.
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
