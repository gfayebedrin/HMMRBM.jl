"""
    hmm_rbm(rbms, n_states; l2=0.0)

Create a `MultiSeqHMM` that ties each observed sequence to one RBM while sharing a common
number of hidden Markov states. Initial probabilities and transition matrices are seeded
with simple defaults, and the returned model stores the provided ℓ2 regularisation value
in its hyperparameters.
"""
function hmm_rbm(rbms::AbstractVector{<:RestrictedBoltzmannMachines.RBM}, n_states::Integer; l2::Real=0.0)

    @argcheck !isempty(rbms) ArgumentError("At least one RBM is required")

    _size_hidden = unique(size.(getfield.(rbms, :hidden)))
    @argcheck length(_size_hidden) == 1 ArgumentError("All RBMs must have the same hidden layer size")
    size_hidden = only(_size_hidden)

    init = fill(1 / n_states, n_states)
    inits = [copy(init) for _ in rbms]

    function random_transition()
        trans = abs.(1 .+ 0.01 .* randn(n_states, n_states))
        foreach(sum_to_one!, eachrow(trans))
        trans
    end
    transitions = [random_transition() for _ in rbms]

    emissions = RBMEmissionFamily.(rbms, l2)

    θ = zeros(eltype(init), n_states, size_hidden...)

    return MultiSeqHMM(inits, transitions, emissions, θ, (; l2))
end
