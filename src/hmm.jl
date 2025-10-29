function hmm_rbm(rbms::AbstractVector, n_states::Integer; l2::Real=0.0)

    _size_hidden = unique(size.(getfield.(rbms, :hidden)))
    @argcheck length(_size_hidden) == 1 ArgumentError("All RBMs must have the same hidden layer size")
    size_hidden = only(_size_hidden)

    init = fill(1 / n_states, n_states)
    inits = [copy(init) for _ in rbms]

    trans = 1.0 .+ 0.01 .* randn(n_states, n_states) .|> abs |> row_normalize
    transitions = [copy(trans) for _ in rbms]

    emissions = RBMEmissionFamily.(rbms, l2)

    θ = zeros(n_states, size_hidden...)

    return MultiSeqHMM(init, trans, emissions, θ, (; l2))
end
