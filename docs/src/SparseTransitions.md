# Sparse transition matrices

`SparseTransitions` represents a row-stochastic transition matrix with an L1-like
shrinkage on off-diagonal probabilities. During Baum-Welch updates, each off-diagonal
entry is reduced by `λ` and clipped at zero before rows are renormalised. Larger values
of `λ` therefore encourage self-transitions and sparsity away from the diagonal.

```julia
julia> trans = HMMRBM.SparseTransitions(4, 0.2);

julia> size(trans)
(4, 4)

julia> trans[1, 4]
0.25
```

Sparse transitions can be passed directly to the `transitions` field of `MultiSeqHMM` and
remain stochastic through the EM loop thanks to
`HMMRBM.baum_welch_transition_update!`. If a row’s expected counts are fully zeroed, it is reset to a uniform distribution.

## Constructors

```@docs
HMMRBM.SparseTransitions
```

## Accessors

```@docs
HMMRBM.state_count(::HMMRBM.SparseTransitions)
```

## Training support

```@docs
HMMRBM.baum_welch_transition_update!(::HMMRBM.SparseTransitions, ::HMMRBM.SparseTransitions, ::AbstractMatrix)
```

# Annealed transition matrices

`AnnealedTransitions` represents a row-stochastic transition matrix with a parameter β that controls the inverse temperature of the model during training. Lower values lead to more uniform transitions; higher values to sparser transitions.

## Constructors

```@docs
HMMRBM.AnnealedTransitions
```

## Accessors

```@docs
HMMRBM.state_count(::HMMRBM.AnnealedTransitions)
```

## Training support

```@docs
HMMRBM.baum_welch_transition_update!(::HMMRBM.AnnealedTransitions, ::HMMRBM.AnnealedTransitions, ::AbstractMatrix)
```