# Grouped transition matrices

`GroupedTransitions` stores transition probabilities for models whose states can be
partitioned into equally sized groups. Entries on the diagonal blocks stay fully
parameterised while every off-diagonal block shares a single probability per pair of
groups. This reduces the number of learnable parameters and keeps the transition matrix
stochastic after each Baum–Welch update.

```julia
julia> trans = HMMRBM.GroupedTransitions(3, 2);  # 3 groups, 2 states each

julia> size(trans)
(6, 6)

julia> trans[1, 4]
0.16666666666666666
```

Grouped transitions can be passed directly to the `transitions` field of `MultiSeqHMM`
and will be kept consistent during the EM loop via
`HMMRBM.baum_welch_transition_update!`.

## Constructors

```@docs
HMMRBM.GroupedTransitions
```

## Accessors

```@docs
HMMRBM.group_count
HMMRBM.states_per_group
HMMRBM.state_count(::HMMRBM.GroupedTransitions)
```

## Training support

```@docs
HMMRBM.baum_welch_transition_update!(::HMMRBM.GroupedTransitions, ::HMMRBM.GroupedTransitions, ::AbstractMatrix)
```
