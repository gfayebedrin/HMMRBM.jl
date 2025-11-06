# Internal API

The following helpers are documented but not exported. They remain available as
`HMMRBM` internals for advanced workflows and extension hooks.

## Optimisation

```@docs
HMMRBM.gradient_descent
HMMRBM.my_mult
```

## Multi-sequence Helpers

```@docs
HMMRBM.state_count(::HMMRBM.MultiSeqHMM)
HMMRBM.state_count(::HMMRBM.SingleSeqHMM)
```

## RBM Utilities

```@docs
HMMRBM.stack_vector_matrix
HMMRBM.unstack_vector_matrix
HMMRBM.log_P_v_given_h
HMMRBM.∂ₕlog_P_v_given_h
HMMRBM.∂ₕ²log_P_v_given_h
```

## Algorithm Extensions

```@docs
HiddenMarkovModels.baum_welch(::HMMRBM.MultiSeqHMM, ::AbstractVector{<:AbstractVector})
```

```@docs
DensityInterface.logdensityof(::HMMRBM.SingleSeqHMM{<:HMMRBM.MultiSeqHMM{<:Any,<:HMMRBM.RBMEmissionFamily,<:Any,<:Any,<:Any},<:Any})
DensityInterface.logdensityof(::HMMRBM.SingleSeqHMM{<:HMMRBM.MultiSeqHMM{<:Any,<:HMMRBM.RBMMultiEmissionFamily,<:Any,<:Any,<:Any},<:Any})
```
