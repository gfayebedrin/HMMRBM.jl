# Internal API

The following helpers are documented but not exported. They remain available as
`HMMRBM` internals for advanced workflows and extension hooks.

## Utilities

```@docs
HMMRBM.my_mult
HMMRBM.stack_vector_matrix
HMMRBM.unstack_vector_matrix
```

## RBM Utilities

```@docs
HMMRBM.log_P_v_given_h
HMMRBM.∂ₕlog_P_v_given_h
HMMRBM.∂ₕ²log_P_v_given_h
HMMRBM.cgfs!
```

## Algorithm Extensions

```@docs
HiddenMarkovModels.baum_welch(::HMMRBM.MultiSeqHMM, ::AbstractVector{<:AbstractVector})
```

```@docs
DensityInterface.logdensityof(::HMMRBM.SingleSeqHMM{<:HMMRBM.MultiSeqHMM{<:Any,<:HMMRBM.RBMEmissionFamily,<:Any,<:Any,<:Any},<:Any})
DensityInterface.logdensityof(::HMMRBM.SingleSeqHMM{<:HMMRBM.MultiSeqHMM{<:Any,<:HMMRBM.RBMMultiEmissionFamily,<:Any,<:Any,<:Any},<:Any})
```
