# Multi-sequence Hidden Markov Models

HMMRBM extends `HiddenMarkovModels.jl` with multi-sequence containers that reuse the same
emission parameters across several observation streams. Each sequence keeps its own
initial distribution and transition matrix, while a shared parameter tensor drives the
Restricted Boltzmann Machine emissions.

## Types

```@docs
HMMRBM.MultiSeqHMM
HMMRBM.SingleSeqHMM
HMMRBM.Distribution
HMMRBM.DistributionFamily
```

## Functions

```@docs
HMMRBM.inits
HMMRBM.loginits
HMMRBM.transitions
HMMRBM.logtransitions
HMMRBM.emissions
HMMRBM.emission
HMMRBM.emission_parameters
HMMRBM.hyperparameters
HMMRBM.parent
HMMRBM.sequence_index
HMMRBM.hmm_rbm
HMMRBM.distribution
HMMRBM.family
HMMRBM.parameter
HMMRBM.state_count
HMMRBM.baum_value_gradient
HMMRBM.baum_welch_transition_update!
```

## Saving

Persistence helpers are provided when the optional dependency
[`HDF5.jl`](https://juliaio.github.io/HDF5.jl/stable/) is available. Loading the package
with `using HDF5` triggers the extension that defines the persistence methods.

```@docs
HMMRBM.MultiSeqHMM_to_group!
HMMRBM.MultiSeqHMM_from_group
HMMRBM.save_hmm
HMMRBM.load_hmm
```
