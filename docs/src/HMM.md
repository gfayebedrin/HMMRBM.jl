# Multi-sequence Hidden Markov Models

HMMRBM extends `HiddenMarkovModels.jl` with multi-sequence containers that reuse the same
emission parameters across several observation streams. Each sequence keeps its own
initial distribution and transition matrix, while a shared parameter tensor drives the
Restricted Boltzmann Machine emissions.

## API reference

```@docs
HMMRBM.MultiSeqHMM
HMMRBM.SingleSeqHMM
HMMRBM.hmm_rbm
HMMRBM.MultiSeqHMM_to_group!
HMMRBM.MultiSeqHMM_from_group
HMMRBM.save_hmm
HMMRBM.load_hmm
```
