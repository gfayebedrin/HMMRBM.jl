# Multi-sequence Hidden Markov Models

`MultiSeqHMM` wraps several single-sequence HMMs `SingleSeqHMM` that share emission parameters. It describes one initial distribution and transition matrix for each observed sequence, while the emission parameter is shared.

## Types

```@docs
MultiSeqHMM
```

```@docs
SingleSeqHMM
```

```@docs
Distribution
```

```@docs
DistributionFamily
```