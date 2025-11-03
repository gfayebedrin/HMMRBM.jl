# Package guide

HMMRBM.jl couples Hidden Markov Models with Restricted Boltzmann Machine (RBM) emission models. It extends [`HiddenMarkovModels.jl`](https://gdalle.github.io/HiddenMarkovModels.jl/stable/) by providing multi-sequence training utilities and RBM-based observation families sourced from [`RestrictedBoltzmannMachines.jl`](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable/).

It implements hidden Markov models trained on multiple sequences of observations obtained from distinct emission distributions that share the emission parameter.
The result of training is a transition matrix for each sequence and one emission parameter for all sequences.

It also implements emission distributions that use a pre-trained restricted Boltzmann machine as parametric emission, and hidden vector as emission parameter.
This allows training HMMs with latent-aligned RBMs as the emission distribution.

Trained HMMs can be persisted with [HDF5.jl](https://juliaio.github.io/HDF5.jl/stable/).

## Installation

HMMRBM.jl is registered in the [BrainRegistry.jl](https://github.com/gfayebedrin/BrainRegistry.jl) registry.

To activate the registry, do

```julia
using Pkg
pkg"registry add https://github.com/gfayebedrin/BrainRegistry.jl"
```

This only needs to be done once per Julia installation.

To install the package, do

```julia
using Pkg
Pkg.add("HMMRBM")
Pkg.add("HDF5") # Optional
```