# HMMRBM.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://gfayebedrin.github.io/HMMRBM.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://gfayebedrin.github.io/HMMRBM.jl/dev)
[![Build Status](https://github.com/gfayebedrin/HMMRBM.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/gfayebedrin/HMMRBM.jl/actions/workflows/ci.yml)

HMMRBM.jl couples multi-sequence Hidden Markov Models with Restricted Boltzmann Machine
emission distributions. It reuses a shared RBM parameter across several observation
streams, making it convenient to fit HMMs whose emissions are tied through a common latent
representation.

## Features
- Multi-sequence HMM container that exposes each sequence as a standard `HiddenMarkovModels.jl` model
- RBM-backed emission families with optional mixture logits and ℓ2 regularisation
- Baum–Welch integrations that optimise the shared RBM parameters alongside HMM structure
- Optional HDF5 extension for persisting trained models

## Installation

HMMRBM.jl lives in the [BrainRegistry.jl](https://github.com/gfayebedrin/BrainRegistry.jl)
registry. The setup only needs to be run once per Julia installation:

```julia
using Pkg
pkg"registry add https://github.com/gfayebedrin/BrainRegistry.jl"
Pkg.add("HMMRBM")
```

To enable HDF5 serialization helpers, also install `HDF5.jl`; loading it activates the
package extension automatically.

## Quick Start

```julia
using HMMRBM
using HiddenMarkovModels

# Assume `rbms` contains one trained RBM per observation sequence
rbms = [...]  # e.g. Vector{RestrictedBoltzmannMachines.RBM}

hmm0 = HMMRBM.hmm_rbm(rbms, 3; l2=1e-2)

# Observations must be provided as a vector of sequences (one per RBM)
obs_sequences = [...]

hmm_est, logL = HiddenMarkovModels.baum_welch(
    hmm0,
    obs_sequences;
    max_iterations=50,
    loglikelihood_increasing=true,
)
```
