# RBM emission families

HMMRBM relies on Restricted Boltzmann Machines to describe observation models. A single
hidden vector per state is handled through `RBMEmission`, while mixtures reuse the same
infrastructure with a stacked parameter representation.

## API reference

```@docs
HMMRBM.RBMEmissionFamily
HMMRBM.RBMEmission
HMMRBM.RBMMultiEmissionFamily
HMMRBM.RBMMultiEmission
HMMRBM.baum_value_gradient_hessian
```
