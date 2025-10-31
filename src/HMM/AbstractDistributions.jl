"""
The following functions must be defined:
- `family(::T)::DistributionFamily{T} where T<:Distribution`
- `parameter(::T) where T<:Distribution`
- `Random.rand(::AbstractRNG, dist::T) where T<:Distribution`
- `DensityInterface.logdensityof(dist::T, obs) where T<:Distribution`
"""
abstract type Distribution end

"""
The following functions must be defined:
- `distribution(::DistributionFamily{T}, parameter)::T where T<:Distribution`
- `baum_value_gradient_hessian(dist::Distribution, obs_seq, γ)::Function`
where the returned function has signature `value_gradient_hessian(parameter)::Tuple{Real,AbstractVector,Union{AbstractMatrix,Nothing}}`
"""
abstract type DistributionFamily{T<:Distribution} end

function distribution end

function family end
function parameter end

function baum_value_gradient_hessian end
baum_value_gradient_hessian(dist::Distribution, obs_seq, γ) = baum_value_gradient_hessian(family(dist), obs_seq, γ)
