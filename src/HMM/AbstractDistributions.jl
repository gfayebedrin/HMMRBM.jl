"""
    Distribution

The following functions must be defined:
- `family(::T)::DistributionFamily{T} where T<:Distribution`
- `parameter(::T) where T<:Distribution`
- `Random.rand(::AbstractRNG, dist::T) where T<:Distribution`
- `DensityInterface.logdensityof(dist::T, obs) where T<:Distribution`
"""
abstract type Distribution end

"""
    DistributionFamily{T<:Distribution}

The following functions must be defined:
- `distribution(::DistributionFamily{T}, parameter)::T where T<:Distribution`
- `baum_value_gradient_hessian(dist::Distribution, obs_seq, γ)::Function`
where the returned function has signature `value_gradient_hessian(parameter)::Tuple{Real,AbstractVector,Union{AbstractMatrix,Nothing}}`
"""
abstract type DistributionFamily{T<:Distribution} end

"""
    distribution(family, parameter)

Get the distribution instance associated with the given `family` and `parameter`.
"""
function distribution end

"""
    family(dist)

Get the distribution family associated with the given `dist`.
"""
function family end

"""
    parameter(dist)

Get the parameter object associated with the given `dist`.
"""
function parameter end

"""
    baum_value_gradient_hessian(dist, obs_seq, γ)
    baum_value_gradient_hessian(family, obs_seq, γ)

Get a function that computes the Baum-Welch value, gradient, and Hessian for the given `dist`, observation sequence `obs_seq`, and posteriors `γ`.
"""
function baum_value_gradient_hessian end
baum_value_gradient_hessian(dist::Distribution, obs_seq, γ) = baum_value_gradient_hessian(family(dist), obs_seq, γ)
