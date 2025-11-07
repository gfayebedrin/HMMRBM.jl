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
    baum_value_gradient(dist, obs_seq, γ)

Prepare the objective and gradient evaluations required by the Baum-Welch
update for a single HMM state.

Returns a tuple of two functions `(f, grad!)`
where `f(x)` returns the objective value at `x`,
and `grad!(∇ₓf, x)` fills `∇ₓf` with the gradient at `x`.
"""
function baum_value_gradient end
baum_value_gradient(dist::Distribution, obs_seq, γ) = baum_value_gradient(family(dist), obs_seq, γ)
