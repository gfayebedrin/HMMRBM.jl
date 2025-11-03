# RBMDistribution.jl

# --- RBMEmission and RBMEmissionFamily definitions ---

"""
    RBMEmission(rbm, hidden, l2)

Concrete emission distribution backed by an RBM and a single hidden vector. The `l2`
value stores the regularisation strength used during training.
"""
struct RBMEmission{R,H} <: Distribution
    rbm::R
    hidden::H
    l2::Real
end

"""
    RBMEmissionFamily(rbm, l2)

Distribution family that turns hidden vectors into `RBMEmission` instances using the
supplied RBM and regularisation coefficient.
"""
struct RBMEmissionFamily{R} <: DistributionFamily{RBMEmission{R}}
    rbm::R
    l2::Real
end

# -- Accessors ---

hidden(dist::RBMEmission) = dist.hidden

rbm(dist::RBMEmission) = dist.rbm
rbm(dist::RBMEmissionFamily) = dist.rbm

l2(dist::RBMEmission) = dist.l2
l2(dist::RBMEmissionFamily) = dist.l2

# --- Distribution interface ---

family(dist::RBMEmission{R,H}) where {R,H} = RBMEmissionFamily{R}(dist.rbm, dist.l2)

parameter(dist::RBMEmission) = dist.hidden

Random.rand(dist::RBMEmission) = RestrictedBoltzmannMachines.sample_v_from_h(dist.rbm, dist.hidden)
Random.rand(::AbstractRNG, dist::RBMEmission) = Random.rand(dist)

DensityInterface.DensityKind(::RBMEmission) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(dist::RBMEmission, obs::AbstractVector)
    dot(rbm(dist).visible.par, obs) + dot(obs, rbm(dist).w, hidden(dist)) - sum(log1pexp.(rbm(dist).visible.par[:] .+ rbm(dist).w * hidden(dist)))
end


# --- DistributionFamily interface ---

function distribution(dist::RBMEmissionFamily, hidden)
    RBMEmission(dist.rbm, hidden, dist.l2)
end

"""
    baum_value_gradient_hessian(dist, obs_seq, γ)

Prepare the objective, gradient, and Hessian evaluations required by the Baum–Welch
update for a single HMM state. The returned closure accepts a hidden vector and produces
the corresponding value, gradient, and optional Hessian information with respect to that
vector.
"""
function baum_value_gradient_hessian(dist::RBMEmissionFamily, obs_seq::AbstractVector, γⱼ::AbstractVector)

    rbm_model = rbm(dist)
    l2_penalty = l2(dist)

    gᵥ = rbm_model.visible.par[:] # Column vector indexed by i (visible units)
    γⱼoW = γⱼ' * reduce(hcat, obs_seq)' * rbm_model.w # Row vector indexed by μ (hidden units)
    ∑ₜγⱼₜ = sum(γⱼ)
    ∑ₜγⱼₜWᵀ = ∑ₜγⱼₜ * rbm_model.w' # Matrix indexed by μ (hidden units), i (visible units)
    ∑ₜγⱼₜW²ᵀ = ∑ₜγⱼₜ * (rbm_model.w .^ 2)' # Matrix indexed by μ (hidden units), i (visible units)

    function value_gradient_hessian(hidden::AbstractVector{<:Real})
        value = γⱼoW * hidden - ∑ₜγⱼₜ * sum(log1pexp.(gᵥ .+ rbm_model.w * hidden)) - l2_penalty * sum(abs2, hidden) |> only

        grad = γⱼoW' - ∑ₜγⱼₜWᵀ * σ.(gᵥ .+ rbm_model.w * hidden) - 2l2_penalty * hidden

        hess = Diagonal(-∑ₜγⱼₜW²ᵀ * σ_prime.(gᵥ .+ rbm_model.w * hidden) .- 2l2_penalty)

        return value, grad, hess
    end

    return value_gradient_hessian
end

"""
HMMs using RBMEmission must have hyperparameter `l2` for regularization.
"""
function DensityInterface.logdensityof(hmm::SingleSeqHMM{<:MultiSeqHMM{<:Any,<:RBMEmissionFamily,<:Any,<:Any,<:Any},<:Any})
    l2 = hyperparameters(hmm).l2
    norm2 = sum(sum(abs2, hidden(d)) for d in obs_distributions(hmm))
    return -l2 * norm2
end
