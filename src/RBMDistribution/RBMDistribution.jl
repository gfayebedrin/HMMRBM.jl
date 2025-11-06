# RBMDistribution.jl

# --- RBMEmission and RBMEmissionFamily definitions ---

"""
    RBMEmission(rbm, hidden, l2)

Concrete emission distribution backed by an RBM and a single hidden vector. The `l2`
value stores the regularisation strength used during training.
"""
struct RBMEmission{H} <: Distribution
    rbm::RestrictedBoltzmannMachines.RBM
    hidden::H
    l2::Real
end

"""
    RBMEmissionFamily(rbm, l2)

Distribution family that turns hidden vectors into `RBMEmission` instances using the
supplied RBM and regularisation coefficient.
"""
struct RBMEmissionFamily <: DistributionFamily{RBMEmission}
    rbm::RestrictedBoltzmannMachines.RBM
    l2::Real
end

# -- Accessors ---

hidden(dist::RBMEmission) = dist.hidden

rbm(dist::RBMEmission) = dist.rbm
rbm(dist::RBMEmissionFamily) = dist.rbm

l2(dist::RBMEmission) = dist.l2
l2(dist::RBMEmissionFamily) = dist.l2

# --- Distribution interface ---

family(dist::RBMEmission{H}) where {H} = RBMEmissionFamily(dist.rbm, dist.l2)

parameter(dist::RBMEmission) = dist.hidden

Random.rand(dist::RBMEmission) = RestrictedBoltzmannMachines.sample_v_from_h(dist.rbm, dist.hidden)
Random.rand(::AbstractRNG, dist::RBMEmission) = Random.rand(dist)

DensityInterface.DensityKind(::RBMEmission) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(dist::RBMEmission, obs::AbstractVector)
    -RestrictedBoltzmannMachines.energy(rbm(dist), obs, hidden(dist)) + RestrictedBoltzmannMachines.free_energy_h(rbm(dist), hidden(dist))
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

    obs_mat = reduce(hcat, obs_seq)
    ∑ₜγⱼₜ = sum(γⱼ)
    Wᵀv = rbm(dist).w' * obs_mat
    Eᵥ = RestrictedBoltzmannMachines.energy(rbm(dist).visible, obs_mat)

    function value_gradient_hessian!(grad, hess, h::AbstractVector{<:Real})
        value = γⱼ ⋅ log_P_v_given_h(rbm(dist), Eᵥ, Wᵀv, h) - l2(dist) * sum(abs2, h)
        grad .= ∂ₕlog_P_v_given_h(rbm(dist), Wᵀv, h) * γⱼ .- 2 * l2(dist) * h
        hess .= ∂ₕ²log_P_v_given_h(rbm(dist), h) * ∑ₜγⱼₜ - 2 * l2(dist) * I

        return value, grad, hess
    end

    return value_gradient_hessian!
end

"""
    logdensityof(hmm::SingleSeqHMM)

Log-density regularisation term for `SingleSeqHMM` instances whose emissions are
`RBMEmission`.

HMMs using RBMEmission must have hyperparameter `l2` for regularization.
"""
function DensityInterface.logdensityof(hmm::SingleSeqHMM{<:MultiSeqHMM{<:Any,<:RBMEmissionFamily,<:Any,<:Any,<:Any},<:Any})
    l2 = hyperparameters(hmm).l2
    norm2 = sum(sum(abs2, hidden(d)) for d in obs_distributions(hmm))
    return -l2 * norm2
end
