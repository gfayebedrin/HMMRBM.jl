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

family(dist::RBMEmission) = RBMEmissionFamily(dist.rbm, dist.l2)

parameter(dist::RBMEmission) = dist.hidden

Random.rand(dist::RBMEmission) = RestrictedBoltzmannMachines.sample_v_from_h(dist.rbm, dist.hidden)
Random.rand(::AbstractRNG, dist::RBMEmission) = Random.rand(dist)

DensityInterface.DensityKind(::RBMEmission) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(dist::RBMEmission, obs::AbstractVector)
    log_P_v_given_h(rbm(dist), obs, hidden(dist))
end


# --- DistributionFamily interface ---

function distribution(dist::RBMEmissionFamily, hidden)
    RBMEmission(dist.rbm, hidden, dist.l2)
end


function baum_value_gradient(dist::RBMEmissionFamily, obs_seq::AbstractVector, γⱼ::AbstractVector)

    obs_mat = obs_seq isa Base.Slices ? Base.parent(obs_seq) : reduce(hcat, obs_seq)
    Wᵀv = rbm(dist).w' * obs_mat
    Eᵥ = RestrictedBoltzmannMachines.energy(rbm(dist).visible, obs_mat)

    out_f = similar(γⱼ)
    out_∇ = similar(γⱼ, (length(rbm(dist).hidden), length(γⱼ)))

    function f(h::Vector)
        γⱼ ⋅ log_P_v_given_h!(out_f, rbm(dist), Eᵥ, Wᵀv, h) - l2(dist) * sum(abs2, h)
    end

    function grad!(∇ₕ::Vector, h::Vector)
        ∇ₕ .= ∂ₕlog_P_v_given_h!(out_∇, rbm(dist), Wᵀv, h) * γⱼ .- 2 * l2(dist) * h
    end

    return f, grad!
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
