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


"""
    baum_value_gradient(dist::RBMEmissionFamily, obs_seq, γⱼ)

Objective and gradient of the Baum-Welch update of the hidden vector of one state.

## Optimisation
The `γⱼ`-weighted sums over time are computed once, so each evaluation costs
`O(n_visible * n_hidden)` independently of the sequence length.
"""
function baum_value_gradient(dist::RBMEmissionFamily, obs_seq::AbstractVector, γⱼ::AbstractVector)

    obs_mat = obs_seq isa Base.Slices ? Base.parent(obs_seq) : reduce(hcat, obs_seq)
    r = rbm(dist)
    Wᵀv̄ = r.w' * (obs_mat * γⱼ)
    Ē = γⱼ ⋅ vec(RestrictedBoltzmannMachines.energy(r.visible, obs_mat))
    Γ = sum(γⱼ)

    lasth = fill(convert(eltype(γⱼ), NaN), length(r.hidden))
    inputs = similar(γⱼ, length(r.visible))
    cumulants = similar(inputs)

    function calculate_common!(h::Vector)
        if h != lasth
            copy!(lasth, h)
            inputs .= RestrictedBoltzmannMachines.inputs_v_from_h(r, h)
        end
    end

    function f(h::Vector)
        calculate_common!(h)
        Wᵀv̄ ⋅ h - Ē - Γ * sum(cgfs!(cumulants, r.visible, inputs)) - l2(dist) * sum(abs2, h)
    end

    function grad!(∇ₕ::Vector, h::Vector)
        calculate_common!(h)
        mean_v = RestrictedBoltzmannMachines.mean_from_inputs(r.visible, inputs)
        ∇ₕ .= Wᵀv̄ .- Γ .* (r.w' * mean_v) .- 2 * l2(dist) .* h
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
