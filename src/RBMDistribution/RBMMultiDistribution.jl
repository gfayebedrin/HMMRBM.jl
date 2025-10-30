# RBMMultiDistribution.jl

# --- RBMMultiEmission and RBMMultiEmissionFamily definitions ---

struct RBMMultiEmission{R,Θ} <: Distribution
    rbm::R
    θ::Θ
    l2::Real
end

struct RBMMultiEmissionFamily{R} <: DistributionFamily{RBMMultiEmission{R}}
    rbm::R
    l2::Real
end


# --- Constructors ---

function RBMMultiEmission(rbm, hiddens::AbstractArray, logits::AbstractArray, l2::Real)
    RBMMultiEmission(rbm, stack_vector_matrix(logits, hiddens), l2)
end


# --- Accessors ---

θ(dist::RBMMultiEmission) = dist.θ

rbm(dist::RBMMultiEmission) = dist.rbm
rbm(dist::RBMMultiEmissionFamily) = dist.rbm

l2(dist::RBMMultiEmission) = dist.l2
l2(dist::RBMMultiEmissionFamily) = dist.l2

logits(dist::RBMMultiEmission) = first(unstack_vector_matrix(θ(dist)))
hiddens(dist::RBMMultiEmission) = last(unstack_vector_matrix(θ(dist)))

# --- Distribution interface ---

family(dist::RBMMultiEmission{R,Θ}) where {R,Θ} = RBMMultiEmissionFamily{R}(dist.rbm, dist.l2)

parameter(dist::RBMMultiEmission) = θ(dist)

function Random.rand(dist::RBMMultiEmission)
    aₖ = softmax(logits(dist))
    k = rand(Categorical(aₖ))
    RestrictedBoltzmannMachines.sample_v_from_h(rbm(dist), selectdim(hiddens(dist), 1, k))
end
Random.rand(::AbstractRNG, dist::RBMMultiEmission) = Random.rand(dist)

DensityInterface.DensityKind(::RBMMultiEmission) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(dist::RBMMultiEmission, obs::AbstractVector)
    aₖ = softmax(logits(dist))
    rbm = rbm(dist)
    hiddens = hiddens(dist)

    dot(rbm.visible.par[:], obs) + dot(obs, rbm.w, hiddens' * aₖ) - sum(log1pexp.(rbm.visible.par[:] .+ rbm.w * hiddens') * aₖ)
end

function distribution(dist::RBMMultiEmissionFamily, θ)
    RBMMultiEmission(rbm(dist), θ, l2(dist))
end

function baum_value_gradient_hessian(dist::RBMMultiEmissionFamily, obs_seq::AbstractVector, γⱼ::AbstractVector)

    rbm = rbm(dist)
    l2 = l2(dist)
    
    gᵥ = rbm.visible.par[:] # Column vector indexed by i (visible units)
    o = reduce(hcat, obs_seq)' # Matrix indexed by t (time), i (visible units)
    ogᵥ = o * gᵥ # Column vector indexed by t (time)
    oW = o * rbm.w # Matrix indexed by t (time), μ (hidden units)


    function value_gradient_hessian(θ_vec::AbstractVector{<:Real})

        M = size(rbm.hidden, 2)
        θ = similar(θ_vec, length(θ_vec) ÷ (M+1), M+1)
        θ[:] .= θ_vec
        logits, hiddens = unstack_vector_matrix(θ)
        aₖ = softmax(logits)

        oWhᵀ = oW * hiddens' # Matrix indexed by t (time), k (mixture components)


    end


    error("Not implemented")
end


# --- Integration with multi-sequence HMMs ---

function MultiSeqHMM(; inits, transitions, rbms::AbstractVector, hiddens::AbstractArray{<:Any,3}, logits::AbstractMatrix, l2::Real=0.0)
    families = RBMMultiEmissionFamily.(rbms, l2)
    θ = stack_vector_matrix(logits, hiddens)
    MultiSeqHMM(inits, transitions, families, θ, (;l2))
end

"""
HMMs using RBMMultiEmission must have hyperparameter `l2` for regularization.
"""
function DensityInterface.logdensityof(hmm::SingleSeqHMM{<:MultiSeqHMM{<:Any,<:RBMMultiEmissionFamily,<:Any,<:Any,<:Any,<:Any},<:Any})
    l2 = hyperparameters(hmm).l2
    norm2 = sum(sum(abs2, hiddens(d)) for d in obs_distributions(hmm))
    return -l2 * norm2
end
