# RBMMultiDistribution.jl

# --- RBMMultiEmission and RBMMultiEmissionFamily definitions ---

"""
    RBMMultiEmission(rbm, θ, l2)

Emission distribution representing an RBM backed mixture. The stacked parameter `θ`
contains logits and hidden vectors, while `l2` records the regularisation strength.
"""
struct RBMMultiEmission{R,Θ} <: Distribution
    rbm::R
    θ::Θ
    l2::Real
end

"""
    RBMMultiEmissionFamily(rbm, l2)

Distribution family that converts stacked logits and hidden vectors into
`RBMMultiEmission` instances.
"""
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
    rbm_model = rbm(dist)
    hidden_states = hiddens(dist)

    dot(rbm_model.visible.par[:], obs) + dot(obs, rbm_model.w, hidden_states' * aₖ) - sum(log1pexp.(rbm_model.visible.par[:] .+ rbm_model.w * hidden_states') * aₖ)
end

function distribution(dist::RBMMultiEmissionFamily, θ)
    RBMMultiEmission(rbm(dist), θ, l2(dist))
end

"""
    baum_value_gradient_hessian(dist, obs_seq, γ)

Return a closure that evaluates the Baum–Welch objective and gradient with respect to the
stacked logits and hidden vectors used by a mixture RBM emission family. The Hessian is
omitted for this variant.
"""
function baum_value_gradient_hessian(dist::RBMMultiEmissionFamily, obs_seq::AbstractVector, γⱼ::AbstractVector)

    rbm_model = rbm(dist)
    l2_penalty = l2(dist)

    # γⱼ: Vector indexed by t (time)
    gᵥ = rbm_model.visible.par[:] # Column vector indexed by i (visible units)
    o = reduce(hcat, obs_seq)' # Matrix indexed by t (time), i (visible units)
    ogᵥ = o * gᵥ # Vector indexed by t (time)
    oW = o * rbm_model.w # Matrix indexed by t (time), μ (hidden units)


    function value_gradient_hessian(θ_vec::AbstractVector{<:Real})

        M = size(rbm_model.w, 2)
        θ = similar(θ_vec, length(θ_vec) ÷ (M + 1), M + 1)
        θ[:] .= θ_vec
        logits, hiddens = unstack_vector_matrix(θ)
        log_aₖ = logits .- logsumexp(logits) # Vector indexed by k (mixture components)

        Whᵀ = rbm_model.w * hiddens' # Matrix indexed by i (visible units), k (mixture components)
        gᵥpWhᵀ = gᵥ .+ Whᵀ # Matrix indexed by i (visible units), k (mixture components)
        ∑ᵢlog1pexpgᵥpWhᵀ = sum(log1pexp.(gᵥpWhᵀ); dims=1) # Row vector indexed by k (mixture components)

        log_prob_component = ogᵥ .+ oW * hiddens' .- ∑ᵢlog1pexpgᵥpWhᵀ # Matrix indexed by t (time), k (mixture components)
        log_prob_state = logsumexp(log_aₖ' .+ log_prob_component; dims=2) # Column vector indexed by t (time)
        responsability = exp.(log_aₖ' .+ log_prob_component .- log_prob_state) .- exp.(log_aₖ') # Matrix indexed by t (time), k (mixture components)
        γresp = γⱼ .* responsability # Matrix indexed by t (time), k (mixture components)

        value = γⱼ ⋅ log_prob_state - l2_penalty * sum(abs2, hiddens)

        grad_logits = vec(sum(γresp, dims=1)) # Vector indexed by k (mixture components)

        Wᵀσ = rbm_model.w' * σ.(gᵥpWhᵀ) # Matrix indexed by μ (hidden units), k (mixture components)

        grad_hiddens = γresp' * oW .- grad_logits .* Wᵀσ' .- 2l2_penalty * hiddens

        grad_θ_vec = stack_vector_matrix(grad_logits, grad_hiddens)[:]

        return value, grad_θ_vec, nothing
    end

    return value_gradient_hessian
end


"""
HMMs using RBMMultiEmission must have hyperparameter `l2` for regularization.
"""
function DensityInterface.logdensityof(hmm::SingleSeqHMM{<:MultiSeqHMM{<:Any,<:RBMMultiEmissionFamily,<:Any,<:Any,<:Any},<:Any})
    l2 = hyperparameters(hmm).l2
    norm2 = sum(sum(abs2, hiddens(d)) for d in obs_distributions(hmm))
    return -l2 * norm2
end
