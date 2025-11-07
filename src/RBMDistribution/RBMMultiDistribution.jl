# RBMMultiDistribution.jl

# --- RBMMultiEmission and RBMMultiEmissionFamily definitions ---

"""
    RBMMultiEmission(rbm, θ, l2)

Emission distribution representing an RBM backed mixture. The stacked parameter `θ`
contains logits and hidden vectors, while `l2` records the regularisation strength.
"""
struct RBMMultiEmission{Θ} <: Distribution
    rbm::RestrictedBoltzmannMachines.RBM
    θ::Θ
    l2::Real
end

"""
    RBMMultiEmissionFamily(rbm, l2)

Distribution family that converts stacked logits and hidden vectors into
`RBMMultiEmission` instances.
"""
struct RBMMultiEmissionFamily <: DistributionFamily{RBMMultiEmission}
    rbm::RestrictedBoltzmannMachines.RBM
    l2::Real
end


# --- Constructors ---

function RBMMultiEmission(rbm::RestrictedBoltzmannMachines.RBM, hiddens::AbstractArray, logits::AbstractArray, l2::Real)
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

family(dist::RBMMultiEmission{Θ}) where {Θ} = RBMMultiEmissionFamily(dist.rbm, dist.l2)

parameter(dist::RBMMultiEmission) = θ(dist)

function Random.rand(rng::AbstractRNG, dist::RBMMultiEmission)
    aₖ = softmax(logits(dist))
    k = rand(rng, Categorical(aₖ))
    RestrictedBoltzmannMachines.sample_v_from_h(rbm(dist), selectdim(hiddens(dist), 1, k))
end
Random.rand(dist::RBMMultiEmission) = Random.rand(Random.default_rng(), dist)

DensityInterface.DensityKind(::RBMMultiEmission) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(dist::RBMMultiEmission, obs::AbstractVector)
    log_aₖ = logits(dist) .- logsumexp(logits(dist))
    E = -RestrictedBoltzmannMachines.energy(rbm(dist), obs, hiddens(dist)')
    F = RestrictedBoltzmannMachines.free_energy_h(rbm(dist), hiddens(dist)')
    logsumexp(log_aₖ + F - E)
end

function distribution(dist::RBMMultiEmissionFamily, θ)
    RBMMultiEmission(rbm(dist), θ, l2(dist))
end

mutable struct BaumCache{T}
    lastθ::Vector{T}
    log_aₖ::Vector{T}
    hiddens::Matrix{T}
    log_prob_component::Matrix{T}
    log_prob_state::Vector{T}
    is_initialized::Bool
end

function BaumCache(::Type{T}) where {T}
    return BaumCache{T}(
        T[],
        T[],
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        T[],
        false,
    )
end

function baum_value_gradient(dist::RBMMultiEmissionFamily, obs_seq::AbstractVector, γⱼ::AbstractVector)

    # γⱼ: Vector indexed by t (time)
    obs_mat = obs_seq isa Base.Slices ? Base.parent(obs_seq) : reduce(hcat, obs_seq)
    o = obs_mat' # Matrix indexed by t (time), i (visible units)
    W = rbm(dist).w # Matrix indexed by i (visible units), μ (hidden units)
    oW = o * W # Matrix indexed by t (time), μ (hidden units)
    Eₒ = RestrictedBoltzmannMachines.energy(rbm(dist).visible, o') # Vector indexed by t (time)

    n_hidden = size(W, 2)
    n_time = length(obs_seq)

    function initialize!(cache::BaumCache{T}; n_time, n_hidden, n_mix) where {T}
        cache.lastθ = Vector{T}(undef, (n_hidden + 1) * n_mix)
        cache.log_aₖ = Vector{T}(undef, n_mix)
        cache.hiddens = Matrix{T}(undef, n_mix, n_hidden)
        cache.log_prob_component = Matrix{T}(undef, n_time, n_mix)
        cache.log_prob_state = Vector{T}(undef, n_time)
        cache.is_initialized = true
    end

    # Common calculations

    cache = BaumCache(eltype(γⱼ))

    function calculate_common!(θ::Vector)
        if !cache.is_initialized
            n_mix = length(θ) ÷ (n_hidden + 1)
            initialize!(cache; n_time, n_hidden, n_mix)
        end

        if θ != cache.lastθ
            copy!(cache.lastθ, θ)
            logits, hiddens = unstack_vector_matrix(reshape(θ, :, n_hidden + 1))
            cache.log_aₖ .= logits .- logsumexp(logits)
            cache.hiddens .= hiddens
            cache.log_prob_component .= log_P_v_given_h(rbm(dist), Eₒ, oW', hiddens')
            cache.log_prob_state .= logsumexp(cache.log_aₖ' .+ cache.log_prob_component; dims=2)
        end
    end

    # Objective and gradient

    function f(θ::Vector)
        calculate_common!(θ)
        γⱼ ⋅ cache.log_prob_state - l2(dist) * sum(abs2, cache.hiddens)
    end

    function grad!(∇::Vector, θ::Vector)
        calculate_common!(θ)

        responsability = exp.(cache.log_aₖ' .+ cache.log_prob_component .- cache.log_prob_state) .- exp.(cache.log_aₖ') # Matrix indexed by t (time), k (mixture components)
        γresp = γⱼ .* responsability # Matrix indexed by t (time), k (mixture components)

        grad_logits = vec(sum(γresp, dims=1)) # Vector indexed by k (mixture components)

        Wᵀσ = W' * RestrictedBoltzmannMachines.mean_v_from_h(rbm(dist), cache.hiddens') # Matrix indexed by μ (hidden units), k (mixture components)

        grad_hiddens = γresp' * oW .- grad_logits .* Wᵀσ' .- 2 * l2(dist) * cache.hiddens

        ∇ .= stack_vector_matrix(grad_logits, grad_hiddens)[:]
    end

    return f, grad!
end


"""
HMMs using RBMMultiEmission must have hyperparameter `l2` for regularization.
"""
function DensityInterface.logdensityof(hmm::SingleSeqHMM{<:MultiSeqHMM{<:Any,<:RBMMultiEmissionFamily,<:Any,<:Any,<:Any},<:Any})
    l2 = hyperparameters(hmm).l2
    norm2 = sum(sum(abs2, hiddens(d)) for d in obs_distributions(hmm))
    return -l2 * norm2
end
