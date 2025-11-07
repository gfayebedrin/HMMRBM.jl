function baum_welch_has_converged(
    logL_evolution::Vector; atol::Real, loglikelihood_increasing::Bool
)
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end-1]
        progress = logL - logL_prev
        if loglikelihood_increasing && progress < min(0, -atol)
            error("Loglikelihood decreased from $logL_prev to $logL in Baum-Welch")
        elseif progress < atol
            return true
        end
    end
    return false
end

function HiddenMarkovModels.baum_welch!(
    fb_storages::Vector{<:HiddenMarkovModels.ForwardBackwardStorage},
    logL_evolution::Vector,
    hmm::MultiSeqHMM,
    obs_sequences::AbstractVector{<:AbstractVector};
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
    callback=(x -> nothing),
)
    controls = [fill(nothing, length(obs_sequences[s])) for s in eachindex(obs_sequences)]
    seq_ends = [(length(obs_sequences[s]),) for s in eachindex(obs_sequences)]
    for iteration in 1:max_iterations
        callback((; iteration, logL_evolution))

        Threads.@threads for (storage, subhmm, obs_seq, ctrl, ends) in zip(fb_storages, adapt(Array, hmm), obs_sequences, controls, seq_ends) |> collect
            HiddenMarkovModels.forward_backward!(storage, subhmm, obs_seq, ctrl; seq_ends=ends)
        end

        push!(logL_evolution, logdensityof(hmm) + sum(sum(fs.logL) for fs in fb_storages))
        fit!(hmm, adapt.(eltype(obs_sequences), fb_storages), obs_sequences)
        if baum_welch_has_converged(logL_evolution; atol, loglikelihood_increasing)
            break
        end
    end
    return nothing
end

"""
    baum_welch(hmm_guess, obs_seq; atol, max_iterations, loglikelihood_increasing)

Apply the Baum-Welch algorithm to estimate the parameters of an HMM on `obs_seq`, starting from `hmm_guess`.

Return a tuple `(hmm_est, loglikelihood_evolution)` where `hmm_est` is the estimated HMM and `loglikelihood_evolution` is a vector of loglikelihood values, one per iteration of the algorithm.

# Keyword arguments

- `atol`: minimum loglikelihood increase at an iteration of the algorithm (otherwise the algorithm is deemed to have converged)
- `max_iterations`: maximum number of iterations of the algorithm
- `loglikelihood_increasing`: whether to throw an error if the loglikelihood decreases
"""
function HiddenMarkovModels.baum_welch(
    hmm_guess::MultiSeqHMM,
    obs_sequences::AbstractVector{<:AbstractVector};
    atol=1e-5,
    max_iterations=100,
    loglikelihood_increasing=true,
    callback=(x -> nothing),
)
    hmm = deepcopy(hmm_guess)

    fb_storages = [
        HiddenMarkovModels.initialize_forward_backward(
            hmm[s],
            obs_sequences[s],
            fill(nothing, length(obs_sequences[s]));
            seq_ends=(length(obs_sequences[s]),),
        )
        for s in eachindex(obs_sequences)
    ]
    logL_evolution = eltype(fb_storages).parameters[1][]
    sizehint!(logL_evolution, max_iterations)
    HiddenMarkovModels.baum_welch!(
        fb_storages,
        logL_evolution,
        hmm,
        obs_sequences;
        atol,
        max_iterations,
        loglikelihood_increasing,
        callback,
    )
    return hmm, logL_evolution
end


function StatsAPI.fit!(
    hmm::MultiSeqHMM,
    fb_storages::Vector{<:HiddenMarkovModels.ForwardBackwardStorage},
    obs_sequences::AbstractVector{<:AbstractVector},
)
    γs = getproperty.(fb_storages, :γ)
    ξs = getproperty.(fb_storages, :ξ)


    # Fit inits
    Threads.@threads for (γ, init, loginit) in zip(γs, inits(hmm), loginits(hmm)) |> collect
        sum!(init, γ)
        sum_to_one!(init)
        loginit .= log.(init)
    end

    # Fit transitions
    Threads.@threads for (ξ, trans, logtrans) in zip(ξs, transitions(hmm), logtransitions(hmm)) |> collect
        scratch = ξ[end]  # use ξ[end] as scratch space since it is zero anyway
        fill!(scratch, zero(eltype(scratch)))
        for t in 1:(length(ξ)-1)
            scratch .+= ξ[t]
        end
        trans .= scratch
        foreach(sum_to_one!, eachrow(trans))
        logtrans .= log.(trans)
    end

    # Fit observations
    for j in 1:state_count(hmm)
        fit!(
            distribution.(emissions(hmm), Ref(selectdim(emission_parameters(hmm), 1, j))),
            obs_sequences,
            view.(γs, j, :)
        )
    end


    # Safety check
    @argcheck HiddenMarkovModels.valid_hmm(hmm)

    return nothing
end


function StatsAPI.fit!(
    dists::AbstractVector,
    obs_sequences::AbstractVector{<:AbstractVector},
    γsⱼ::AbstractVector{<:AbstractVector}
)
    @argcheck length(dists) == length(obs_sequences) == length(γsⱼ) DimensionMismatch

    u_p = unique(parameter.(dists))
    @argcheck length(u_p) == 1 ArgumentError("All distributions must share the same parameter object")

    θ0 = only(u_p)
    ∇ = similar(θ0)
    ∇_tmp = similar(θ0)

    # Compute cost, gradient, hessian as sum over sequences of those of individual sequences
    baum_objs_grads = baum_value_gradient.(dists, obs_sequences, γsⱼ)
    
    function f(θ)
        -sum(baum_objs_grads[i][1](θ) for i in eachindex(baum_objs_grads))
    end

    function grad!(∇, θ)
        fill!(∇, zero(eltype(∇)))
        for i in eachindex(baum_objs_grads)
            baum_objs_grads[i][2](∇_tmp, θ)
            ∇ .-= ∇_tmp
        end
    end

    res = Optim.optimize(f, grad!, θ0, Optim.LBFGS())

    θ0[:] .= Optim.minimizer(res)

    return nothing
end
