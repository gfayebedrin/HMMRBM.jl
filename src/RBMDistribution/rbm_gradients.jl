const AbstractRBM = Union{RestrictedBoltzmannMachines.RBM,RestrictedBoltzmannMachines.StandardizedRBM}


ΔE(::RestrictedBoltzmannMachines.RBM, ::Any) = 0.0
function ΔE(rbm::RestrictedBoltzmannMachines.StandardizedRBM, inputs)
    RestrictedBoltzmannMachines.energy(
        RestrictedBoltzmannMachines.Binary(; θ = rbm.offset_v),
        inputs
        )
end

"""
    log_P_v_given_h(rbm, v, h)

Log probability of the visible units given the hidden units of the RBM.
"""
function log_P_v_given_h(rbm::AbstractRBM, v::AbstractArray, h::AbstractArray)
    Eᵥ = RestrictedBoltzmannMachines.energy(rbm.visible, v)
    E_interaction = RestrictedBoltzmannMachines.interaction_energy(rbm, v, h)
    inputs = RestrictedBoltzmannMachines.inputs_v_from_h(rbm, h)
    F = -RestrictedBoltzmannMachines.cgf(rbm.visible, inputs)
    return F .- Eᵥ .- E_interaction .- ΔE(rbm, inputs)
end

"""
    weights(rbm)

Access the weight matrix of the RBM.
"""
weights(rbm::RestrictedBoltzmannMachines.RBM) = rbm.w
weights(rbm::RestrictedBoltzmannMachines.StandardizedRBM) = RestrictedBoltzmannMachines.unstandardized_weights(rbm)

"""
    ∂ₕlog_P_v_given_h(rbm, v, h)

Gradient of the log probability of the visible units given the hidden units of the RBM.
"""
function ∂ₕlog_P_v_given_h(rbm::AbstractRBM, v::AbstractArray, h::AbstractVector)
    weights(rbm)' * (v .- RestrictedBoltzmannMachines.mean_v_from_h(rbm, h))
end

"""
    ∂ₕ²log_P_v_given_h(rbm, v, h)

Diagonal of the Hessian of the log probability of the visible units given the hidden units of the RBM.
"""
function ∂ₕ²log_P_v_given_h(rbm::AbstractRBM, ::AbstractArray, h::AbstractVector)
    mean_v = RestrictedBoltzmannMachines.mean_v_from_h(rbm, h)
    Diagonal(-(weights(rbm) .^ 2)' * (mean_v .* (1 .- mean_v)))
end
