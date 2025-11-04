const AbstractRBM = Union{RestrictedBoltzmannMachines.RBM,RestrictedBoltzmannMachines.StandardizedRBM}

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
