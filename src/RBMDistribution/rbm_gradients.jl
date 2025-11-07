"""
    log_P_v_given_h(rbm, Eᵥ, Wᵀv, h)

Log probability of the visible units given the hidden units of the RBM.

`Eᵥ` should be precomputed as `energy(rbm.visible, v)`.
`Wᵀv` should be precomputed as `rbm.w' * v`.
"""
function log_P_v_given_h(rbm::RestrictedBoltzmannMachines.RBM, Eᵥ::Union{AbstractArray, Real}, Wᵀv::AbstractArray, h::AbstractArray)
    interaction = my_mult(Wᵀv', h)
    inputs = RestrictedBoltzmannMachines.inputs_v_from_h(rbm, h)
    cumulant = -RestrictedBoltzmannMachines.cgf(rbm.visible, inputs)
    return interaction .- Eᵥ .- cumulant'
end

"""
    ∂ₕlog_P_v_given_h(rbm, Wᵀv, h)

Gradient of the log probability of the visible units given the hidden units of the RBM.

`Wᵀv` should be precomputed as `rbm.w' * v`.
"""
function ∂ₕlog_P_v_given_h(rbm::RestrictedBoltzmannMachines.RBM, Wᵀv::AbstractArray, h::AbstractVector)
    Wᵀv .- rbm.w' * RestrictedBoltzmannMachines.mean_v_from_h(rbm, h)
end

"""
    ∂ₕ²log_P_v_given_h(rbm, h)

Diagonal of the Hessian of the log probability of the visible units given the hidden units of the RBM.
"""
function ∂ₕ²log_P_v_given_h(rbm::RestrictedBoltzmannMachines.RBM, h::AbstractVector)
    mean_v = RestrictedBoltzmannMachines.mean_v_from_h(rbm, h)
    Diagonal(-((rbm.w .^ 2)' * (mean_v .* (1 .- mean_v))))
end
