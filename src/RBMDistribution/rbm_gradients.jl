"""
    log_P_v_given_h(rbm, v, h)
    log_P_v_given_h(rbm, Eᵥ, Wᵀv, h)

Log probability of the visible units given the hidden units of the RBM.

## Optimisation
Use `log_P_v_given_h(rbm, Eᵥ, Wᵀv, h)` when repeatedly evaluating for the same visible vector `v`.
- `Eᵥ` should be precomputed as `energy(rbm.visible, v)`.
- `Wᵀv` should be precomputed as `rbm.w' * v`.
"""
function log_P_v_given_h(rbm::RestrictedBoltzmannMachines.RBM, Eᵥ::Union{AbstractArray, Real}, Wᵀv::AbstractArray, h::AbstractArray)
    interaction = my_mult(Wᵀv', h)
    inputs = RestrictedBoltzmannMachines.inputs_v_from_h(rbm, h)
    cumulant = RestrictedBoltzmannMachines.cgf(rbm.visible, inputs)
    return interaction .- Eᵥ .- cumulant'
end

function log_P_v_given_h(rbm::RestrictedBoltzmannMachines.RBM, v::AbstractArray, h::AbstractArray)
    Eᵥ = RestrictedBoltzmannMachines.energy(rbm.visible, v)
    Wᵀv = rbm.w' * v
    log_P_v_given_h(rbm, Eᵥ, Wᵀv, h)
end


"""
    log_P_v_given_h!(out, rbm, Eᵥ, Wᵀv, h)
"""
function log_P_v_given_h!(out::AbstractArray, rbm::RestrictedBoltzmannMachines.RBM, Eᵥ::Union{AbstractArray, Real}, Wᵀv::AbstractArray, h::AbstractArray)
    interaction = my_mult(Wᵀv', h)
    inputs = RestrictedBoltzmannMachines.inputs_v_from_h(rbm, h)
    cumulant = RestrictedBoltzmannMachines.cgf(rbm.visible, inputs)
    @. out = interaction - Eᵥ - cumulant'
    return out
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
    ∂ₕlog_P_v_given_h!(out, rbm, Wᵀv, h)
"""
function ∂ₕlog_P_v_given_h!(out::AbstractArray, rbm::RestrictedBoltzmannMachines.RBM, Wᵀv::AbstractArray, h::AbstractVector)
    tmp = rbm.w' * RestrictedBoltzmannMachines.mean_v_from_h(rbm, h)
    @. out = Wᵀv - tmp
    return out
end


"""
    ∂ₕ²log_P_v_given_h(rbm, h)

Diagonal of the Hessian of the log probability of the visible units given the hidden units of the RBM.
"""
function ∂ₕ²log_P_v_given_h(rbm::RestrictedBoltzmannMachines.RBM, h::AbstractVector)
    mean_v = RestrictedBoltzmannMachines.mean_v_from_h(rbm, h)
    Diagonal(-((rbm.w .^ 2)' * (mean_v .* (1 .- mean_v))))
end
