# MultiSeqHMM.jl

# --- MultiSeqHMM and SingleSeqHMM definitions ---

struct SingleSeqHMM{T,I<:Integer} <: AbstractHMM
    parent::T
    sequence_index::I
end

"""
Type `F` is the emission distribution family type and must implement `distribution(::F, parameter)`.

The returned distribution type must implement:
- `Random.rand(::AbstractRNG, dist::Distribution)`
- `DensityInterface.logdensityof(dist::Distribution, obs)`
"""
struct MultiSeqHMM{T,F,AV,AM,Aθ} <: AbstractVector{SingleSeqHMM{MultiSeqHMM{T,F,AV,AM,Aθ},Int}}
    inits::Vector{AV}
    loginits::Vector{AV}
    transitions::Vector{AM}
    logtransitions::Vector{AM}
    emissions::Vector{F}
    θ::Aθ
    hyperparameters::NamedTuple


    function MultiSeqHMM(
        inits::Vector{AV},
        transitions::Vector{AM},
        emissions::Vector{F},
        θ::Aθ,
        hyperparameters::NamedTuple=(;)
    ) where {T<:Real,F<:DistributionFamily,AV<:AbstractVector{T},AM<:AbstractMatrix{T},Aθ<:AbstractArray{T}}

        # --- Basic length checks ---
        number_of_submodels = length(inits)
        @argcheck number_of_submodels == length(transitions) "Number of inits and transitions must match"
        @argcheck number_of_submodels == length(emissions) "Number of inits, transitions, and emissions must match"

        # --- Check transition shapes and init sizes ---
        Ns = Int[]
        for (s, (π, A_mat)) in enumerate(zip(inits, transitions))
            nA, mA = size(A_mat)
            @argcheck nA == mA "Transition matrix $s must be square, got $nA×$mA"
            @argcheck length(π) == nA "Init vector $s length $(length(π)) ≠ transition size $nA"
            push!(Ns, nA)
        end

        # --- Consistent state count ---
        N = Ns[1]
        @argcheck all(Ns .== N) "Sequences have inconsistent number of states: $(Ns)"
        @argcheck size(θ, 1) == N "θ first dimension $(size(θ,1)) must match number of states $N"

        # --- Backend consistency checks ---
        @argcheck eltype(first(inits)) == eltype(first(transitions)) == eltype(θ) "Element types of inits, transitions, θ must match"
        @argcheck typeof(first(inits)) <: AbstractArray && typeof(first(transitions)) <: AbstractArray "Init and transition arrays must be concrete AbstractArrays"

        # --- Log versions ---
        loginits = [log.(π) for π in inits]
        logtransitions = [log.(trans) for trans in transitions]

        return new{T,F,AV,AM,Aθ}(inits, loginits, transitions, logtransitions, emissions, θ, hyperparameters)
    end
end


# --- MultiSeqHMM methods ---

# Accessors
inits(hmm::MultiSeqHMM) = hmm.inits
loginits(hmm::MultiSeqHMM) = hmm.loginits
transitions(hmm::MultiSeqHMM) = hmm.transitions
logtransitions(hmm::MultiSeqHMM) = hmm.logtransitions
emissions(hmm::MultiSeqHMM) = hmm.emissions
emission_parameters(hmm::MultiSeqHMM) = hmm.θ
hyperparameters(hmm::MultiSeqHMM) = hmm.hyperparameters

state_count(hmm::MultiSeqHMM) = size(emission_parameters(hmm), 1)

# Interface
Base.size(hmm::MultiSeqHMM) = (length(inits(hmm)),)
Base.getindex(hmm::MultiSeqHMM, s::Integer) = SingleSeqHMM(hmm, Int(s))

function Adapt.adapt_structure(to, m::MultiSeqHMM)
    MultiSeqHMM(
        [adapt(to, x) for x in m.inits],
        [adapt(to, x) for x in m.transitions],
        [adapt(to, x) for x in m.emissions],
        adapt(to, m.θ)
    )
end

function DensityInterface.logdensityof(hmm::MultiSeqHMM)
    sum(logdensityof.(hmm))
end

function HiddenMarkovModels.valid_hmm(hmm::MultiSeqHMM)
    all(HiddenMarkovModels.valid_hmm, hmm)
end


# --- SingleSeqHMM methods ---

# Constructors
function SingleSeqHMM(
    parent::MultiSeqHMM,
    sequence_index::I
) where {I<:Integer}

    S = length(parent)
    @argcheck 1 <= sequence_index <= S "sequence_index $sequence_index out of bounds (1 ≤ index ≤ $S)"

    return SingleSeqHMM{typeof(parent),I}(parent, sequence_index)
end

# Accessors
parent(hmm::SingleSeqHMM) = hmm.parent
sequence_index(hmm::SingleSeqHMM) = hmm.sequence_index
hyperparameters(hmm::SingleSeqHMM) = hyperparameters(parent(hmm))

state_count(hmm::SingleSeqHMM) = state_count(parent(hmm))

# Interface
function HiddenMarkovModels.initialization(hmm::SingleSeqHMM)
    return inits(parent(hmm))[sequence_index(hmm)]
end

function HiddenMarkovModels.transition_matrix(hmm::SingleSeqHMM)
    return transitions(parent(hmm))[sequence_index(hmm)]
end

function HiddenMarkovModels.obs_distributions(hmm::SingleSeqHMM)
    θ = emission_parameters(parent(hmm))
    return [
        distribution(
            emissions(parent(hmm))[sequence_index(hmm)],
            selectdim(θ, 1, s)
        )
        for s in 1:size(θ, 1)
    ]
end
