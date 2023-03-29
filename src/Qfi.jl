module Qfi

import QuantumOptics
import QuantumOptics: 
       dagger, spre, spost, ptrace
import QuantumOptics.QuantumOpticsBase: 
       SuperOperator, AbstractOperator
import IterativeSolvers: bicgstabl!
import ChainRulesCore:
       frule, rrule
import LinearAlgebra
import LinearAlgebra: I

export steadyState, diffSteadyState, sld, qfi

abstract type AbstractPhySys end
liouv(phy::AbstractPhySys)::Function = identity
dliouv(phy::AbstractPhySys)::Function = identity

function steadyState(lθ::SuperOperator)
    M = length(lθ.basis_l[1])
    x0 = similar(lθ.data, M^2+1)
    x0[begin] = one(eltype(lθ.data))
    x0[end] = zero(eltype(lθ.data))
    
    y = similar(lθ.data, M^2+1)
    y[end] = one(eltype(lθ.data))

    lm = [lθ.data  similar(lθ.data, M^2); permutedims([(i ÷ M + 1) == i % M ? one(eltype(lθ.data)) : zero(eltype(lθ.data)) for i in 1:M^2+1])]

    ρθ = bicgstabl!(x0, lm, y)[1:end-1] # Super vector
    ρθ = Operator(lθ.basis_l[1], lθ.basis_l[2], reshape(ρθ, (M, M)))
    return (ρθ + dagger(ρθ))/2.0
end

function diffSteadyState(lθ::SuperOperator, dlθ::SuperOperator, ρθ::AbstractOperator)
    M = length(dlθ.basis_l[1])
    x0 = 1e-6 * rand(eltype(ρθ.data), M^2 + 1)

    y = similar(dlθ.data, M^2+1)
    y[1:end-1].= reshape((- dlθ * ρθ).data, M^2)
    y[end] = zero(eltype(dlθ.data))

    lm = [lθ.data  similar(lθ.data, M^2); permutedims([(i ÷ M + 1) == i % M ? one(eltype(lθ.data)) : zero(eltype(lθ.data)) for i in 1:M^2+1])]

    dρθ = bicgstabl!(x0, lm, y)[1:end-1] # Super vector
    dρθ = Operator(lθ.basis_l[1], lθ.basis_l[2], reshape(dρθ, (M, M)))
    return (dρθ + dagger(dρθ)) / 2.0
end

function sld(ρθ::AbstractOperator, dρθ::AbstractOperator)
    M = length(ρθ.basis_l)
    x0 = 1e-6 * rand(eltype(ρθ.data), M^2)

    y = reshape(dρθ.data, M^2)
    
    lm = (spre(ρθ) + spost(ρθ)).data / 2.0
 
    sld = Operator(ρθ.basis_l, ρθ.basis_r, reshape(bicgstabl!(x0, lm * lm, lm * y), (M, M))) # Matrix

    return (sld + dagger(sld)) / 2.0
end

"""
    qfi(θ::Real, liouv::Function, dliouv::Function; indices=nothing, n_sld::Integer=2)

Calculate quantun fisher information given SuperOperator-valued functions `liouv` and `dliouv`.
`liouv` defines the Liouvillian and `dliouv` defines its derivative. `indices` is the same as
that in `ptrace`. `n_sld` is used to stableize the calculation of the symmetric logarithmic
derivative
"""
function qfi(θ::Real, liouv::Function, dliouv::Function; indices=nothing)
    lθ = liouv(θ)                                         # SuperOperator
    dlθ = dliouv(θ)                                       # SuperOperator

    ρθ = steadyState(lθ)
    
    dρθ = diffSteadyState(lθ, dlθ, ρθ)

    if !(indices === nothing)
        ρθ = ptrace(ρθ, indices)
        dρθ = ptrace(dρθ, indices)
    end

    sld = sld(ρθ, dρθ)

    return real(tr(ρθ * sld * sld))
end

macro define(liouv, dliouv, indices)
    if !(indices === nothing)
        return quote
            ss = (θ -> ptrace(steadyState(liouv(θ)), $indices))
            dss = (θ -> ptrace(diffSteadyState($liouv(θ), $dliouv(θ), ss(θ)), $indices))
            frule((_, Δx), ::typeof(ss), x) = begin
                ss(x), dss(x)*Δx
            end
            rrule(::typeof(ss), θ) = begin
                ss_pullback(Δy) = (NoTangent(),  dss(x) * Δy)
                return ss(θ), ss_pullback
            end
        end
    else
        return quote
            ss = (θ -> steadyState(liouv(θ)))
            dss = (θ -> diffSteadyState($liouv(θ), $dliouv(θ), ss(θ)))
            frule((_, Δx), ::typeof(ss), x) = begin
                ss(x), dss(x)*Δx
            end
            rrule(::typeof(ss), θ) = begin
                ss_pullback(Δy) = (NoTangent(),  dss(x) * Δy)
                return ss(θ), ss_pullback
            end
        end
    end    
end

end
