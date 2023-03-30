module Qfi

import QuantumOptics
import QuantumOptics: 
       dagger, spre, spost, ptrace
import QuantumOptics.QuantumOpticsBase: 
       SuperOperator, AbstractOperator, Operator
import IterativeSolvers: bicgstabl
import ChainRulesCore:
       frule, rrule
import LinearAlgebra
import LinearAlgebra: I

export steadyState, diffSteadyState, sld, qfi

abstract type AbstractPhySys end
liouv(phy::AbstractPhySys)::Function = identity
dliouv(phy::AbstractPhySys)::Function = identity

dim_hilbert(sop::SuperOperator) = length(sop.basis_l[begin])

function _solve(sop::SuperOperator; 
    bop::AbstractOperator = Operator(
        sop.basis_l[begin], sop.basis_l[begin+1],
        similar(sop.data, dim_hilbert(sop)^2)),
    xin::AbstractArray=1e-6 * rand(eltype(sop.data), dim_hilbert(sop)^2), 
    is_herm::Bool=true, is_tp::Bool=false, trace=0)
    M = dim_hilbert(sop)
    b = reshape(bop.data, dim_hilbert(lθ)^2)
    dtype = eltype(sop.data)
    x0 = [is_tp ? trace*one(dtype) - sum(xin[begin+1:end-1]) : 1e-6*rand(dtype); xin]
    y = [b ; is_tp ? trace*one(dtype) : zero(dtype)]
    lm = [sop.data  similar(sop.data, M^2); 
          permutedims([(i ÷ M + 1) == i % M && is_tp ? one(dtype) : zero(dtype) for i in 1:M^2+1])]
    svec = bicgstabl(x0, lm, y)[1:end-1] # Super vector
    op = Operator(sop.basis_l[begin], sop.basis_l[begin+1], reshape(svec, (M, M)))
    is_herm ? (op + dagger(op))/2.0 : op
end

steadyState(lθ::SuperOperator) = _solve(lθ; is_tp=true, trace=1)

diffSteadyState(lθ::SuperOperator, dlθ::SuperOperator, ρθ::AbstractOperator) = 
    _solve(lθ; bop=-dlθ*ρθ, is_tp=true, trace=0)

sld(ρθ::AbstractOperator, dρθ::AbstractOperator) = _solve((spre(ρθ) + spost(ρθ))/2.0; bop=dρθ)

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
    ss_expr = !(indices === nothing) ? 
              :(ss = (θ -> ptrace(steadyState(liouv(θ)), $indices))) :
              :(ss = (θ -> steadyState(liouv(θ))))
    dss_expr = !(indices === nothing) ?
        :(dss = (θ -> ptrace(diffSteadyState($liouv(θ), $dliouv(θ), ss(θ)), $indices))) :
        :(dss = (θ -> diffSteadyState($liouv(θ), $dliouv(θ), ss(θ))))
    return quote
        $ss_expr
        $dss_expr
        frule((_, Δx), ::typeof(ss), x) = (ss(x), dss(x)*Δx)
        rrule(::typeof(ss), θ) = begin
            ss_pullback(Δy) = (NoTangent(),  dss(x) * Δy)
            return ss(θ), ss_pullback
        end
    end
end

end
