module Qfi

import QuantumOptics
import QuantumOptics: 
       dagger, spre, spost, ptrace, expect
import QuantumOptics.QuantumOpticsBase: 
       SuperOperator, AbstractOperator, Operator
import IterativeSolvers: bicgstabl!
import ChainRulesCore:
       frule, rrule
import LinearAlgebra
import LinearAlgebra: I, tr

export qfi

hs_norm(A) = tr(A*dagger(A))
dim_hilbert(sop::SuperOperator) = length(sop.basis_l[begin])

function _solve(
    sop::SuperOperator; 
    bop::AbstractOperator = Operator(
        sop.basis_l[begin], sop.basis_l[begin+1],
        similar(sop.data, dim_hilbert(sop), dim_hilbert(sop))),
    xin::AbstractArray=1e-6 * rand(eltype(sop.data), dim_hilbert(sop)^2), 
    is_herm::Bool=true,
    is_tp::Bool=false,
    trace=0,
    max_mv_products=10000)
    M = dim_hilbert(sop)
    b = reshape(bop.data, dim_hilbert(sop)^2)
    dtype = eltype(sop.data)
    x0 = [is_tp ? trace*one(dtype) - sum(xin[begin+1:end-1]) : 1e-6*rand(dtype); xin]
    y = [b ; is_tp ? trace*one(dtype) : zero(dtype)]
    lm = [sop.data  similar(sop.data, M^2); 
          permutedims([(i ÷ M + 1) == i % M && is_tp ? one(dtype) : zero(dtype) for i in 1:M^2+1])]
    svec = bicgstabl!(x0, lm, y; max_mv_products=max_mv_products)[1:end-1]  # Super vector
    op = Operator(sop.basis_l[begin], sop.basis_l[begin+1], reshape(svec, (M, M)))
    is_herm ? (op + dagger(op))/2.0 : op
end

steadyState(lθ::SuperOperator) = _solve(lθ; is_tp=true, trace=1)

diffSteadyState(lθ::SuperOperator, dlθ::SuperOperator, ρθ::AbstractOperator) = 
    _solve(lθ; bop=-dlθ*ρθ, is_tp=true, trace=0)

diff2SteadyState(lθ::SuperOperator, dlθ::SuperOperator, d2lθ::SuperOperator,
                 ρθ::AbstractOperator, dρθ::AbstractOperator) =
                     _solve(lθ; bop=-(d2lθ*ρθ + 2*dlθ*dρθ), is_tp=true, trace=0)

sld(ρθ::AbstractOperator, dρθ::AbstractOperator) = _solve((spre(ρθ) + spost(ρθ))/2.0; bop=dρθ)

diffSld(ρθ::AbstractOperator, dρθ::AbstractOperator, d2ρθ::AbstractOperator, SLD::AbstractOperator) =
    _solve((spre(ρθ) + spost(ρθ))/2.0;
           bop=d2ρθ-(dρθ*SLD + SLD*dρθ)/2.0)

qfi(ρθ::AbstractOperator, SLD::AbstractOperator) = real(tr(ρθ * SLD * SLD))

dqfi(ρθ::AbstractOperator, dρθ::AbstractOperator, SLD::AbstractOperator, dSLD::AbstractOperator) =
    real(tr(dρθ*SLD*SLD + ρθ*(SLD*dSLD+dSLD*SLD)))

"""
    qfi(θ::Real, liouv::Function, dliouv::Function, d2liouv::Funciton; indices=nothing)
    qfi(θ::Real, liouv::Function, dliouv::Function; indices=nothing)

Calculate quantun fisher information given SuperOperator-valued functions `liouv` and `dliouv`.
`liouv` defines the Liouvillian, `dliouv` defines its derivative, `d2lioub` defines its second.
`indices` is the same as in `ptrace`. Without specificaiton, d2liouv is the zero operator.

The output is a dictionary with the following keys:

'qfi': the value of quantum fisher information
'dqfi': the value of the derivative of the quantum fisher information
'prec_ss': the precision of the steady state
'prec_dss': the precision of the derivative of the steady state
'prec_sld': the precision of the symmetric logarithmic derivative (SLD)
'prec-dsld': the precision of the derivative of the SLD
"""
function qfi(θ::Real, liouv::Function, dliouv::Function, d2liouv::Function;
             indices=nothing,
             ops=[])
    lθ = liouv(θ)
    dlθ = dliouv(θ)
    d2lθ = d2liouv(θ)

    ρθ = steadyState(lθ)
    dρθ = diffSteadyState(lθ, dlθ, ρθ)
    # d2ρθ = diff2SteadyState(lθ, dlθ, d2lθ, ρθ, dρθ)

    prec_ρθ = hs_norm(lθ*ρθ)
    prec_dρθ = hs_norm(dlθ*ρθ+lθ*dρθ)

    if !(indices === nothing)
        ρθ = ptrace(ρθ, indices)
        dρθ = ptrace(dρθ, indices)
        # d2ρθ = ptrace(d2ρθ, indices)
    end

    SLD = sld(ρθ, dρθ)
    # dSLD = diffSld(ρθ, dρθ, d2ρθ, SLD)

    QFI = qfi(ρθ, SLD)
    # dQFI = dqfi(ρθ, dρθ, SLD, dSLD)

    prec_SLD = hs_norm((spre(ρθ)*SLD + spost(ρθ)*SLD)/2.0-dρθ)
    # prec_dSLD = hs_norm((spre(ρθ)*dSLD + spost(ρθ)*dSLD)/2.0
    #                     + (spre(dρθ)*SLD + spost(dρθ)*SLD)/2.0 - d2ρθ)
 
    return Dict{Any, Any}(
        "qfi" => QFI,
        # "dqfi" => dQFI,
        "prec_ss" => real(prec_ρθ),
        "prec_dss" => real(prec_dρθ),
        "prec_sld" => real(prec_SLD),
        "ops" => [expect(op, ρθ) for op in ops]
        # "prec_dsld" => prec_dSLD
    )
end

qfi(θ::Real, liouv::Function, dliouv::Function; indices=nothing, ops=[])=
    qfi(θ, liouv, dliouv, θ->(0*dliouv(θ)); indices=indices, ops=ops)

end
