using QuantumOptics
include("../src/Qfi.jl")
using .Qfi

ω_atom = 0.0
ω_field = 0.0
Ω = 1.0

b_spin = SpinBasis(1//2)
sp = sigmap(b_spin)
sm = sigmam(b_spin)

H_atom = ω_atom*sp*sm

b_fock = FockBasis(20)
a = destroy(b_fock)
at = create(b_fock)
n = number(b_fock)

H_field = ω_field*n + a + at
H_int = Ω * (a ⊗ sp + at ⊗ sm)

b = b_fock ⊗ b_spin # Basis of composite system
H = embed(b, 1, H_field) + embed(b, 2, H_atom) + H_int

J = [embed(b, 1, a), embed(b, 2, sm)]
liouv(θ) = liouvillian(H + θ * embed(b, 1, n),  J)
dliouv(θ) = liouvillian(embed(b, 1, n), [])

println(qfi(0.0, liouv, dliouv; ops=[embed(b, 1, n)]))

