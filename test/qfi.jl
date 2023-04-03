using QuantumOptics
using Qfi
# Implement ptrace
ω_atom = 2
ω_field = 1

# 2 level atom described as spin
b_spin = SpinBasis(1//2)
sp = sigmap(b_spin)
sm = sigmam(b_spin)

H_atom = ω_atom*sp*sm

# Use a Fock basis with a maximum of 20 photons to model a cavity mode
b_fock = FockBasis(20)
a = destroy(b_fock)
at = create(b_fock)
n = number(b_fock)

H_field = ω_field*n + a + at

Ω = 1
H_int = Ω * (a ⊗ sp + at ⊗ sm)

# I_field = identityoperator(b_fock)
# I_atom = identityoperator(b_spin)

# H_atom_ = I_field ⊗ H_atom
# H_field_ = I_atom ⊗ H_field

b = b_fock ⊗ b_spin # Basis of composite system
H = embed(b, 1, H_field) + embed(b, 2, H_atom) + H_int

J = [embed(b, 1, a), embed(b, 2, sm)]

liouv(θ) = liouvillian(H + θ * embed(b, 1, n),  J)
dliouv(θ) = liouvillian(embed(b, 1, n), [])

println(qfi(0.0, liouv, dliouv))
# println(qfi(5.0, liouv, dliouv; indices=[2]))
# println(qfi(20.0, liouv, dliouv; indices=[1], n_sld=3))
