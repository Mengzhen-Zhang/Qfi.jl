using QuantumOptics
using Optim
using Qfi

# ω_atom = 2
# ω_field = 1
# Ω = 1

function get_liouv_dliouv(ω_atom, ω_field, Ω)
    # Implement ptrace

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
    H_int = Ω * (a ⊗ sp + at ⊗ sm)

    b = b_fock ⊗ b_spin # Basis of composite system
    H = embed(b, 1, H_field) + embed(b, 2, H_atom) + H_int

    J = [embed(b, 1, a), embed(b, 2, sm)]
    liouv(θ) = liouvillian(H + θ * embed(b, 1, n),  J)
    dliouv(θ) = liouvillian(embed(b, 1, n), [])
    return (liouv, dliouv)
end

function fg!(F, G, x)
    # do common computations here
    liouv, dliouv = get_liouv_dliouv(x...)
    QFI = qfi(0.0, liouv, dliouv)
    fval = QFI["qfi"]
    Gval = QFI["dqfi"]
    if G !== nothing
        # writing the result to the vector G (gradient)
        G .= -Gval              # Minus sign for maximization
    end
    if F !== nothing
        # value = ... code to compute objective function
        return -fval            # Minus sign for maximization
    end
end

# Use Fminbox algorithm for box constraint optimization
# with LBFGS the inner optimzer
# The option outer_iterations controls the number of interations
# for Fminbox; the number of interations for LBFGS can
# be controlled by inner_iterations
res= Optim.optimize(Optim.only_fg!(fg!),
                    [1.8, 0.8, 0.8], # lower bound
                    [2.2, 1.2, 1.2], # upper bound
                    [2.0, 1.0, 1.0], # initial guess
                    Fminbox(LBFGS()), 
                    Optim.Options(outer_iterations = 2))

println(res)
# println(qfi(5.0, liouv, dliouv; indices=[2]))
# println(qfi(20.0, liouv, dliouv; indices=[1], n_sld=3))
