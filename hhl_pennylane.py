import jax
import qrisp
import numpy as np

def QPE(psi, U, precision=None, res=None):

    if res is None:
        res = qrisp.QuantumFloat(precision, -precision)

    qrisp.h(res)

    # Performs a loop with a dynamic bound in Jasp mode.
    for i in qrisp.jrange(res.size):
        with qrisp.control(res[i]):
            for j in qrisp.jrange(2**i):
                U(psi)

    return qrisp.QFT(res, inv=True)



def U(psi):
    phi_1 = 0.5
    phi_2 = 0.125

    qrisp.p(phi_1 * 2 * np.pi, psi[0])  # p applies a phase gate
    qrisp.p(phi_2 * 2 * np.pi, psi[1])

def fake_inversion(qf, res=None):

    if res is None:
        res = qrisp.QuantumFloat(qf.size + 1)

    for i in qrisp.jrange(qf.size):
        qrisp.cx(qf[i], res[qf.size - i])

    return res

@qrisp.RUS(static_argnums=[0, 1])
def HHL_encoding(b, hamiltonian_evolution, n, precision):

    # Prepare the state |b>. Step 1
    qf = qrisp.QuantumFloat(n)
    # Reverse the endianness for compatibility with Hamiltonian simulation.
    qrisp.prepare(qf, b, reversed=True)

    qpe_res = QPE(qf, hamiltonian_evolution, precision=precision)  # Step 2
    inv_res = fake_inversion(qpe_res)  # Step 3

    case_indicator = qrisp.QuantumFloat(inv_res.size)

    with qrisp.conjugate(qrisp.h)(case_indicator):
        qbl = (case_indicator >= inv_res)

    cancellation_bool = (qrisp.measure(case_indicator) == 0) & (qrisp.measure(qbl) == 0)

    # The first return value is a boolean value. Additional return values are QuantumVariables.
    return cancellation_bool, qf, qpe_res, inv_res


def HHL(b, hamiltonian_evolution, n, precision):
    qf, qpe_res, inv_res = HHL_encoding(b, hamiltonian_evolution, n, precision)
    # Uncompute qpe_res and inv_res
    with qrisp.invert():
        QPE(qf, hamiltonian_evolution, res=qpe_res)
        fake_inversion(qpe_res, res=inv_res)
    # Reverse the endianness for compatibility with Hamiltonian simulation.
    for i in qrisp.jrange(qf.size // 2):
        qrisp.swap(qf[i], qf[n - i - 1])
    return qf

from qrisp.operators import QubitOperator

def hermitian_matrix_with_power_of_2_eigenvalues(n):
    # Generate eigenvalues as inverse powers of 2.
    eigenvalues = 1 / np.exp2(np.random.randint(1, 4, size=n))

    # Generate a random unitary matrix.
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # Construct the Hermitian matrix.
    A = Q @ np.diag(eigenvalues) @ Q.conj().T

    return A


# Example
n = 3
A = hermitian_matrix_with_power_of_2_eigenvalues(2**n)

H = QubitOperator.from_matrix(A).to_pauli()


def U(qf):
    H.trotterization()(qf, t=-np.pi, steps=5)


b = np.random.randint(0, 2, size=2**n)

print("Hermitian matrix A:")
print(A)

print("Eigenvalues:")
print(np.linalg.eigvals(A))

print("b:")
print(b)

@qrisp.terminal_sampling
def main():
    x = HHL(tuple(b), U, n, 4)
    return x


res_dict = main()

for k, v in res_dict.items():
    res_dict[k] = v**0.5

np.array([res_dict[key] for key in sorted(res_dict)])

