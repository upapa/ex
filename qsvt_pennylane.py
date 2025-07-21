# ref: https://pennylane.ai/qml/demos/tutorial_apply_qsvt

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

A = np.array(
    [
        [0.65713691, -0.05349524, 0.08024556, -0.07242864],
        [-0.05349524, 0.65713691, -0.07242864, 0.08024556],
        [0.08024556, -0.07242864, 0.65713691, -0.05349524],
        [-0.07242864, 0.08024556, -0.05349524, 0.65713691],
    ]
)

b = np.array([1, 2, 3, 4], dtype="complex")
target_x = np.linalg.inv(A) @ b  # true solution

kappa = 4
s = 0.10145775
phi_pyqsp = [-2.287, 2.776, -1.163, 0.408, -0.16, -0.387, 0.385, -0.726, 0.456, 0.062, -0.468, 0.393, 0.028, -0.567, 0.76, -0.432, -0.011, 0.323, -0.573, 0.82, -1.096, 1.407, -1.735, 2.046, -2.321, 2.569, -2.819, -0.011, 2.71, -2.382, 2.574, 0.028, -2.749, 2.673, 0.062, -2.685, 2.416, 0.385, -0.387, -0.16, 0.408, -1.163, -0.365, 2.426]
phi_qsvt = qml.transform_angles(phi_pyqsp, "QSP", "QSVT")  # convert pyqsp angles to be compatible with QSVT

x_vals = np.linspace(0, 1, 50)
target_y_vals = [s * (1 / x) for x in np.linspace(s, 1, 50)]

qsvt_y_vals = []
for x in x_vals:

    block_encoding = qml.BlockEncode(x, wires=[0])
    projectors = [qml.PCPhase(angle, dim=1, wires=[0]) for angle in phi_qsvt]

    poly_x = qml.matrix(qml.QSVT, wire_order=[0])(block_encoding, projectors)
    qsvt_y_vals.append(np.real(poly_x[0, 0]))
    

def sum_even_odd_circ(x, phi, ancilla_wire, wires):
    phi1, phi2 = phi[: len(phi) // 2], phi[len(phi) // 2 :]
    block_encode = qml.BlockEncode(x, wires=wires)

    qml.Hadamard(wires=ancilla_wire)  # equal superposition

    # apply even and odd polynomial approx

    dim = x.shape[0] if x.ndim > 0 else 1
    projectors_even = [qml.PCPhase(angle, dim= dim, wires=wires) for angle in phi1]
    qml.ctrl(qml.QSVT, control=(ancilla_wire,), control_values=(0,))(block_encode, projectors_even)

    projectors_odd = [qml.PCPhase(angle, dim= dim, wires=wires) for angle in phi2]
    qml.ctrl(qml.QSVT, control=(ancilla_wire,), control_values=(0,))(block_encode, projectors_odd)

    qml.Hadamard(wires=ancilla_wire)  # un-prepare superposition
    

np.random.seed(42)  # set seed for reproducibility
phi = np.random.rand(51)

def real_u(A, phi):
    qml.Hadamard(wires="ancilla1")

    qml.ctrl(sum_even_odd_circ, control=("ancilla1",), control_values=(0,))(A, phi, "ancilla2", [0, 1, 2])
    qml.ctrl(qml.adjoint(sum_even_odd_circ), control=("ancilla1",), control_values=(1,))(A.T, phi, "ancilla2", [0, 1, 2])

    qml.Hadamard(wires="ancilla1")

# Normalize states:
norm_b = np.linalg.norm(b)
normalized_b = b / norm_b

norm_x = np.linalg.norm(target_x)
normalized_x = target_x / norm_x

@qml.qnode(qml.device("default.qubit", wires=["ancilla1", "ancilla2", 0, 1, 2]))
def linear_system_solver_circuit(phi):
    qml.StatePrep(normalized_b, wires=[1, 2])
    real_u(A.T, phi)  # invert the singular values of A transpose to get A^-1
    return qml.state()


transformed_state = linear_system_solver_circuit(phi)[:4]  # first 4 entries of the state
rescaled_computed_x = transformed_state * norm_b / s
normalized_computed_x = rescaled_computed_x / np.linalg.norm(rescaled_computed_x)

print("target x:", np.round(normalized_x, 3))
print("computed x:", np.round(normalized_computed_x, 3))