import pennylane as qml
from pennylane import numpy as np

dev = qml.device("lightning.qubit", wires=[0,1])

A = np.array([[0.1, 0.2], [0.3, 0.4]])
phase_angles = np.array([0.0, 1.0, 2.0, 3.0])

block_encoding = qml.BlockEncode(A, wires=[0,1])
projectors = [qml.PCPhase(angle, dim=2, wires=[0,1]) for angle in phase_angles]

@qml.qnode(dev)
def my_circuit():
    qml.QSVT(block_encoding, projectors)
    return qml.state()

my_circuit()
print(qml.draw(my_circuit, level="top")())

print(qml.draw(my_circuit)())

import pyqsp
kappa=4
pcoefs, s = pyqsp.poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True,return_scale=True)
phi_pyqsp = pyqsp.angle_sequence.QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=0.00001)