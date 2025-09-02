# quantum phase estimation 
# ref: https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/quantum-phase-estimation.ipynb
# 
# fourier 역변환을 아직 제대로 하지 못했다. 


#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
# from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

num_qubit = 5 
qpe = QuantumCircuit(num_qubit+1, num_qubit)
qpe.x(num_qubit)
# print(qpe.draw())

for qubit in range(num_qubit):
    qpe.h(qubit)
# print(qpe.draw())

angle = 2*math.pi/3
repetitions = 1
for counting_qubit in range(num_qubit):
    for i in range(repetitions):
        qpe.cp(angle, counting_qubit, num_qubit); # controlled-T
    repetitions *= 2
# print(qpe.draw())

qpe.barrier()
# Apply inverse QFT
# qpe = qpe.compose(QFT(3, inverse=True), [0,1,2])
# qpe.inverse_qft(3,approximation_degree=0, do_swaps=False)
qpe = qpe.compose(QFT(num_qubits=num_qubit, approximation_degree=0, do_swaps=False, 
           inverse=True, insert_barriers=True, name='iqft'))
# Measure
qpe.barrier()
for n in range(num_qubit):
    qpe.measure(n,n)

# print(qpe.draw())

# back end: version1.3
from qiskit_aer import AerSimulator 

aer_sim = AerSimulator()
shots = 4096
results = aer_sim.run(qpe, shots=shots).result()
answer = results.get_counts()
print(answer)
plot_histogram(answer)