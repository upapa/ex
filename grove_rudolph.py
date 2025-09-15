import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Initialize
from qiskit.visualization import plot_bloch_multivector, plot_state_city


# 정규 분포 파라미터
mu = 0       # 평균
sigma = 1    # 표준편차
num_qubits = 5
dim = 2 ** num_qubits

# 정규 분포 확률 계산
x = np.linspace(-2, 2, dim)
pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
pdf /= np.sum(pdf)  # 정규화

# 진폭은 확률의 제곱근
amplitudes = np.sqrt(pdf)

# 진폭을 이용한 상태 초기화
init_gate = Initialize(amplitudes)
qc = QuantumCircuit(num_qubits)
qc.append(init_gate, range(num_qubits))
qc.barrier()
qc.measure_all()

# # 시뮬레이션
from qiskit_aer import AerSimulator
aer_sim = AerSimulator()
shots = 4096
results = aer_sim.run(qc.decompose(reps=10), shots=shots).result()
counts = results.get_counts()

# 결과 시각화
from qiskit.visualization import plot_histogram
plot_histogram(counts).show()


# from msvcrt import getch
# getch()