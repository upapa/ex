import numpy as np
from qiskit import Aer, execute
from qiskit.algorithms.linear_solvers import HHL
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance
from qiskit.algorithms.linear_solvers.observables import AbsoluteAverage

# 1차원 열방정식: -d²u/dx² = f(x)
# 격자점 3개, Dirichlet 경계조건 u(0)=u(1)=0
# 이산화하면: [2 -1  0][u1]   [f1]
#             [-1 2 -1][u2] = [f2]
#             [0 -1 2][u3]   [f3]

A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])
f = np.array([1, 0, 0])  # 예시: f(x) = 1 at x1, 0 elsewhere

# Qiskit의 HHL 알고리즘은 행렬 크기가 2^n이어야 하므로 4x4로 패딩
A_padded = np.zeros((4, 4))
A_padded[:3, :3] = A
f_padded = np.zeros(4)
f_padded[:3] = f

# 양자 인스턴스 설정
backend = Aer.get_backend('statevector_simulator')
qi = QuantumInstance(backend)

# HHL 알고리즘 실행
hhl = HHL(observable=AbsoluteAverage())
result = hhl.solve(A_padded, f_padded, quantum_instance=qi)

# 결과 출력
print("HHL Solution (quantum):", np.round(result.state.solution, 4))

# 비교: 고전적 해
u_classical = np.linalg.solve(A, f)
print("Classical Solution:", np.round(u_classical, 4))