# ref: https://qiskit-community.github.io/qiskit-finance/tutorials/00_amplitude_estimation.html

p = 0.2

import numpy as np
from qiskit.circuit import QuantumCircuit


class BernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(theta_p, 0)


class BernoulliQ(QuantumCircuit):
    """A circuit representing the Bernoulli Q operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        self._theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self._theta_p, 0)

    def power(self, k):
        # implement the efficient power of Q
        q_k = QuantumCircuit(1)
        q_k.ry(2 * k * self._theta_p, 0)
        return q_k
    
A = BernoulliA(p)
Q = BernoulliQ(p)  

from qiskit_algorithms import EstimationProblem

problem = EstimationProblem(
    state_preparation=A,  # A operator
    grover_operator=Q,  # Q operator
    objective_qubits=[0],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
)

from qiskit.primitives import Sampler
sampler = Sampler()
# from qiskit.primitives import StatevectorSampler
# sampler = StatevectorSampler()

# #### 1 
from qiskit_algorithms import AmplitudeEstimation
ae = AmplitudeEstimation(
    num_eval_qubits=3,  # the number of evaluation qubits specifies circuit width and accuracy
    sampler=sampler,
)
ae_result = ae.estimate(problem)
print(ae_result.estimation)
print("Interpolated MLE estimator:", ae_result.mle)


# import matplotlib.pyplot as plt

# # plot estimated values
# gridpoints = list(ae_result.samples.keys())
# probabilities = list(ae_result.samples.values())

# plt.bar(gridpoints, probabilities, width=0.5 / len(probabilities))
# plt.axvline(p, color="r", ls="--")
# plt.xticks(size=15)
# plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
# plt.title("Estimated Values", size=15)
# plt.ylabel("Probability", size=15)
# plt.xlabel(r"Amplitude $a$", size=15)
# plt.ylim((0, 1))
# plt.grid()
# plt.show()


# ae_circuit = ae.construct_circuit(problem)
# ae_circuit.decompose().draw(
#     "mpl", style="clifford"
# )  # decompose 1 level: exposes the Phase estimation circuit!

# from qiskit import transpile
# basis_gates = ["h", "ry", "cry", "cx", "ccx", "p", "cp", "x", "s", "sdg", "y", "t", "cz"]
# transpile(ae_circuit, basis_gates=basis_gates, optimization_level=2).draw("mpl", style="clifford")


from qiskit_algorithms import IterativeAmplitudeEstimation

iae = IterativeAmplitudeEstimation(
    epsilon_target=0.01,  # target accuracy
    alpha=0.05,  # width of the confidence interval
    sampler=sampler,
)
iae_result = iae.estimate(problem)

print("Estimate:", iae_result.estimation)

# iae_circuit = iae.construct_circuit(problem, k=3)
# iae_circuit.draw("mpl", style="clifford")

from qiskit_algorithms import MaximumLikelihoodAmplitudeEstimation

mlae = MaximumLikelihoodAmplitudeEstimation(
    evaluation_schedule=3,  # log2 of the maximal Grover power
    sampler=sampler,
)
mlae_result = mlae.estimate(problem)

print("Estimate:", mlae_result.estimation)

from qiskit_algorithms import FasterAmplitudeEstimation

fae = FasterAmplitudeEstimation(
    delta=0.01,  # target accuracy
    maxiter=3,  # determines the maximal power of the Grover operator
    sampler=sampler,
)
fae_result = fae.estimate(problem)

print("Estimate:", fae_result.estimation)