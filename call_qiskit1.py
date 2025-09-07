# ref: https://qiskit-community.github.io/qiskit-finance/tutorials/03_european_call_option_pricing.html

import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution

# number of qubits to represent the uncertainty
num_uncertainty_qubits = 3

# parameters for considered random distribution
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)

# plot probability distribution
x = uncertainty_model.values
y = uncertainty_model.probabilities
plt.bar(x, y, width=0.2)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.grid()
plt.xlabel("Spot Price at Maturity $S_T$ (\$)", size=15)
plt.ylabel("Probability ($\%$)", size=15)
# plt.show()

# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1.896

# set the approximation scaling for the payoff function
c_approx = 0.25

# setup piecewise linear objective fcuntion
breakpoints = [low, strike_price]
slopes = [0, 1]
offsets = [0, 0]
f_min = 0
f_max = high - strike_price
european_call_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx,
)

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
num_qubits = european_call_objective.num_qubits
european_call = QuantumCircuit(num_qubits)
european_call.append(uncertainty_model, range(num_uncertainty_qubits))
european_call.append(european_call_objective, range(num_qubits))

## draw the circuit
# european_call.draw()

# # plot exact payoff function (evaluated on the grid of the uncertainty model)
# x = uncertainty_model.values
# y = np.maximum(0, x - strike_price)
# plt.plot(x, y, "ro-")
# plt.grid()
# plt.title("Payoff Function", size=15)
# plt.xlabel("Spot Price", size=15)
# plt.ylabel("Payoff", size=15)
# plt.xticks(x, size=15, rotation=90)
# plt.yticks(size=15)
# plt.show()

# evaluate exact expected value (normalized to the [0, 1] interval)
exact_value = np.dot(uncertainty_model.probabilities, y)
exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])
print("exact expected value:\t%.4f" % exact_value)
print("exact delta value:   \t%.4f" % exact_delta)

# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

problem = EstimationProblem(
    state_preparation=european_call,
    objective_qubits=[3],
    post_processing=european_call_objective.post_processing,
)
# # construct amplitude estimation
# ae = IterativeAmplitudeEstimation(
#     epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 100, "seed": 75})
# )
# result = ae.estimate(problem)
# conf_int = np.array(result.confidence_interval_processed)
# print("Exact value:        \t%.4f" % exact_value)
# print("Estimated value:    \t%.4f" % (result.estimation_processed))
# print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))

from qiskit_algorithms import FasterAmplitudeEstimation
ae = FasterAmplitudeEstimation(
    delta=0.01,  # target accuracy
    maxiter=3,  # determines the maximal power of the Grover operator
    sampler=Sampler()
)
result = ae.estimate(problem)
print(result)


# # qiskit finance module
# from qiskit_finance.applications.estimation import EuropeanCallPricing

# european_call_pricing = EuropeanCallPricing(
#     num_state_qubits=num_uncertainty_qubits,
#     strike_price=strike_price,
#     rescaling_factor=c_approx,
#     bounds=(low, high),
#     uncertainty_model=uncertainty_model,
# )

# # set target precision and confidence level
# epsilon = 0.01
# alpha = 0.05

# problem = european_call_pricing.to_estimation_problem()
# # construct amplitude estimation
# ae = IterativeAmplitudeEstimation(
#     epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 100, "seed": 75})
# )
# result = ae.estimate(problem)

# conf_int = np.array(result.confidence_interval_processed)
# print("Exact value:        \t%.4f" % exact_value)
# print("Estimated value:    \t%.4f" % (european_call_pricing.interpret(result)))
# print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))