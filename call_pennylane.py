# 참고 : https://docs.pennylane.ai/en/stable/code/api/pennylane.QuantumMonteCarlo.html

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

m = 5
M = 2 ** m

xmax = np.pi  # bound to region [-pi, pi]
xs = np.linspace(-xmax, xmax, M)

probs = np.array([norm().pdf(x) for x in xs])
probs /= np.sum(probs)  # standard normal distribution
# plt.figure()
# plt.plot(xs,probs,"ro-")
# plt.show()

# func = lambda i: np.sin(xs[i]) ** 2
r = 0.0; S0 = 100.0; K = 100.0; vol = 0.2; T = 1.0 
S = lambda i: S0*np.exp((r-vol**2/2)*T + vol*np.sqrt(T)*xs[i])
payoff = lambda i: max(S(i)-K,0)
ys = [payoff(i) for i in range(len(xs))]
scale_factor = max(ys)+1
func = lambda i: payoff(i)/scale_factor
# plt.figure()
# plt.plot(xs,ys,"ro-")
# plt.show()

def BSexact():
    d1 = (np.log(S0/K) + (r +vol**2/2)*T)/vol/np.sqrt(T)
    d2 = d1 - vol*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

n = 10
N = 2 ** n

target_wires = range(m + 1)
estimation_wires = range(m + 1, n + m + 1)

dev = qml.device("default.qubit", wires=(n + m + 1))

@qml.qnode(dev)
def circuit():
    qml.templates.QuantumMonteCarlo(
        probs,
        func,
        target_wires=target_wires,
        estimation_wires=estimation_wires,
    )
    return qml.probs(estimation_wires)

phase_estimated = np.argmax(circuit()[:int(N / 2)]) / N
simVal = (1 - np.cos(np.pi * phase_estimated)) / 2 * scale_factor
exactVal = BSexact()
print("sim val = ", simVal, "\nexact val = ", exactVal)