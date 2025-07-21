# 참고 : https://docs.pennylane.ai/en/stable/code/api/pennylane.QuantumMonteCarlo.html

# parameter for opton pricing 
r = 0.0; K = 100.0; vol = 0.2; T = 1.0 


import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

m = 5
M = 2 ** m
n = 10
N = 2 ** n

xmax = np.pi  # bound to region [-pi, pi]
xs = np.linspace(-xmax, xmax, M)

probs = np.array([norm().pdf(x) for x in xs])
probs /= np.sum(probs)  # standard normal distribution
# plt.figure()
# plt.plot(xs,probs,"ro-")
# plt.show()

# func = lambda i: np.sin(xs[i]) ** 2
def getFunc(S0):
    S = lambda i: S0*np.exp((r-vol**2/2)*T + vol*np.sqrt(T)*xs[i])
    payoff = lambda i: max(S(i)-K,0)
    ys = [payoff(i) for i in range(len(xs))]
    scale_factor = max(ys)+1
    return (lambda i: payoff(i)/scale_factor, scale_factor)

def BSexact(S0):
    d1 = (np.log(S0/K) + (r +vol**2/2)*T)/vol/np.sqrt(T)
    d2 = d1 - vol*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

target_wires = range(m + 1)
estimation_wires = range(m + 1, n + m + 1)

dev = qml.device("default.qubit", wires=(n + m + 1))

@qml.qnode(dev)
def circuit(func):
    qml.templates.QuantumMonteCarlo(
        probs,
        func,
        target_wires=target_wires,
        estimation_wires=estimation_wires,
    )
    return qml.probs(estimation_wires)

def qMC(S0): 
    func, scale_factor = getFunc(S0)
    phase_estimated = np.argmax(circuit(func)[:int(N / 2)]) / N
    simVal = np.exp(-r*T)*(1 - np.cos(np.pi * phase_estimated)) / 2 * scale_factor
    return simVal 

S0_exact = range(70,130)
final_payoff = [max(x-K,0) for x in S0_exact]
val_exact = [BSexact(x) for x in S0_exact]

S0_qmc = range(70,131,10)
val_qmc = [qMC(x) for x in S0_qmc]
# exactVal = BSexact(S0)
# print("sim val = ", simVal, "\nexact val = ", exactVal)


plt.figure()
plt.plot(S0_exact,val_exact,"b-")
plt.plot(S0_exact,final_payoff,"r-")
plt.plot(S0_qmc,val_qmc,"ro")
plt.show()
