# pip install pennylane autograd scipy
import autograd.numpy as np
import pennylane as qml
from autograd import jacobian
from scipy.linalg import expm

# ---------- ODE: du/dt = A u ----------
A = np.array([
    [-0.1,  0.4,  0.2, -0.7],
    [ 0.9,  0.1, -0.1, -1.1],
    [ 0.5,  0.2, -0.4, -0.5],
    [ 0.6,  0.5,  0.3, -1.6]
], dtype=float)

n = 2                  # 2 qubits -> dim = 4
dim = 2**n
L = 4                  # Ansatz depth

dev = qml.device("default.qubit", wires=n)

# ---------- Ansatz ----------
def ring_entangle():
    for q in range(n):
        qml.CNOT(wires=[q, (q + 1) % n])

def apply_layer(angles):
    for w in range(n):
        qml.RY(angles[w], wires=w)
    ring_entangle()

@qml.qnode(dev, interface="autograd")
def state_from_betas(betas):
    qml.BasisState(np.array([0] * n), wires=range(n))
    for l in range(L):
        apply_layer(betas[l])
    return qml.state()  # complex, normalized

# ---------- θ unpacking ----------
def unpack_theta(theta):
    alpha = theta[0]
    betas = theta[1:].reshape(L, n)
    return alpha, betas

def psi_hat(theta):
    alpha, betas = unpack_theta(theta)
    return alpha * state_from_betas(betas)  # not normalized

# ---------- Real/Imag for Autograd ----------
def psi_hat_real(theta):
    return np.real(psi_hat(theta))

def psi_hat_imag(theta):
    return np.imag(psi_hat(theta))

# ---------- Build A_mat and C_vec ----------
def build_A_C(theta):
    psi = psi_hat(theta)  # complex, not normalized
    dpsi_real = jacobian(psi_hat_real)(theta)  # (dim, n_params)
    dpsi_imag = jacobian(psi_hat_imag)(theta)  # (dim, n_params)
    dpsi_dtheta = dpsi_real + 1j * dpsi_imag   # (dim, n_params)

    P = dpsi_dtheta.T                          # (n_params, dim)
    A_mat = (P @ P.conj().T).real              # (n_params, n_params)
    Hpsi = A @ psi                             # (dim,)
    C_vec = np.real(P @ Hpsi.conj())           # (n_params,)
    return A_mat, C_vec

# ---------- TDVP step ----------
def step(theta, dt, reg=1e-10):
    A_mat, C_vec = build_A_C(theta)
    theta_dot = np.linalg.solve(A_mat + reg * np.eye(len(theta)), C_vec)
    return theta + dt * theta_dot

# ---------- Run Simulation ----------
theta = np.zeros(1 + L * n)  # [α, β.flatten()]
theta[0] = 1.0               # α = 1 → |ψ⟩ = |00⟩

T, dt = 10.0, 0.02
ts = np.arange(0, T + 1e-9, dt)

traj = []
for t in ts:
    traj.append(psi_hat(theta))  # full state vector (not normalized)
    theta = step(theta, dt)

traj = np.stack(traj, axis=0)  # shape: (Nt, dim)

# ---------- Exact solution ----------
u0 = np.zeros(dim); u0[0] = 1.0
exact = np.stack([(expm(A * t) @ u0).real for t in ts], axis=0)

# ---------- Compare ----------
rms = np.sqrt(np.mean((traj.real - exact) ** 2))
print("RMS error vs exact:", float(rms))

for k in [0, 20, 40, 80, len(ts)-1]:
    print(f"t={ts[k]:.2f}  VQS = {traj[k].real}  exact = {exact[k]}")

import matplotlib.pyplot as plt

labels = [f"$u_{k}(t)$" for k in range(dim)]

plt.figure(figsize=(10, 6))
for i in range(dim):
    plt.plot(ts, exact[:, i], 'k--', linewidth=1.5, label=f"Exact {labels[i]}")
    plt.plot(ts, traj[:, i].real, label=f"VQS {labels[i]}", alpha=0.8)

plt.title("VQS vs Exact: Time Evolution of State Components")
plt.xlabel("Time $t$")
plt.ylabel("Component Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()