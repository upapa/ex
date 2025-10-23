import autograd.numpy as np
import pennylane as qml
from scipy.linalg import expm

# ============================ PDE → mode ODE ============================
# u_t = D u_xx, periodic BC on [0, Lx]
D = 0.02
Lx = 1.0            
K  = 4              
n  = 2              # qubits -> dim=4
assert 2**n == K, "지금 코드는 2^n = K 로 가정"

# wave numbers kappa_k = 2πk / Lx 
k_idx  = np.arange(1, K+1)
kappa  = 2*np.pi*k_idx / Lx
lmbda  = -D * (kappa**2)              # eigenvalues per mode
A = np.diag(lmbda.astype(float))      # ODE: a'(t) = A a(t)

# ============================ TDVP(McLachlan) / VQS params =========================
L = 4              # ansatz depth
T, dt = 4.0, 0.05  # integration window

# ------------------------ Pauli decomposition (2 qubits) ----------------
PAULI = {
    'I': np.array([[1,0],[0,1]], dtype=complex),
    'X': np.array([[0,1],[1,0]], dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
    'Z': np.array([[1,0],[0,-1]], dtype=complex),
}
def kronN(mats):
    out = np.array([[1]], dtype=complex)
    for M in mats:
        out = np.kron(out, M)
    return out

def decompose_to_pauli_2q(M, tol=1e-10):
    labels = ['I','X','Y','Z']
    terms = []
    for a in labels:
        for b in labels:
            P = kronN([PAULI[a], PAULI[b]])
            coeff = np.trace(P.conj().T @ M) / 4.0  # 1/2^n for n=2
            if abs(coeff) > tol:
                terms.append((coeff, [(a,0),(b,1)]))
    return terms

H_terms = decompose_to_pauli_2q(A)

# ------------------------ Devices ------------------------
dev_state    = qml.device("default.qubit", wires=n)
dev_hadamard = qml.device("default.qubit", wires=1+n)
AUX = 0
DATA = [1,2][:n]

# ------------------------ Ansatz on hadamard device -------------------
def ring_entangle_h():
    for q in range(n):
        qml.CNOT(wires=[DATA[q], DATA[(q+1)%n]])

def apply_layer_h(angles):
    for w in range(n):
        qml.RY(angles[w], wires=DATA[w])
    ring_entangle_h()

def apply_ansatz_h(betas):
    for l in range(betas.shape[0]):
        apply_layer_h(betas[l])

def apply_ansatz_with_Y_on_h(betas, target_l, target_w):
    for l in range(betas.shape[0]):
        for w in range(n):
            qml.RY(betas[l, w], wires=DATA[w])
            if (l == target_l) and (w == target_w):
                qml.Y(wires=DATA[w])
        ring_entangle_h()

# ------------------------ Ansatz on state device ----------------------
@qml.qnode(dev_state, interface="autograd")
def state_from_betas(betas):
    qml.BasisState(np.array([0]*n), wires=range(n))
    for l in range(betas.shape[0]):
        for w in range(n):
            qml.RY(betas[l, w], wires=w)
        for q in range(n):
            qml.CNOT(wires=[q, (q+1)%n])
    return qml.state()

@qml.qnode(dev_state, interface="autograd")
def expval_of_word(betas, word):
    qml.BasisState(np.array([0]*n), wires=range(n))
    for l in range(betas.shape[0]):
        for w in range(n):
            qml.RY(betas[l, w], wires=w)
        for q in range(n):
            qml.CNOT(wires=[q, (q+1)%n])

    ops = []
    for (label, wi) in word:
        if label == 'I':
            continue
        ops.append(getattr(qml, f"Pauli{label}")(wires=wi))
    if not ops:
        return qml.expval(qml.Identity(0))
    return qml.expval(qml.prod(*ops))

def energy_value(betas):
    val = 0.0 + 0.0j
    for coeff, word in H_terms:
        val += coeff * expval_of_word(betas, word)
    return float(np.real(val))

# ------------------------ Controlled-call helpers ---------------------
def ctrl1(fn, control):
    def wrapped(*args, **kwargs):
        qml.ctrl(fn, control=control)(*args, **kwargs)
    return wrapped

def ctrl0(fn, control):
    def wrapped(*args, **kwargs):
        qml.PauliX(wires=control)
        qml.ctrl(fn, control=control)(*args, **kwargs)
        qml.PauliX(wires=control)
    return wrapped

def apply_pauli_word_on_data(word):
    for (label, wi) in word:
        if label == 'I':
            continue
        getattr(qml, f"Pauli{label}")(wires=DATA[wi])

# ------------------------ Hadamard tests: Aik, Ck ---------------------
@qml.qnode(dev_hadamard, interface="autograd")
def measure_Aik_raw(betas, i_l, i_w, k_l, k_w):
    qml.H(wires=AUX)
    ctrl0(apply_ansatz_with_Y_on_h, control=AUX)(betas, k_l, k_w)
    ctrl1(apply_ansatz_with_Y_on_h, control=AUX)(betas, i_l, i_w)
    qml.H(wires=AUX)
    return qml.expval(qml.PauliZ(AUX))

def Aik_hadamard(betas, i_l, i_w, k_l, k_w):
    raw = measure_Aik_raw(betas, i_l, i_w, k_l, k_w)
    return float(raw) / 4.0   # (∂ scale 보정)

@qml.qnode(dev_hadamard, interface="autograd")
def measure_Ck_term_raw(betas, k_l, k_w, word):
    qml.H(wires=AUX)
    ctrl0(apply_ansatz_with_Y_on_h, control=AUX)(betas, k_l, k_w)
    def branch1(betas_inner, word_inner):
        apply_ansatz_h(betas_inner)
        apply_pauli_word_on_data(word_inner)
    ctrl1(branch1, control=AUX)(betas, word)
    qml.H(wires=AUX)
    return qml.expval(qml.PauliZ(AUX))

def Ck_hadamard(betas, k_l, k_w):
    total = 0.0 + 0.0j
    for coeff, word in H_terms:
        raw = measure_Ck_term_raw(betas, k_l, k_w, word)
        total += coeff * (float(raw) / 2.0)
    return float(np.real(total))

def assemble_A_C_hadamard(betas):
    M = L * n
    A_mat = np.zeros((M, M))
    C_vec = np.zeros(M)
    def lw(j): return (j // n, j % n)
    for k in range(M):
        lk, wk = lw(k)
        C_vec[k] = Ck_hadamard(betas, lk, wk)
        for i in range(M):
            li, wi = lw(i)
            A_mat[k, i] = Aik_hadamard(betas, li, wi, lk, wk)
    return A_mat, C_vec

# ------------------------ One TDVP step -------------------------------
def step(betas, alpha, dt, reg=1e-8):
    A_mat, C_vec = assemble_A_C_hadamard(betas)
    beta_flat = betas.reshape(-1)
    beta_dot = np.linalg.solve(A_mat + reg*np.eye(A_mat.shape[0]), C_vec)
    betas_new = (beta_flat + dt * beta_dot).reshape(L, n)
    E = energy_value(betas)        
    alpha_new = alpha + dt * E * alpha
    return betas_new, alpha_new

# ------------------------ Run VQS on PDE-in-modes ---------------------
betas = np.zeros((L, n))   
alpha = 1.0
ts = np.arange(0, T + 1e-12, dt)

traj = []
for t in ts:
    psi_norm = state_from_betas(betas)
    psi_hat  = alpha * psi_norm  
    traj.append(np.real(psi_hat[:K]))
    betas, alpha = step(betas, alpha, dt)

traj = np.stack(traj, axis=0)  # shape: (Nt, 4)

# ------------------------ Compare: coefficients space ------------------
# exact : k(t) = a_k(0) e^{lambda_k t}, a(0)=[1,0,0,0]
u0_coef = np.zeros(K); u0_coef[0] = 1.0
exact_coef = np.stack([(np.exp(lmbda * t) * u0_coef).real for t in ts], axis=0)

coef_rms = np.sqrt(np.mean((traj.real - exact_coef)**2))
print("RMS (coeff space) vs exact:", float(coef_rms))

for idx in [0, min(20,len(ts)-1), min(40,len(ts)-1), len(ts)-1]:
    print(f"t={ts[idx]:.2f}")
    print("  VQS coef =", traj[idx].real)
    print("  exact    =", exact_coef[idx])

# ------------------------ Reconstruct u(x,t) and compare --------------
Nx = 128
xs = np.linspace(0.0, Lx, Nx, endpoint=False)

def synth_from_coef(a_vec, xgrid):
    # u(x) = sum_{k=1..K} a_k sin(2π k x / Lx)
    out = np.zeros_like(xgrid, dtype=float)
    for k in range(1, K+1):
        out += a_vec[k-1] * np.sin(2*np.pi*k*xgrid/Lx)
    return out

u_vqs = np.stack([synth_from_coef(traj[t].real, xs) for t in range(len(ts))], axis=0)
u_exact = np.stack([np.sin(2*np.pi*xs/Lx) * np.exp(lmbda[0]*t) for t in ts], axis=0)

space_rms = np.sqrt(np.mean((u_vqs - u_exact)**2))
print("RMS (physical space u(x,t)) vs exact:", float(space_rms))


import matplotlib.pyplot as plt


# ------------------------ Plot: solution snapshots ---------------------
plt.figure(figsize=(10,6))
for idx in [0, min(20,len(ts)-1), min(40,len(ts)-1), len(ts)-1]:
    plt.plot(xs, u_vqs[idx], label=f"VQS t={ts[idx]:.2f}")
    plt.plot(xs, u_exact[idx], "--", label=f"Exact t={ts[idx]:.2f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.title("Reconstructed solution snapshots")
plt.grid(True)
plt.show()