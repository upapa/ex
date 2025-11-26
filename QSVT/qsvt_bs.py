# -*- coding: utf-8 -*-

from __future__ import annotations
from scipy.stats import norm
import numpy as np
import pennylane as qml
from pennylane import transform_angles
import pyqsp
from pyqsp.angle_sequence import QuantumSignalProcessingPhases


## use hadamrad version

# =========================
# 0) Utilities
# =========================
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return v / n

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

# =========================
# 1) Transforms: pick (alpha, beta) so u_tau = (1/2) sigma^2 u_xx
# =========================
def bs_transform_params(r: float, sigma: float, q: float = 0.0):
    alpha = 0.5 - (r - q) / (sigma**2)
    beta  = -(sigma**2)/8.0 - (r - q)/2.0 - ((r - q)**2)/(2.0 * sigma**2)
    return alpha, beta

# =========================
# 2) Grid, CN matrices, boundary
# =========================
def build_cn_matrices(M: int, x_min: float, x_max: float, sigma: float, dtau: float, theta: float = 0.5):
    if M < 2 or not is_power_of_two(M):
        raise ValueError("Choose M as a power of two (e.g., 8,16,32,64).")
    nu = 0.5 * sigma**2
    h  = (x_max - x_min) / (M + 1)
    r  = (nu * dtau) / (h**2)
    main = (-2.0 / h**2) * np.ones(M)
    off  = ( 1.0 / h**2) * np.ones(M - 1)
    L    = np.diag(main) + np.diag(off, +1) + np.diag(off, -1)
    I    = np.eye(M)
    A    = I - theta * dtau * nu * L
    B    = I + (1.0 - theta) * dtau * nu * L
    return A, B, h, r

def payoff_uN_loggrid(M: int, x_min: float, x_max: float, alpha: float, K: float, kind: str = "call"):
    xs  = np.linspace(x_min, x_max, M + 2)  # include boundaries
    xin = xs[1:-1]
    S   = K * np.exp(xin)
    if kind == "call":
        payoff = np.maximum(S - K, 0.0)
    elif kind == "put":
        payoff = np.maximum(K - S, 0.0)
    else:
        raise ValueError("Unsupported payoff kind")
    # u = e^{-alpha x} V  (omit beta*T in u^N; cancels during marching)
    uN = np.exp(-(alpha * xin)) * payoff
    return uN, xin, S

def u_boundary_from_V_boundary(x: float, tau: float, alpha: float, beta: float, V_boundary: float) -> float:
    """Map V boundary value at (x, tau) to u boundary via V = e^{alpha x + beta tau} u."""
    return np.exp(-(alpha * x + beta * tau)) * V_boundary

def build_boundary_histories_call(
    N: int, x_min: float, x_max: float, alpha: float, beta: float,
    r: float, T: float, K: float
):
    """Natural boundaries for a European call:
       S->0:  V ~ 0
       S->∞:  V ~ S - K e^{-r t}
       Map to u using V = e^{alpha x + beta tau} u, with tau = T - t.
       Return arrays gL_hist[n] = u_L at tau_n, gR_hist[n] = u_R at tau_n  (n=0..N)
    """
    taus = np.linspace(0.0, T, N+1)  # tau_n
    gL_hist = []
    gR_hist = []
    for tau in taus:
        t = T - tau
        S_L = K * np.exp(x_min)
        S_R = K * np.exp(x_max)
        # V at boundaries (call):
        V_L = 0.0
        V_R = max(S_R - K * np.exp(-r * t), 0.0)
        uL = u_boundary_from_V_boundary(x_min, tau, alpha, beta, V_L)
        uR = u_boundary_from_V_boundary(x_max, tau, alpha, beta, V_R)
        gL_hist.append(uL)
        gR_hist.append(uR)
    return np.array(gL_hist), np.array(gR_hist)

def boundary_vec_dirichlet_from_hist(M: int, r_dimless: float, theta: float,
                                     gL_hist: np.ndarray, gR_hist: np.ndarray, n: int):
    """Build g^n using Dirichlet values at steps n+1 and n."""
    g = np.zeros(M)
    g[0]  = theta * r_dimless * gL_hist[n+1] + (1.0 - theta) * r_dimless * gL_hist[n]
    g[-1] = theta * r_dimless * gR_hist[n+1] + (1.0 - theta) * r_dimless * gR_hist[n]
    return g

# =========================
# 3) Closed-form BS price (vanilla call) for sanity check
# =========================

def bs_call_price(S: np.ndarray, K: float, r: float, q: float, sigma: float, T: float):
    if T <= 0:
        return np.maximum(S - K, 0.0)

    vol = sigma * np.sqrt(T)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol

    # S=0 같은 경계 처리를 안전하게
    d1 = np.where(S > 0, d1, -np.inf)
    d2 = np.where(S > 0, d2, -np.inf)

    # 벡터화된 정규 CDF
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    return np.exp(-q*T) * S * Nd1 - np.exp(-r*T) * K * Nd2


# =========================
# 4) Classical Chebyshev inverse (for reference)
# =========================
def cheb_coeffs_inv_on_interval(a: float, b: float, deg: int) -> np.ndarray:
    if a <= 0:
        raise ValueError("Interval must be positive: a>0 for 1/x.")
    N = deg + 1
    thetas = (np.arange(N) + 0.5) * np.pi / N
    tjs = np.cos(thetas)
    xjs = 0.5 * ((b - a) * tjs + (b + a))
    fj  = 1.0 / xjs
    c = np.zeros(deg + 1)
    for k in range(deg + 1):
        c[k] = (2.0 / N) * np.sum(fj * np.cos(k * thetas))
    c[0] *= 0.5
    return c

def cheb_apply_matrix(A: np.ndarray, v: np.ndarray, a: float, b: float, c: np.ndarray) -> np.ndarray:
    I = np.eye(A.shape[0])
    S = (2.0 * A - (b + a) * I) / (b - a)  # t(A)
    m = len(c) - 1
    b1 = np.zeros_like(v)
    b2 = np.zeros_like(v)
    for k in range(m, 0, -1):
        bk = 2.0 * (S @ b1) - b2 + c[k] * v
        b2, b1 = b1, bk
    pSv = (S @ b1) - b2 + c[0] * v
    return pSv

def apply_Ainv_cheb(A: np.ndarray, b: np.ndarray, deg: int = 61, safety: float = 1.03) -> np.ndarray:
    evals = np.linalg.eigvalsh(A)
    lam_min = max(1e-12, float(np.min(evals)))
    lam_max = float(np.max(evals))
    a = lam_min / safety
    bnd = lam_max * safety
    c = cheb_coeffs_inv_on_interval(a, bnd, deg)
    return cheb_apply_matrix(A, b, a, bnd, c)

# =========================
# 5) QSVT helpers
# =========================
def qsvt_apply_Ainv_once(A: np.ndarray, b: np.ndarray, phases, eps: float = 3e-3) -> np.ndarray:
    """
    Apply QSVT(A/alpha) ~ A^{-1} to vector b.
    - No extra control ancilla.
    - Wires: [data ... ancillas ...]
    - Postselect ancillas = |0...0>.
    """
    M = A.shape[0]
    if not is_power_of_two(M):
        raise ValueError("M must be a power of two.")

    evals = np.linalg.eigh(A)[0]
    alpha_exact = float(np.max(evals))
    Atil = A / alpha_exact

    # 2) Wires: data first, then ancillas
    m = int(np.log2(M))
    control_ancilla = "ancilla1"           # 제어 큐비트 (홀수/짝수 다항식 선택용)
    block_ancilla_wires = list(range(m))     # BlockEncode용 앵실러 (인덱스 0..m-1)
    data_wires = list(range(m, 2*m))         # 데이터 큐비트 (인덱스 m..2m-1)

    # BlockEncode와 PCPhase가 적용될 큐비트들
    block_encoding_wires = block_ancilla_wires + data_wires

    # 회로의 전체 큐비트 리스트
    all_wires = [control_ancilla] + block_encoding_wires

    dev = qml.device("default.qubit", wires=all_wires)
    b_norm = normalize(b)

    @qml.qnode(dev, interface=None, diff_method=None)
    def circuit():
        # |b> on data_wires, |0...0> on others
        qml.StatePrep(b_norm, wires=data_wires)

        # Block-encode Atil (BlockEncode용 앵실러 + 데이터 큐비트 사용)
        block_encoding = qml.BlockEncode(Atil, wires=block_encoding_wires)

        # Projectors for QSVT
        projs = [qml.PCPhase(phi, dim=A.shape[0], wires=block_encoding_wires) for phi in phases]

        # 제어-QSVT (홀수 다항식 구현)
        qml.Hadamard(wires=control_ancilla)
        qml.ctrl(qml.QSVT, control=(control_ancilla,), control_values=[1])(block_encoding, projs)
        qml.ctrl(qml.adjoint(qml.QSVT), control=(control_ancilla,), control_values=[0])(block_encoding, projs)
        qml.Hadamard(wires=control_ancilla)

        return qml.state()

    psi = circuit()

    # 3) Postselect
    n_data = len(data_wires)
    n_block_anc = len(block_ancilla_wires)

    # Reshape: (control_ancilla, block_ancilla, data)
    psi_mat = psi.reshape((2, 2**n_block_anc, 2**n_data), order="C")

    # control_ancilla=0, block_ancilla=0...0 상태 선택
    sol = np.array(psi_mat[0, 0, :])

    # 4) 간단 스케일 복원(최소제곱)
    As  = A @ sol
    num = np.vdot(As, b)               # (A s)^H b
    den = np.vdot(As, As) + 1e-14
    c   = num / den
    return c * sol  # (unnormalized)


def cn_step_qsvt(A: np.ndarray, B: np.ndarray, phases, g: np.ndarray, u_next: np.ndarray,) -> np.ndarray:
    rhs = B @ u_next + g
    sol = qsvt_apply_Ainv_once(A, rhs, phases)
    return sol.real


def sanity_check_polynomial(A, phases):
    errs = []
    for j in range(A.shape[0]):
        b = np.zeros(A.shape[0]); b[j] = 1.0
        x = qsvt_apply_Ainv_once(A, b, phases)
        x_true = np.linalg.solve(A, b)
        denom = np.linalg.norm(x_true) + 1e-12
        errs.append(np.linalg.norm(x - x_true) / denom)
    print("median rel err over basis:", float(np.median(errs)))



# =========================
# 6) End-to-end
# =========================
if __name__ == "__main__":
    # ---- Market / contract ----
    r, sigma, q = 0.03, 0.20, 0.0
    K, T        = 100.0, 1.0
    kind        = "call"

    # ---- Transform ----
    alpha, beta = bs_transform_params(r, sigma, q)

    # ---- Grid/time (tunable) ----
    M      = 32          # power of two, x에대한 분해 (공간)
    x_min  = -4.0
    x_max  = +4.0
    N      = 1024   # tau에 대한 분해 (시간)
    dtau   = T / N
    theta  = 0.5  # 유한차분법의 가중치, 0.5 사용 -> Crank-Nicolson 방법

    # ---- Initial u^N from payoff on log-grid ----
    uN, xin, Sgrid = payoff_uN_loggrid(M, x_min, x_max, alpha, K, kind=kind)

    # ---- CN matrices ----
    A, B, h, r_dimless = build_cn_matrices(M, x_min, x_max, sigma, dtau, theta)

    # ---- Boundary histories (natural for call) ----
    gL_hist, gR_hist = build_boundary_histories_call(N, x_min, x_max, alpha, beta, r, T, K)

    evals = np.linalg.eigvalsh(A)                 # ← A 자체 스펙트럼
    alpha_exact = float(np.max(evals))            # 최대 고유값 (정확히)
    Atil  = A / alpha_exact
    lam_min = float(np.min(evals)) / alpha_exact  # Atil의 최소 고유값
    kappa_eff = 1.0 / lam_min                     # == λ_max(Atil)/λ_min(Atil) but λ_max=1

    pcoeffs,s=pyqsp.poly.PolyOneOverX().generate(kappa_eff,return_coef=True, ensure_bounded=True, return_scale=True)
    phi_pyqsp = QuantumSignalProcessingPhases(pcoeffs, signal_operator="Wx", tolerance=0.00001)
    phases = transform_angles(phi_pyqsp, "QSP", "QSVT")

    sanity_check_polynomial(A, phases)

    xs = np.linspace(lam_min, 1.0, 200)
    errs=[]
    for x in xs:
        block = qml.BlockEncode(np.array([[x]]), wires=[0])
        projs = [qml.PCPhase(phi, dim=1, wires=[0]) for phi in phases]
        temp = qml.matrix(qml.QSVT, wire_order=[0])(block, projs)
        y = np.real(temp[0,0])
        errs.append(abs(y - s/x))  # s는 pyqsp가 준 scale
    print("max sup error on [mu,1]:", max(errs))

    # ---- March backward in tau: QSVT path ----
    u_next_q = uN.copy()
    for n in reversed(range(N)):
        g_n = boundary_vec_dirichlet_from_hist(M, r_dimless, theta, gL_hist, gR_hist, n)
        u_now_q = cn_step_qsvt(A, B, phases, g_n, u_next_q)
        u_next_q = u_now_q

    u0_q = u_next_q
    V0_q = np.exp(alpha * xin) * u0_q

    # ---- Classical reference (Chebyshev polynomial inverse) ----
    u_next_c = uN.copy()
    for n in reversed(range(N)):
        g_n = boundary_vec_dirichlet_from_hist(M, r_dimless, theta, gL_hist, gR_hist, n)
        rhs = B @ u_next_c + g_n
        u_now_c = apply_Ainv_cheb(A, rhs, deg=81, safety=1.02)
        u_next_c = u_now_c
    u0_c = u_next_c
    V0_c = np.exp(alpha * xin) * u0_c

    # ---- Closed-form BS price on same S-grid ----
    V0_cf = bs_call_price(Sgrid, K, r, q, sigma, T)

    # ---- Report ----
    jK = np.argmin(np.abs(Sgrid - K))
    print(f"[SETUP] M={M}, N={N}, dtau={dtau:.4f}, grid x in [{x_min},{x_max}]")
    print(f"[POINT] S≈K -> S={Sgrid[jK]:.2f}")
    print(f"  V_QSVT(S≈K) : {V0_q[jK]:.6f}")
    print(f"  V_CHEB(S≈K) : {V0_c[jK]:.6f}")
    print(f"  V_CF  (S≈K) : {V0_cf[jK]:.6f}")
    print(f"  |QSVT-CF| : {abs(V0_q[jK]-V0_cf[jK]):.6f}")
    print(f"  |CHEB-CF  | : {abs(V0_c[jK]-V0_cf[jK]):.6f}")

    # Few grid samples
    for frac in [0.2, 0.5, 0.8]:
        j = int(frac * (M-1))
        print(f"x={xin[j]: .3f}, S={Sgrid[j]:.2f} | V_QSVT={V0_q[j]:.6f}, V_CHEB={V0_c[j]:.6f}, V_CF={V0_cf[j]:.6f}")
