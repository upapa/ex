"""
Six-way comparison for ODE/PDE time integration (with paper-style & optimal Chebyshev QAE),
with strict thread caps (64) including Qiskit Aer, progress logging, and thread diagnostics.

Algorithms compared:
1) Classical (Clenshaw–Curtis time quadrature + exact weighted average)
2) QAE-Uniform (Uniform trapezoid time quadrature + QAE: IAE/MLAE)
3) QAE-Chebyshev (paper-style): Kacewicz time partition + Chebyshev knots + dense sampling via
   Chebyshev barycentric + [0,1] shift/scale + QAE   (per-interval QAE, 2*Nk calls)
4) QAE-Uniform (paper-style): Kacewicz time partition + Uniform coarse knots + dense sampling via
   linear interpolation + [0,1] shift/scale + QAE     (per-interval QAE, 2*Nk calls)
5) Cheb-Spectral: Chebyshev time-spectral collocation per step (no QAE)
6) QAE-Cheb_Optimal (NEW): Kacewicz partition + Chebyshev coarse knots + cubic-spline dense sampling,
   then single global [0,1] normalization and ONE QAE call per component (cos/sin) for entire [0,h]
   → total 2 QAE calls per lambda per step-factor (outside the loop; reused across steps)

Problems:
- Scalar ODE: y'(t) = λ y(t), y(0)=1
- Heat 1D (periodic):      u_t = D u_xx
- Convection–Diffusion 1D: u_t + v u_x = D u_xx
"""

from __future__ import annotations

# -------- Thread caps (set BEFORE numpy/qiskit imports) --------
import os as _os
_QPDE_MAX_THREADS = int(_os.environ.get("QPDE_MAX_THREADS", "64"))
if _QPDE_MAX_THREADS > 0:
    for _var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"
    ):
        _os.environ.setdefault(_var, str(_QPDE_MAX_THREADS))

# ===== Thread diagnostics =====
def report_threads(where: str = ""):
    import os, sys, threading
    print("\n[THREAD-CHECK] ----", where, "----")
    print("[THREAD-CHECK] pid =", os.getpid(), "| python =", sys.version.split()[0])
    caps = ["QPDE_MAX_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","RAYON_NUM_THREADS"]
    print("[THREAD-CHECK] caps:", {k: os.environ.get(k) for k in caps})
    print("[THREAD-CHECK] threading.active_count() =", threading.active_count())
    try:
        with open("/proc/self/status","r") as f:
            for line in f:
                if line.startswith("Threads:"):
                    print("[THREAD-CHECK] /proc/self/status Threads =", line.strip().split()[-1])
                    break
    except Exception as e:
        print("[THREAD-CHECK] /proc/self/status not available:", type(e).__name__)
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        for p in pools:
            print("[THREAD-CHECK] BLAS pool:", {k:p.get(k) for k in ("internal_api","num_threads","prefix","filepath")})
    except Exception as e:
        print("[THREAD-CHECK] threadpoolctl not available:", type(e).__name__)
    print("[THREAD-CHECK] -----------------------------\n")


# -------- Standard imports --------
import numpy as np
import os, time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Dict, Any, Sequence, Optional
from functools import lru_cache

# ============================================================
# Optional Qiskit imports (QAE 경로에서만 사용)
# ============================================================
HAS_QISKIT = True
try:
    try:
        from qiskit_algorithms import (
            EstimationProblem,
            MaximumLikelihoodAmplitudeEstimation,
            IterativeAmplitudeEstimation,
        )
    except Exception:
        from qiskit.algorithms import (  # fallback
            EstimationProblem,
            MaximumLikelihoodAmplitudeEstimation,
            IterativeAmplitudeEstimation,
        )
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, GroverOperator, RYGate
    from qiskit.primitives import Sampler
except Exception:
    HAS_QISKIT = False

# ---- Qiskit Aer (있으면 사용) + Sampler 생성 헬퍼 ----
_HAS_AER = False
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Sampler as AerSampler
    _HAS_AER = True
except Exception:
    _HAS_AER = False

def _make_sampler(shots: int, max_threads: int | None = None, seed: int | None = None):
    """
    Create a Qiskit Sampler with strict thread caps.
    - If Aer is available: use AerSimulator and cap OpenMP + disable exp/shot parallelism.
    - Else: use Terra Sampler (thread caps enforced via env vars).
    """
    import os
    if max_threads is None:
        max_threads = int(os.environ.get("QPDE_MAX_THREADS", "64"))
    max_threads = max(1, min(max_threads, os.cpu_count() or max_threads))

    if _HAS_AER:
        try:
            backend = AerSimulator()
            backend.set_options(
                max_parallel_threads=max_threads,
                max_parallel_experiments=1,
                max_parallel_shots=1
            )
            run_opts = {"shots": shots}
            if seed is not None:
                run_opts["seed_simulator"] = seed
            sp = AerSampler(backend=backend, run_options=run_opts)
            try: report_threads("Sampler created (Aer)")
            except: pass
            return sp
        except Exception:
            pass  # Aer 실패 시 Terra로 폴백

    try:
        from qiskit.primitives import Sampler as TerraSampler
        opts = {"shots": shots}
        if seed is not None:
            opts["seed_simulator"] = seed
        sp = TerraSampler(options=opts)
        try: report_threads("Sampler created (Terra)")
        except: pass
        return sp
    except Exception:
        from qiskit.primitives import Sampler as TerraSampler
        sp = TerraSampler()
        try: report_threads("Sampler created (Terra fallback)")
        except: pass
        return sp


# ============================================================
# A. 시간 구적 규칙
# ============================================================
@lru_cache(maxsize=64)
def _cc_nodes_weights_unit(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Clenshaw–Curtis nodes/weights on [-1,1], exact for Chebyshev moments up to degree N."""
    if N < 1:
        return np.array([0.0]), np.array([2.0])
    theta = np.pi * np.arange(N + 1) / N
    C = np.cos(np.outer(np.arange(N + 1), theta))
    rhs = np.zeros(N + 1); rhs[0] = 2.0
    for k in range(1, N // 2 + 1):
        rhs[2*k] = -2.0 / (4.0*k*k - 1.0)
    w = np.linalg.solve(C, rhs)
    x = np.cos(theta)
    return x, w

def clenshaw_curtis_nodes_weights(h: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    if K <= 0: raise ValueError("K must be >= 1")
    N = max(1, K - 1)
    x, w = _cc_nodes_weights_unit(N)
    t = 0.5*(x + 1.0)*h
    w = 0.5*h*w
    order = np.argsort(t)
    return t[order], w[order]

def uniform_trap_nodes_weights(h: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    if K <= 0: raise ValueError("K must be >= 1")
    if K == 1: return np.array([0.5*h]), np.array([h])
    nodes = np.linspace(0.0, h, K)
    w = np.zeros(K); w[0] = w[-1] = 0.5*h/(K-1); w[1:-1] = h/(K-1)
    return nodes, w

def cheb_trap_nodes_weights(h: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    if K <= 0: raise ValueError("K must be >= 1")
    if K == 1: return np.array([0.5*h]), np.array([h])
    N = K - 1
    nodes = (np.cos(np.pi*np.arange(0, K)/N) + 1.0)*0.5*h
    nodes = np.sort(nodes)
    w = np.zeros(K)
    w[0] = 0.5*(nodes[1]-nodes[0])
    for i in range(1, K-1): w[i] = 0.5*(nodes[i+1]-nodes[i-1])
    w[-1] = 0.5*(nodes[-1]-nodes[-2])
    return nodes, w

def uniform_simpson_nodes_weights(h: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    if K <= 0: raise ValueError("K must be >= 1")
    if (K - 1) % 2 != 0:
        raise ValueError("Simpson requires K=2m+1 (odd).")
    if K == 1: return np.array([0.5*h]), np.array([h])
    nodes = np.linspace(0.0, h, K)
    m = (K - 1)//2
    dx = h/(2*m)
    coef = np.zeros(K); coef[0]=coef[-1]=1; coef[1:-1:2]=4; coef[2:-1:2]=2
    w = coef*(dx/3.0)
    return nodes, w

def get_time_nodes_weights(rule: str, h: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    rule = rule.lower()
    if rule in ["clenshaw_curtis", "cc", "cheb_cc", "chebyshev_cc"]:
        return clenshaw_curtis_nodes_weights(h, K)
    elif rule in ["cheb_trap", "chebyshev_trap", "chebtrap"]:
        return cheb_trap_nodes_weights(h, K)
    elif rule in ["uniform_trap", "trap", "trapezoid"]:
        return uniform_trap_nodes_weights(h, K)
    elif rule in ["uniform_simpson", "simpson"]:
        return uniform_simpson_nodes_weights(h, K)
    else:
        raise ValueError(f"Unknown rule '{rule}'")


# ============================================================
# B. QAE 기반 가중 평균 (또는 CLASSICAL 평균)
# ============================================================
def _to_eval_schedule(schedule: Any) -> Sequence[int]:
    if isinstance(schedule, int):
        s = max(1, schedule)
        ms = [2**j - 1 for j in range(s)]
        if 0 not in ms: ms = [0] + ms
        return ms
    return list(schedule)

def build_state_prep_for_weighted_average(y_vals: Sequence[float], p_vals: Sequence[float]):
    if not HAS_QISKIT:
        raise RuntimeError("Qiskit is not available.")
    y = np.clip(np.asarray(y_vals, dtype=float), 0.0, 1.0)
    p = np.asarray(p_vals, dtype=float); p = p/p.sum()
    K = len(y); assert K == len(p)
    n_index = int(np.ceil(np.log2(K))) if K > 1 else 1
    num_qubits = n_index + 1; anc = n_index
    qc = QuantumCircuit(num_qubits, name="A")
    amps = np.zeros(2**n_index, dtype=float); amps[:K] = np.sqrt(p)
    sp = StatePreparation(amps); qc.append(sp, qargs=list(range(n_index)))
    for i in range(K):
        if y[i] <= 0.0: continue
        theta = 2.0*np.arcsin(np.sqrt(y[i]))
        bits = [(i >> b) & 1 for b in range(n_index)]
        for b, bit in enumerate(bits):
            if bit == 0: qc.x(b)
        ctrls = list(range(n_index))
        mc_ry = RYGate(theta).control(len(ctrls))
        qc.append(mc_ry, ctrls + [anc])
        for b, bit in enumerate(bits):
            if bit == 0: qc.x(b)
    return qc, anc

def qae_weighted_average(y_vals: Sequence[float], weights: Sequence[float],
                         algo: str = "MLAE", eval_schedule: Any = 4,
                         shots: int = 4000, sampler=None, seed: Optional[int] = None) -> float:
    w = np.asarray(weights, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("y must be in [0,1].")
    if np.all(w == 0): return 0.0
    if algo.upper() == "CLASSICAL" or not HAS_QISKIT:
        return float(np.dot(w, y) / w.sum())
    if sampler is None:
        sampler = _make_sampler(shots=shots, max_threads=_QPDE_MAX_THREADS, seed=seed)
    A, anc = build_state_prep_for_weighted_average(y, w)
    oracle = QuantumCircuit(A.num_qubits, name="Oracle"); oracle.z(anc)
    grover = GroverOperator(oracle=oracle, state_preparation=A)
    problem = EstimationProblem(state_preparation=A, objective_qubits=[anc], grover_operator=grover)
    if algo.upper() == "IAE":
        estimator = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, sampler=sampler)
    else:
        estimator = MaximumLikelihoodAmplitudeEstimation(
            evaluation_schedule=_to_eval_schedule(eval_schedule), sampler=sampler
        )
    res = estimator.estimate(problem)
    return float(res.estimation)


# ============================================================
# C. (구적 기반) 평균 → per-step (Classical / QAE-Uniform)
# ============================================================
def averaged_exp_lambda(lam: complex, h: float,
                        rule: str = "clenshaw_curtis", K: int = 16,
                        algo: str = "CLASSICAL", eval_schedule: Any = 4,
                        shots: int = 4000, sampler=None, seed: Optional[int] = None) -> complex:
    t, w = get_time_nodes_weights(rule, h, K)
    a = np.exp((lam.real)*t); b = lam.imag*t
    g_c = a*np.cos(b); g_s = a*np.sin(b)
    y_c = 0.5*(g_c + 1.0); y_s = 0.5*(g_s + 1.0)
    avg_c = 2.0*qae_weighted_average(y_c, w, algo=algo, eval_schedule=eval_schedule,
                                     shots=shots, sampler=sampler, seed=seed) - 1.0
    avg_s = 2.0*qae_weighted_average(y_s, w, algo=algo, eval_schedule=eval_schedule,
                                     shots=shots, sampler=sampler, seed=seed) - 1.0
    return avg_c + 1j*avg_s

def per_step_factor(lam: complex, h: float,
                    rule: str = "clenshaw_curtis", K: int = 16,
                    algo: str = "CLASSICAL", eval_schedule: Any = 4,
                    shots: int = 4000, sampler=None, seed: Optional[int] = None) -> complex:
    avg_exp = averaged_exp_lambda(lam, h, rule, K, algo, eval_schedule, shots, sampler, seed)
    return 1.0 + h*lam*avg_exp

def propagate_scalar_mode_qae(lam: complex, T: float, n: int, **avg_kwargs) -> complex:
    h = T/n
    step = per_step_factor(lam, h, **avg_kwargs)
    return step**n


# ============================================================
# D. (논문식) Chebyshev-QAE (paper-style, per-interval QAE)
# ============================================================
def _cheb_nodes_on_interval(a: float, b: float, K: int) -> np.ndarray:
    if K < 2:
        return np.array([(a + b)/2.0])
    N = K - 1
    x = np.cos(np.pi*np.arange(0, K)/N)[::-1]
    t = (x + 1.0)*0.5*(b - a) + a
    return t

def _barycentric_weights_cheb_extreme(K: int) -> np.ndarray:
    j = np.arange(0, K)
    c = np.ones(K); c[0] = 0.5; c[-1] = 0.5
    w = ((-1.0)**j)*c
    return w

def _barycentric_eval_cheb(a: float, b: float, y_vals: np.ndarray, xq: np.ndarray) -> np.ndarray:
    K = len(y_vals)
    if K == 1:
        return np.full_like(xq, y_vals[0], dtype=float)
    N = K - 1
    x_nodes_unit = np.cos(np.pi*np.arange(0, K)/N)[::-1]
    x_nodes = (x_nodes_unit + 1.0)*0.5*(b - a) + a
    w = _barycentric_weights_cheb_extreme(K)
    X = xq.reshape(-1, 1)
    diff = X - x_nodes.reshape(1, -1)
    close = np.isclose(diff, 0.0)
    out = np.empty(len(xq), dtype=float)
    if np.any(close):
        for i, row in enumerate(close):
            if row.any():
                out[i] = y_vals[row.argmax()]
            else:
                tmp = (w/(X[i,0] - x_nodes))*y_vals
                out[i] = np.sum(tmp)/np.sum(w/(X[i,0]-x_nodes))
    idxs = np.where(~close.any(axis=1))[0]
    for i in idxs:
        num = np.sum((w/(X[i,0] - x_nodes))*y_vals)
        den = np.sum(w/(X[i,0] - x_nodes))
        out[i] = num/den
    return out

def averaged_exp_lambda_chebQAE_paperstyle(
    lam: complex, h: float,
    Nk: int = 8, Knf: int = 10, Kns: int = 200,
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 4000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    tm = np.linspace(0.0, h, Nk + 1)
    def base_funcs(ts: np.ndarray):
        a = np.exp(lam.real * ts); b = lam.imag * ts
        return a*np.cos(b), a*np.sin(b)
    def avg_component(comp: str) -> float:
        acc_sum = 0.0
        for m in range(Nk):
            a, b = tm[m], tm[m+1]
            t_knots = _cheb_nodes_on_interval(a, b, Knf)
            gc, gs = base_funcs(t_knots)
            vals = gc if comp == "c" else gs
            t_dense = np.linspace(a, b, Kns)
            q_dense = _barycentric_eval_cheb(a, b, vals, t_dense)
            vmin, vmax = float(np.min(q_dense)), float(np.max(q_dense))
            if np.isclose(vmax, vmin):
                mean_patch = vmin
            else:
                y = (q_dense - vmin)/(vmax - vmin)
                w = np.ones_like(y)
                Ey = qae_weighted_average(y, w, algo=algo, eval_schedule=eval_schedule,
                                          shots=shots, sampler=sampler, seed=seed)
                mean_patch = vmin + (vmax - vmin)*Ey
            acc_sum += (b - a)*mean_patch
        return acc_sum / h
    avg_c = avg_component("c"); avg_s = avg_component("s")
    return avg_c + 1j*avg_s

def per_step_factor_chebQAE_paperstyle(
    lam: complex, h: float,
    Nk: int = 8, Knf: int = 10, Kns: int = 200,
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 4000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    avg_exp = averaged_exp_lambda_chebQAE_paperstyle(lam, h, Nk, Knf, Kns, algo, eval_schedule, shots, seed, sampler)
    return 1.0 + h*lam*avg_exp


# ============================================================
# E. Uniform-paper (linear interp)
# ============================================================
def _uniform_nodes_on_interval(a: float, b: float, K: int) -> np.ndarray:
    if K < 2: return np.array([(a + b)/2.0])
    return np.linspace(a, b, K)

def _linear_interp(x_nodes: np.ndarray, y_nodes: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x_nodes, y_nodes).astype(float)

def averaged_exp_lambda_uniformQAE_paperstyle(
    lam: complex, h: float,
    Nk: int = 8, Knf: int = 10, Kns: int = 200,
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 4000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    tm = np.linspace(0.0, h, Nk + 1)
    def base_funcs(ts: np.ndarray):
        a = np.exp(lam.real * ts); b = lam.imag * ts
        return a*np.cos(b), a*np.sin(b)
    def avg_component(comp: str) -> float:
        acc_sum = 0.0
        for m in range(Nk):
            a, b = tm[m], tm[m+1]
            t_knots = _uniform_nodes_on_interval(a, b, Knf)
            gc, gs = base_funcs(t_knots)
            vals = gc if comp == "c" else gs
            t_dense = np.linspace(a, b, Kns)
            q_dense = _linear_interp(t_knots, vals, t_dense)
            vmin, vmax = float(np.min(q_dense)), float(np.max(q_dense))
            if np.isclose(vmax, vmin):
                mean_patch = vmin
            else:
                y = (q_dense - vmin)/(vmax - vmin)
                w = np.ones_like(y)
                Ey = qae_weighted_average(y, w, algo=algo, eval_schedule=eval_schedule,
                                          shots=shots, sampler=sampler, seed=seed)
                mean_patch = vmin + (vmax - vmin)*Ey
            acc_sum += (b - a)*mean_patch
        return acc_sum / h
    avg_c = avg_component("c"); avg_s = avg_component("s")
    return avg_c + 1j*avg_s

def per_step_factor_uniformQAE_paperstyle(
    lam: complex, h: float,
    Nk: int = 8, Knf: int = 10, Kns: int = 200,
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 4000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    avg_exp = averaged_exp_lambda_uniformQAE_paperstyle(lam, h, Nk, Knf, Kns, algo, eval_schedule, shots, seed, sampler)
    return 1.0 + h*lam*avg_exp


# ============================================================
# F. Chebyshev 시간 스펙트럴(콜로케이션)
# ============================================================
def cheb_nodes_and_D(K: int) -> Tuple[np.ndarray, np.ndarray]:
    if K < 2:
        raise ValueError("Chebyshev collocation requires K>=2.")
    N = K - 1
    j = np.arange(0, N + 1)
    x = np.cos(np.pi * j / N)  # descending: x[0]=1 ... x[N]=-1
    c = np.ones(K); c[0]=2.0; c[-1]=2.0
    c = c * ((-1.0)**j)
    X = np.tile(x, (K,1))
    dX = X - X.T
    D = (c[:,None]/c[None,:])/(dX + np.eye(K))
    D = D - np.diag(np.sum(D, axis=1))
    return x, D

def cheb_spectral_step_factor_scalar(lam: complex, h: float, K: int) -> complex:
    x, D = cheb_nodes_and_D(K)
    alpha = 0.5*h*lam
    Kdim = K
    A = D.astype(complex).copy()
    left_idx = Kdim - 1
    for r in range(Kdim):
        if r == left_idx: continue
        A[r, :] -= alpha*np.eye(Kdim, dtype=complex)[r, :]
    b = np.zeros(Kdim, dtype=complex)
    A[left_idx, :] = 0.0; A[left_idx, left_idx] = 1.0; b[left_idx] = 1.0
    y = np.linalg.solve(A, b)
    right_idx = 0
    return complex(y[right_idx])

def cheb_spectral_step_factor(lam: np.ndarray | complex, h: float, K: int) -> np.ndarray | complex:
    lam_arr = np.asarray(lam)
    if lam_arr.shape == ():
        return cheb_spectral_step_factor_scalar(complex(lam_arr), h, K)
    uniq, inv = np.unique(lam_arr, return_inverse=True)
    S_uniq = np.array([cheb_spectral_step_factor_scalar(complex(L), h, K) for L in uniq], dtype=complex)
    S = S_uniq[inv].reshape(lam_arr.shape)
    return S

def propagate_scalar_mode_cheb(lam: complex, T: float, n: int, K: int) -> complex:
    h = T/n
    S = cheb_spectral_step_factor(lam, h, K)
    return S**n


# ============================================================
# G. Cubic Spline (natural) 구현  —— QAE_Cheb_Optimal 용
# ============================================================
def _natural_cubic_spline_coeffs(x: np.ndarray, y: np.ndarray):
    """
    Natural cubic spline on (x_i, y_i), nonuniform allowed.
    Returns arrays (a,b,c,d) per interval: S_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3, t in [x_i,x_{i+1}]
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2: raise ValueError("Need at least two points for spline.")
    h = np.diff(x)
    if np.any(h <= 0): raise ValueError("x must be strictly increasing.")
    # Build tridiagonal system for M (second derivatives), natural: M0=Mn-1=0
    if n == 2:
        # linear segment
        a = y[:-1].copy()
        b = (y[1:]-y[:-1])/h
        c = np.zeros(n-1); d = np.zeros(n-1)
        return a,b,c,d,x
    A = np.zeros((n-2, n-2))
    rhs = np.zeros(n-2)
    for i in range(n-2):
        hi = h[i]; hi1 = h[i+1]
        A[i,i] = 2*(hi+hi1)
        if i>0:  A[i,i-1] = hi
        if i<n-3: A[i,i+1] = hi1
        rhs[i] = 6*((y[i+2]-y[i+1])/hi1 - (y[i+1]-y[i])/hi)
    M = np.zeros(n)
    if n > 2:
        M[1:-1] = np.linalg.solve(A, rhs)
    # Coeffs
    a = y[:-1].copy()
    b = (y[1:]-y[:-1])/h - h*(2*M[:-1]+M[1:])/6.0
    c = M[:-1]/2.0
    d = (M[1:]-M[:-1])/(6.0*h)
    return a,b,c,d,x

def _natural_cubic_spline_eval(a,b,c,d,x_nodes,tq):
    """
    Evaluate natural cubic spline defined by coeffs on query points tq.
    """
    tq = np.asarray(tq, dtype=float)
    out = np.empty_like(tq, dtype=float)
    # locate intervals
    idx = np.searchsorted(x_nodes, tq, side='right') - 1
    idx = np.clip(idx, 0, len(x_nodes)-2)
    dt = tq - x_nodes[idx]
    out = a[idx] + b[idx]*dt + c[idx]*dt*dt + d[idx]*dt*dt*dt
    return out


# ============================================================
# H. QAE_Cheb_Optimal —— (NEW) cubic-spline + global normalization + 2 QAE calls total
# ============================================================
def averaged_exp_lambda_chebQAE_optimal(
    lam: complex, h: float,
    Nk: int = 8, Knf: int = 10, Kns: int = 200,
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 4000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    """
    Kacewicz partition into Nk sub-intervals. At each, take Knf Chebyshev extreme points,
    build natural cubic spline, sample uniformly Kns points. Then concatenate ALL dense samples
    over [0,h], do ONE global [0,1] normalization, and call QAE ONCE per component (cos/sin).
    Weights reflect sub-interval lengths.
    """
    tm = np.linspace(0.0, h, Nk + 1)
    t_all = []
    gc_all = []
    gs_all = []
    w_all = []

    # gather dense samples from all sub-intervals
    for m in range(Nk):
        a, b = tm[m], tm[m+1]
        # Chebyshev coarse knots:
        t_knots = _cheb_nodes_on_interval(a, b, Knf)
        # Evaluate true function on knots
        aexp = np.exp(lam.real * t_knots); bphi = lam.imag * t_knots
        gck = aexp * np.cos(bphi)
        gsk = aexp * np.sin(bphi)
        # Build natural cubic spline (nonuniform knots OK)
        ak,bk,ck,dk,xk = _natural_cubic_spline_coeffs(t_knots, gck)
        as_,bs_,cs_,ds_,xs_ = _natural_cubic_spline_coeffs(t_knots, gsk)
        # Dense queries
        t_dense = np.linspace(a, b, Kns)
        gc_dense = _natural_cubic_spline_eval(ak,bk,ck,dk,xk,t_dense)
        gs_dense = _natural_cubic_spline_eval(as_,bs_,cs_,ds_,xs_,t_dense)
        # weights for uniform dense in this interval
        w = np.full_like(t_dense, (b - a)/Kns, dtype=float)

        t_all.append(t_dense)
        gc_all.append(gc_dense)
        gs_all.append(gs_dense)
        w_all.append(w)

    t_all = np.concatenate(t_all)
    gc_all = np.concatenate(gc_all)
    gs_all = np.concatenate(gs_all)
    w_all = np.concatenate(w_all)        # sum(w_all) == h

    # ONE global min/max per component → [0,1], and ONE QAE call per comp.
    # cos-part
    vmin_c, vmax_c = float(np.min(gc_all)), float(np.max(gc_all))
    if np.isclose(vmax_c, vmin_c):
        avg_c = vmin_c
    else:
        y_c = (gc_all - vmin_c)/(vmax_c - vmin_c)
        Ey = qae_weighted_average(y_c, w_all, algo=algo, eval_schedule=eval_schedule,
                                  shots=shots, sampler=sampler, seed=seed)
        avg_c = vmin_c + (vmax_c - vmin_c)*Ey
    # sin-part
    vmin_s, vmax_s = float(np.min(gs_all)), float(np.max(gs_all))
    if np.isclose(vmax_s, vmin_s):
        avg_s = vmin_s
    else:
        y_s = (gs_all - vmin_s)/(vmax_s - vmin_s)
        Ey = qae_weighted_average(y_s, w_all, algo=algo, eval_schedule=eval_schedule,
                                  shots=shots, sampler=sampler, seed=seed)
        avg_s = vmin_s + (vmax_s - vmin_s)*Ey

    return avg_c + 1j*avg_s

def per_step_factor_chebQAE_optimal(
    lam: complex, h: float,
    Nk: int = 8, Knf: int = 10, Kns: int = 200,
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 4000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    avg_exp = averaged_exp_lambda_chebQAE_optimal(lam, h, Nk, Knf, Kns, algo, eval_schedule, shots, seed, sampler)
    return 1.0 + h*lam*avg_exp


# ============================================================
# NEW: QAE_Cheb_Develop — Adaptive partition + error-controlled recursion
#   - 목표: 전구간 cos/sin 각각 QAE 1회(총 2회) 호출
#   - 방법: [0,h]을 시작 구간으로 삼고, 각 구간 [a,b]에서:
#       (1) Chebyshev 극점 Knf0개에서 함수값 평가 → natural cubic spline 구성
#       (2) spline 기반 '전구간' 적분 I_coarse (Simpson with Kns0 samples)
#       (3) 좌/우 반분하여 같은 방식으로 적분 → I_left + I_right
#       (4) |(I_left+I_right) - I_coarse| <= tol_rel * (|I_left|+|I_right|) 이면 수용,
#           아니면 (depth<max_depth) 재귀 분할
#     최종적으로 모든 '수용된' 구간들의 조밀 샘플(균등 Kns0)을 모아
#     전역 min/max로 [0,1] 정규화 후 cos/sin 각각 QAE 1회 호출
# ============================================================

def _spline_integral_by_samples(t_dense: np.ndarray, q_dense: np.ndarray) -> float:
    """균등 샘플(t_dense, q_dense)로 [a,b] 적분 근사 (사다리꼴)."""
    return float(np.trapz(q_dense, t_dense))

def averaged_exp_lambda_chebQAE_develop(
    lam: complex, h: float,
    # 기본 파라미터 (필요시 튜닝)
    tol_rel: float = 1e-3, max_depth: int = 4,
    Knf0: int = 10, Kns0: int = 200,
    # QAE 옵션
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 2000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    """
    Adaptive subdivision with natural-cubic spline proxy and error-controlled recursion.
    최종적으로 전구간 조밀 샘플을 모아 전역 [0,1] 정규화 후 cos/sin 각각 QAE 1회 호출.
    """
    # 수집 버퍼(전구간)
    all_t, all_gc, all_gs, all_w = [], [], [], []

    def base_funcs(ts: np.ndarray):
        a = np.exp(lam.real * ts)
        b = lam.imag * ts
        return a * np.cos(b), a * np.sin(b)

    def build_spline_vals(a: float, b: float, Knf: int, Kns: int):
        """구간 [a,b]에서 Chebyshev 극점 Knf로 spline 만들고 균등 Kns 샘플 생성"""
        t_knots = _cheb_nodes_on_interval(a, b, Knf)
        gc_k, gs_k = base_funcs(t_knots)

        # natural cubic spline 계수
        ak,bk,ck,dk,xk = _natural_cubic_spline_coeffs(t_knots, gc_k)
        as_,bs_,cs_,ds_,xs_ = _natural_cubic_spline_coeffs(t_knots, gs_k)

        t_dense = np.linspace(a, b, Kns)
        gc_dense = _natural_cubic_spline_eval(ak,bk,ck,dk,xk, t_dense)
        gs_dense = _natural_cubic_spline_eval(as_,bs_,cs_,ds_,xs_, t_dense)
        return t_dense, gc_dense, gs_dense

    def recur(a: float, b: float, depth: int):
        """오류 기준으로 수용/재귀 분할"""
        # coarse 적분(한 번): Kns0 샘플
        t_coarse, gc_coarse, gs_coarse = build_spline_vals(a, b, Knf0, Kns0)
        Ic = _spline_integral_by_samples(t_coarse, gc_coarse)
        Is = _spline_integral_by_samples(t_coarse, gs_coarse)

        # 반분 적분(두 번): 좌/우 각각 Kns0//2 샘플
        mid = 0.5 * (a + b)
        kl = max(4, Kns0//2); kr = max(4, Kns0//2)

        tl, gcl, gsl = build_spline_vals(a, mid, Knf0, kl)
        tr, gcr, gsr = build_spline_vals(mid, b, Knf0, kr)

        Ic_lr = _spline_integral_by_samples(tl, gcl) + _spline_integral_by_samples(tr, gcr)
        Is_lr = _spline_integral_by_samples(tl, gsl) + _spline_integral_by_samples(tr, gsr)

        # 상대 오류 추정치
        denom_c = max(1e-16, abs(Ic_lr))
        denom_s = max(1e-16, abs(Is_lr))
        err_c = abs(Ic_lr - Ic) / denom_c
        err_s = abs(Is_lr - Is) / denom_s
        accept = (err_c <= tol_rel) and (err_s <= tol_rel)

        if accept or depth >= max_depth:
            # 수용: 조밀 샘플을 전역 버퍼에 누적 (가중치는 길이/Kns0)
            w = (b - a) / Kns0
            all_t.append(t_coarse)
            all_gc.append(gc_coarse); all_gs.append(gs_coarse)
            all_w.append(np.full_like(t_coarse, w, dtype=float))
        else:
            # 재귀 분할
            recur(a, mid, depth+1)
            recur(mid, b, depth+1)

    # 루트 구간 재귀
    recur(0.0, h, 0)

    # 전구간 데이터 결합
    t_all = np.concatenate(all_t) if len(all_t) else np.array([0.0, h])
    gc_all = np.concatenate(all_gc) if len(all_gc) else np.array([1.0, 1.0])
    gs_all = np.concatenate(all_gs) if len(all_gs) else np.array([0.0, 0.0])
    w_all  = np.concatenate(all_w)  if len(all_w)  else np.array([h/2, h/2])

    # 전역 [0,1] 정규화 후 cos/sin 각각 QAE 1회
    # cos
    vmin_c, vmax_c = float(np.min(gc_all)), float(np.max(gc_all))
    if np.isclose(vmax_c, vmin_c):
        avg_c = vmin_c
    else:
        y_c = (gc_all - vmin_c)/(vmax_c - vmin_c)
        Ey = qae_weighted_average(y_c, w_all, algo=algo, eval_schedule=eval_schedule,
                                  shots=shots, sampler=sampler, seed=seed)
        avg_c = vmin_c + (vmax_c - vmin_c)*Ey

    # sin
    vmin_s, vmax_s = float(np.min(gs_all)), float(np.max(gs_all))
    if np.isclose(vmax_s, vmin_s):
        avg_s = vmin_s
    else:
        y_s = (gs_all - vmin_s)/(vmax_s - vmin_s)
        Ey = qae_weighted_average(y_s, w_all, algo=algo, eval_schedule=eval_schedule,
                                  shots=shots, sampler=sampler, seed=seed)
        avg_s = vmin_s + (vmax_s - vmin_s)*Ey

    return avg_c + 1j*avg_s

def per_step_factor_chebQAE_develop(
    lam: complex, h: float,
    tol_rel: float = 1e-3, max_depth: int = 4,
    Knf0: int = 10, Kns0: int = 200,
    algo: str = "IAE", eval_schedule: Any = 4, shots: int = 2000,
    seed: Optional[int] = None, sampler=None
) -> complex:
    avg_exp = averaged_exp_lambda_chebQAE_develop(
        lam, h, tol_rel=tol_rel, max_depth=max_depth, Knf0=Knf0, Kns0=Kns0,
        algo=algo, eval_schedule=eval_schedule, shots=shots, seed=seed, sampler=sampler
    )
    return 1.0 + h*lam*avg_exp

# ============================================================
# I. PDE 솔버 (푸리에 모드별 전파)
# ============================================================
def _kappa_1d(Nx: int, L: float) -> np.ndarray:
    k = np.fft.fftfreq(Nx, d=L/Nx)
    return 2.0*np.pi*k

def _pde_metric(u_num: np.ndarray, u_ex: np.ndarray, L: float) -> float:
    dx = L/len(u_num)
    return float(np.sqrt(np.sum((u_num-u_ex)**2)*dx / max(1e-16, np.sum(u_ex**2)*dx)))

def linear1d_apply_step(u0: np.ndarray, lam: np.ndarray, step_factor: np.ndarray, T: float, L: float) -> Dict[str,Any]:
    U_T = u0 * (step_factor ** 1)  # step_factor already represents per-step; power handled by caller
    return {}

def linear1d_qae_solver(u0: Callable[[np.ndarray], np.ndarray],
                        lam_of_kappa: Callable[[np.ndarray], np.ndarray],
                        L: float, T: float, n: int, Nx: int,
                        rule: str, K: int,
                        algo: str, eval_schedule: Any, shots: int,
                        seed: Optional[int]) -> Dict[str, Any]:
    x = np.linspace(0.0, L, Nx, endpoint=False)
    U0 = np.fft.fft(u0(x).astype(complex))
    kappa = _kappa_1d(Nx, L)
    lam = lam_of_kappa(kappa)
    h = T/n
    uniq, inv = np.unique(lam, return_inverse=True)
    step_uniq = np.array([per_step_factor(Li, h, rule=rule, K=K, algo=algo,
                                          eval_schedule=eval_schedule, shots=shots, seed=seed)
                          for Li in uniq], dtype=complex)
    step = step_uniq[inv]
    U_T = U0*(step**n); U_ex = U0*np.exp(lam*T)
    u_num = np.fft.ifft(U_T).real; u_ex = np.fft.ifft(U_ex).real
    return {"u_num": u_num, "u_exact": u_ex, "L2_rel": _pde_metric(u_num, u_ex, L), "x": x}

def linear1d_qae_chebpaper_solver(u0: Callable[[np.ndarray], np.ndarray],
                                  lam_of_kappa: Callable[[np.ndarray], np.ndarray],
                                  L: float, T: float, n: int, Nx: int,
                                  Nk: int, Knf: int, Kns: int,
                                  algo: str, eval_schedule: Any, shots: int,
                                  seed: Optional[int]) -> Dict[str, Any]:
    x = np.linspace(0.0, L, Nx, endpoint=False)
    U0 = np.fft.fft(u0(x).astype(complex))
    kappa = _kappa_1d(Nx, L)
    lam = lam_of_kappa(kappa)
    h = T/n
    uniq, inv = np.unique(lam, return_inverse=True)
    step_uniq = np.array([per_step_factor_chebQAE_paperstyle(
                            Li, h, Nk=Nk, Knf=Knf, Kns=Kns,
                            algo=algo, eval_schedule=eval_schedule, shots=shots, seed=seed
                          ) for Li in uniq], dtype=complex)
    step = step_uniq[inv]
    U_T = U0*(step**n); U_ex = U0*np.exp(lam*T)
    u_num = np.fft.ifft(U_T).real; u_ex  = np.fft.ifft(U_ex).real
    return {"u_num": u_num, "u_exact": u_ex, "L2_rel": _pde_metric(u_num, u_ex, L), "x": x}

def linear1d_qae_uniformpaper_solver(u0: Callable[[np.ndarray], np.ndarray],
                                     lam_of_kappa: Callable[[np.ndarray], np.ndarray],
                                     L: float, T: float, n: int, Nx: int,
                                     Nk: int, Knf: int, Kns: int,
                                     algo: str, eval_schedule: Any, shots: int,
                                     seed: Optional[int]) -> Dict[str, Any]:
    x = np.linspace(0.0, L, Nx, endpoint=False)
    U0 = np.fft.fft(u0(x).astype(complex))
    kappa = _kappa_1d(Nx, L); lam = lam_of_kappa(kappa)
    h = T/n
    uniq, inv = np.unique(lam, return_inverse=True)
    step_uniq = np.array([per_step_factor_uniformQAE_paperstyle(
                            Li, h, Nk=Nk, Knf=Knf, Kns=Kns,
                            algo=algo, eval_schedule=eval_schedule, shots=shots, seed=seed
                          ) for Li in uniq], dtype=complex)
    step = step_uniq[inv]
    U_T = U0*(step**n); U_ex = U0*np.exp(lam*T)
    u_num = np.fft.ifft(U_T).real; u_ex  = np.fft.ifft(U_ex).real
    return {"u_num": u_num, "u_exact": u_ex, "L2_rel": _pde_metric(u_num, u_ex, L), "x": x}

def linear1d_qae_cheboptimal_solver(u0: Callable[[np.ndarray], np.ndarray],
                                    lam_of_kappa: Callable[[np.ndarray], np.ndarray],
                                    L: float, T: float, n: int, Nx: int,
                                    Nk: int, Knf: int, Kns: int,
                                    algo: str, eval_schedule: Any, shots: int,
                                    seed: Optional[int]) -> Dict[str, Any]:
    """
    NEW: QAE_Cheb_Optimal — cubic-spline dense sampling + global normalization + 2 QAE calls total.
    """
    x = np.linspace(0.0, L, Nx, endpoint=False)
    U0 = np.fft.fft(u0(x).astype(complex))
    kappa = _kappa_1d(Nx, L); lam = lam_of_kappa(kappa)
    h = T/n
    uniq, inv = np.unique(lam, return_inverse=True)
    step_uniq = np.array([per_step_factor_chebQAE_optimal(
                            Li, h, Nk=Nk, Knf=Knf, Kns=Kns,
                            algo=algo, eval_schedule=eval_schedule, shots=shots, seed=seed
                          ) for Li in uniq], dtype=complex)
    step = step_uniq[inv]
    U_T = U0*(step**n); U_ex = U0*np.exp(lam*T)
    u_num = np.fft.ifft(U_T).real; u_ex  = np.fft.ifft(U_ex).real
    return {"u_num": u_num, "u_exact": u_ex, "L2_rel": _pde_metric(u_num, u_ex, L), "x": x}

def linear1d_cheb_spectral_solver(u0: Callable[[np.ndarray], np.ndarray],
                                  lam_of_kappa: Callable[[np.ndarray], np.ndarray],
                                  L: float, T: float, n: int, Nx: int, K: int) -> Dict[str, Any]:
    """
    Chebyshev time-spectral (collocation) stepper for a 1D periodic linear PDE u_t = L u,
    with Fourier symbol lam_of_kappa(κ). No QAE is used here.

    - u0:         callable returning initial profile on x-grid
    - lam_of_kappa: function mapping kappa array -> lambda(kappa)
    - L: domain length, Nx: number of spatial grid points (periodic)
    - T: final time, n: number of steps (only used as 'power' of per_step)
    - K: number of Chebyshev G-L time nodes for the per-step collocation
    """
    x = np.linspace(0.0, L, Nx, endpoint=False)
    U0 = np.fft.fft(u0(x).astype(complex))
    kappa = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    lam = lam_of_kappa(kappa)                 # shape (Nx,)
    h = T / n

    # Per-mode Chebyshev time-spectral per-step factor
    S = cheb_spectral_step_factor(lam, h, K)  # vectorized over lam
    U_T = U0 * (S ** n)
    U_ex = U0 * np.exp(lam * T)

    u_num = np.fft.ifft(U_T).real
    u_ex  = np.fft.ifft(U_ex).real

    dx = L / Nx
    L2_rel = float(np.sqrt(np.sum((u_num - u_ex) ** 2) * dx /
                           max(1e-16, np.sum(u_ex ** 2) * dx)))
    return {"u_num": u_num, "u_exact": u_ex, "L2_rel": L2_rel, "x": x}

def linear1d_qae_chebdevelop_solver(u0: Callable[[np.ndarray], np.ndarray],
                                    lam_of_kappa: Callable[[np.ndarray], np.ndarray],
                                    L: float, T: float, n: int, Nx: int,
                                    tol_rel: float, max_depth: int, Knf0: int, Kns0: int,
                                    algo: str, eval_schedule: Any, shots: int,
                                    seed: Optional[int]) -> Dict[str, Any]:
    x = np.linspace(0.0, L, Nx, endpoint=False)
    U0 = np.fft.fft(u0(x).astype(complex))
    kappa = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L/Nx)
    lam = lam_of_kappa(kappa)
    h = T / n
    uniq, inv = np.unique(lam, return_inverse=True)
    step_uniq = np.array([
        per_step_factor_chebQAE_develop(Li, h,
                                        tol_rel=tol_rel, max_depth=max_depth,
                                        Knf0=Knf0, Kns0=Kns0,
                                        algo=algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
        for Li in uniq
    ], dtype=complex)
    step = step_uniq[inv]
    U_T = U0 * (step ** n)
    U_ex = U0 * np.exp(lam * T)
    u_num = np.fft.ifft(U_T).real
    u_ex  = np.fft.ifft(U_ex).real
    dx = L / Nx
    L2_rel = float(np.sqrt(np.sum((u_num - u_ex)**2) * dx / max(1e-16, np.sum(u_ex**2) * dx)))
    return {"u_num": u_num, "u_exact": u_ex, "L2_rel": L2_rel, "x": x}


# ============================================================
# J. 문제별 래퍼
# ============================================================
def ode_qae(lam: complex, T: float, n: int, rule: str, K: int,
            algo: str, eval_schedule: Any, shots: int, seed: Optional[int]) -> Dict[str, Any]:
    a_T = propagate_scalar_mode_qae(lam, T, n, rule=rule, K=K, algo=algo,
                                    eval_schedule=eval_schedule, shots=shots, seed=seed)
    exact = np.exp(lam*T)
    return {"a_num": a_T, "a_exact": exact, "rel_err": float(abs(a_T-exact)/max(1e-16,abs(exact)))}

def ode_qae_chebpaper(lam: complex, T: float, n: int,
                      Nk: int, Knf: int, Kns: int,
                      algo: str, eval_schedule: Any, shots: int, seed: Optional[int]) -> Dict[str, Any]:
    h = T/n
    step = per_step_factor_chebQAE_paperstyle(lam, h, Nk, Knf, Kns, algo, eval_schedule, shots, seed)
    a_T = step**n; exact = np.exp(lam*T)
    return {"a_num": a_T, "a_exact": exact, "rel_err": float(abs(a_T-exact)/max(1e-16,abs(exact)))}

def ode_qae_uniformpaper(lam: complex, T: float, n: int,
                         Nk: int, Knf: int, Kns: int,
                         algo: str, eval_schedule: Any, shots: int, seed: Optional[int]) -> Dict[str, Any]:
    h = T/n
    step = per_step_factor_uniformQAE_paperstyle(lam, h, Nk, Knf, Kns, algo, eval_schedule, shots, seed)
    a_T = step**n; exact = np.exp(lam*T)
    return {"a_num": a_T, "a_exact": exact, "rel_err": float(abs(a_T-exact)/max(1e-16,abs(exact)))}

def ode_qae_cheboptimal(lam: complex, T: float, n: int,
                        Nk: int, Knf: int, Kns: int,
                        algo: str, eval_schedule: Any, shots: int, seed: Optional[int]) -> Dict[str, Any]:
    h = T/n
    step = per_step_factor_chebQAE_optimal(lam, h, Nk, Knf, Kns, algo, eval_schedule, shots, seed)
    a_T = step**n; exact = np.exp(lam*T)
    return {"a_num": a_T, "a_exact": exact, "rel_err": float(abs(a_T-exact)/max(1e-16,abs(exact)))}

def ode_cheb_spectral(lam: complex, T: float, n: int, K: int) -> Dict[str, Any]:
    a_T = propagate_scalar_mode_cheb(lam, T, n, K)
    exact = np.exp(lam*T)
    return {"a_num": a_T, "a_exact": exact, "rel_err": float(abs(a_T-exact)/max(1e-16,abs(exact)))}

def ode_qae_chebdevelop(lam: complex, T: float, n: int,
                        tol_rel: float, max_depth: int, Knf0: int, Kns0: int,
                        algo: str, eval_schedule: Any, shots: int, seed: Optional[int]) -> Dict[str, Any]:
    h = T/n
    step = per_step_factor_chebQAE_develop(lam, h, tol_rel=tol_rel, max_depth=max_depth,
                                           Knf0=Knf0, Kns0=Kns0,
                                           algo=algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
    a_T = step**n; exact = np.exp(lam*T)
    return {"a_num": a_T, "a_exact": exact, "rel_err": float(abs(a_T-exact)/max(1e-16,abs(exact)))}

def heat_qae(u0, D, L, T, n, Nx, rule, K, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2)
    return linear1d_qae_solver(u0, lam_fn, L, T, n, Nx, rule, K, algo, eval_schedule, shots, seed)

def heat_qae_chebpaper(u0, D, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2)
    return linear1d_qae_chebpaper_solver(u0, lam_fn, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed)

def heat_qae_uniformpaper(u0, D, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2)
    return linear1d_qae_uniformpaper_solver(u0, lam_fn, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed)

def heat_qae_cheboptimal(u0, D, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2)
    return linear1d_qae_cheboptimal_solver(u0, lam_fn, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed)

def heat_cheb_spectral(u0, D, L, T, n, Nx, K):
    lam_fn = lambda kappa: -D*(kappa**2)
    return linear1d_cheb_spectral_solver(u0, lam_fn, L, T, n, Nx, K)

def heat_qae_chebdevelop(u0, D, L, T, n, Nx, tol_rel, max_depth, Knf0, Kns0, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2)
    return linear1d_qae_chebdevelop_solver(u0, lam_fn, L, T, n, Nx, tol_rel, max_depth, Knf0, Kns0, algo, eval_schedule, shots, seed)

def convdiff_qae(u0, v, D, L, T, n, Nx, rule, K, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2) - 1j*v*kappa
    return linear1d_qae_solver(u0, lam_fn, L, T, n, Nx, rule, K, algo, eval_schedule, shots, seed)

def convdiff_qae_chebpaper(u0, v, D, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2) - 1j*v*kappa
    return linear1d_qae_chebpaper_solver(u0, lam_fn, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed)

def convdiff_qae_uniformpaper(u0, v, D, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2) - 1j*v*kappa
    return linear1d_qae_uniformpaper_solver(u0, lam_fn, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed)

def convdiff_qae_cheboptimal(u0, v, D, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2) - 1j*v*kappa
    return linear1d_qae_cheboptimal_solver(u0, lam_fn, L, T, n, Nx, Nk, Knf, Kns, algo, eval_schedule, shots, seed)

def convdiff_cheb_spectral(u0, v, D, L, T, n, Nx, K):
    lam_fn = lambda kappa: -D*(kappa**2) - 1j*v*kappa
    return linear1d_cheb_spectral_solver(u0, lam_fn, L, T, n, Nx, K)

def convdiff_qae_chebdevelop(u0, v, D, L, T, n, Nx,
                             tol_rel, max_depth, Knf0, Kns0,
                             algo, eval_schedule, shots, seed):
    lam_fn = lambda kappa: -D*(kappa**2) - 1j*v*kappa
    return linear1d_qae_chebdevelop_solver(u0, lam_fn, L, T, n, Nx,
                                           tol_rel, max_depth, Knf0, Kns0,
                                           algo, eval_schedule, shots, seed)


# ============================================================
# Seven-way harness (adds QAE-Cheb_Develop) → results_sevenway
# ============================================================
def run_sevenway_and_save(
    out_dir: str = "./results_sevenway",
    problems: Sequence[str] = ("ode", "heat1d", "convdiff1d"),
    T: float = 0.05, n: int = 40, L: float = 1.0, D: float = 0.02, v: float = 1.0,
    Nx: int = 512, lam_ode: complex = -0.5 - 3j,
    K_list: Sequence[int] = (9, 17, 33),
    shots_list: Sequence[int] = (2000, 4000),
    qae_algo: str = "IAE", eval_schedule: Any = 4,
    # paper/optimal 공통 파라미터
    Nk: int = 8, Knf: int = 10, Kns: int = 200,
    # develop(적응형) 파라미터
    tol_rel: float = 1e-3, max_depth: int = 4, Knf0: int = 10, Kns0: int = 200,
    seed: Optional[int] = None, verbose: bool = True
) -> Tuple[pd.DataFrame, str]:
    os.makedirs(out_dir, exist_ok=True)
    records = []
    def u0_sine(x): return np.sin(2.0*np.pi*x/L)

    shots_fixed = shots_list[0]
    K_fixed = K_list[len(K_list)//2]

    variants_Kscan = (
        "classical",
        "qae_uniform",
        "qae_cheb_paper",
        "qae_uniform_paper",
        "cheb_spectral",
        "qae_cheb_optimal",
        "qae_cheb_develop",   # NEW
    )
    variants_shots = (
        "qae_uniform",
        "qae_cheb_paper",
        "qae_uniform_paper",
        "qae_cheb_optimal",
        "qae_cheb_develop",   # NEW
    )

    total_jobs = len(problems)*len(K_list)*len(variants_Kscan) + len(problems)*len(shots_list)*len(variants_shots)
    job = 0
    if verbose:
        print(f"[INFO] Seven-way start → problems={list(problems)}, K_list={list(K_list)}, shots_list={list(shots_list)}")
        print(f"[INFO] qae_algo={qae_algo}, Nk={Nk}, Knf={Knf}, Kns={Kns} | dev: tol={tol_rel}, depth={max_depth}, Knf0={Knf0}, Kns0={Kns0}, total_jobs={total_jobs}")
        if _HAS_AER:
            try:
                backend = AerSimulator()
                backend.set_options(
                    max_parallel_threads=int(os.environ.get("QPDE_MAX_THREADS","64")),
                    max_parallel_experiments=1,
                    max_parallel_shots=1
                )
                print("[INFO] AerSimulator thread caps:",
                      backend.options.max_parallel_threads,
                      backend.options.max_parallel_experiments,
                      backend.options.max_parallel_shots)
            except Exception as e:
                print("[WARN] Aer setup failed:", type(e).__name__)
        report_threads("BEGIN run_sevenway_and_save")

    def solve_and_time(prob: str, variant: str, K: int, shots: int):
        t0 = time.time()
        if prob == "ode":
            if variant == "classical":
                out = ode_qae(lam_ode, T, n, rule="clenshaw_curtis", K=K,
                              algo="CLASSICAL", eval_schedule=eval_schedule, shots=0, seed=seed)
            elif variant == "qae_uniform":
                out = ode_qae(lam_ode, T, n, rule="uniform_trap", K=K,
                              algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_paper":
                out = ode_qae_chebpaper(lam_ode, T, n, Nk=Nk, Knf=Knf, Kns=Kns,
                                        algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_uniform_paper":
                out = ode_qae_uniformpaper(lam_ode, T, n, Nk=Nk, Knf=Knf, Kns=Kns,
                                           algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_optimal":
                out = ode_qae_cheboptimal(lam_ode, T, n, Nk=Nk, Knf=Knf, Kns=Kns,
                                          algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_develop":
                out = ode_qae_chebdevelop(lam_ode, T, n,
                                          tol_rel=tol_rel, max_depth=max_depth, Knf0=Knf0, Kns0=Kns0,
                                          algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "cheb_spectral":
                out = ode_cheb_spectral(lam_ode, T, n, K)
            else:
                raise ValueError
            metric = out["rel_err"]

        elif prob == "heat1d":
            if variant == "classical":
                out = heat_qae(u0_sine, D, L, T, n, Nx, rule="clenshaw_curtis", K=K,
                               algo="CLASSICAL", eval_schedule=eval_schedule, shots=0, seed=seed)
            elif variant == "qae_uniform":
                out = heat_qae(u0_sine, D, L, T, n, Nx, rule="uniform_trap", K=K,
                               algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_paper":
                out = heat_qae_chebpaper(u0_sine, D, L, T, n, Nx, Nk=Nk, Knf=Knf, Kns=Kns,
                                         algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_uniform_paper":
                out = heat_qae_uniformpaper(u0_sine, D, L, T, n, Nx, Nk=Nk, Knf=Knf, Kns=Kns,
                                            algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_optimal":
                out = heat_qae_cheboptimal(u0_sine, D, L, T, n, Nx, Nk=Nk, Knf=Knf, Kns=Kns,
                                           algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_develop":
                out = heat_qae_chebdevelop(u0_sine, D, L, T, n, Nx,
                                           tol_rel, max_depth, Knf0, Kns0,
                                           qae_algo, eval_schedule, shots, seed)
            elif variant == "cheb_spectral":
                out = heat_cheb_spectral(u0_sine, D, L, T, n, Nx, K)
            else:
                raise ValueError
            metric = out["L2_rel"]

        elif prob == "convdiff1d":
            if variant == "classical":
                out = convdiff_qae(u0_sine, v, D, L, T, n, Nx, rule="clenshaw_curtis", K=K,
                                   algo="CLASSICAL", eval_schedule=eval_schedule, shots=0, seed=seed)
            elif variant == "qae_uniform":
                out = convdiff_qae(u0_sine, v, D, L, T, n, Nx, rule="uniform_trap", K=K,
                                   algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_paper":
                out = convdiff_qae_chebpaper(u0_sine, v, D, L, T, n, Nx, Nk=Nk, Knf=Knf, Kns=Kns,
                                             algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_uniform_paper":
                out = convdiff_qae_uniformpaper(u0_sine, v, D, L, T, n, Nx, Nk=Nk, Knf=Knf, Kns=Kns,
                                                algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_optimal":
                out = convdiff_qae_cheboptimal(u0_sine, v, D, L, T, n, Nx, Nk=Nk, Knf=Knf, Kns=Kns,
                                               algo=qae_algo, eval_schedule=eval_schedule, shots=shots, seed=seed)
            elif variant == "qae_cheb_develop":
                out = convdiff_qae_chebdevelop(u0_sine, v, D, L, T, n, Nx,
                                               tol_rel, max_depth, Knf0, Kns0,
                                               qae_algo, eval_schedule, shots, seed)
            elif variant == "cheb_spectral":
                out = convdiff_cheb_spectral(u0_sine, v, D, L, T, n, Nx, K)
            else:
                raise ValueError
            metric = out["L2_rel"]

        else:
            raise ValueError(f"Unknown problem {prob}")

        rt = time.time() - t0
        return metric, rt

    # --- K-scan ---
    for prob in problems:
        for K in K_list:
            for variant, shots in [
                ("classical", 0),
                ("qae_uniform", shots_fixed),
                ("qae_cheb_paper", shots_fixed),
                ("qae_uniform_paper", shots_fixed),
                ("cheb_spectral", 0),
                ("qae_cheb_optimal", shots_fixed),
                ("qae_cheb_develop", shots_fixed),   # NEW
            ]:
                metric, rt = solve_and_time(prob, variant, K, shots)
                records.append({"problem": prob, "variant": variant, "K_nodes": K, "shots": shots,
                                "metric": float(metric), "runtime_sec": rt})
                job += 1
                if verbose:
                    print(f"[{job:4d}/{total_jobs}] K-scan | {prob:<10} | {variant:<18} | K={K:<3} shots={shots:<5} "
                          f"metric={metric:.3e} time={rt:.2f}s")
                    if job % 10 == 0:
                        report_threads(f"K-scan checkpoint #{job}")

    # --- shots-scan @ K_fixed ---
    for prob in problems:
        for shots in shots_list:
            for variant in ("qae_uniform", "qae_cheb_paper", "qae_uniform_paper", "qae_cheb_optimal", "qae_cheb_develop"):
                metric, rt = solve_and_time(prob, variant, K_fixed, shots)
                records.append({"problem": prob, "variant": f"{variant}_shots", "K_nodes": K_fixed, "shots": shots,
                                "metric": float(metric), "runtime_sec": rt})
                job += 1
                if verbose:
                    print(f"[{job:4d}/{total_jobs}] shots-scan | {prob:<10} | {variant:<18} | K={K_fixed:<3} shots={shots:<5} "
                          f"metric={metric:.3e} time={rt:.2f}s")
                    if job % 10 == 0:
                        report_threads(f"shots-scan checkpoint #{job}")

    # Save
    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "sevenway_results.csv")
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"[INFO] Saved CSV → {csv_path}")
        report_threads("END run_sevenway_and_save")

    # --- Plots (error: log-scale)
    for prob in problems:
        sub = df[df["problem"] == prob]
        if sub.empty: continue

        # Error vs K  (❌ cheb_spectral 제외, y=log)
        fig = plt.figure()
        labels = [
            ("classical",         "Classical (CC)"),
            ("qae_uniform",       "QAE-Uniform (trap)"),
            ("qae_cheb_paper",    "QAE-Cheb (paper)"),
            ("qae_uniform_paper", "QAE-Uniform (paper)"),
            # ("cheb_spectral",   "Cheb-Spectral"),  # excluded
            ("qae_cheb_optimal",  "QAE-Cheb (optimal)"),
            ("qae_cheb_develop",  "QAE-Cheb (develop)"),
        ]
        for var, lab in labels:
            s2 = sub[(sub["variant"] == var)]
            if s2.empty: continue
            grp = s2.groupby("K_nodes", as_index=False)["metric"].mean()
            plt.plot(grp["K_nodes"], grp["metric"], marker="o", label=lab)
        plt.xlabel("K (time nodes)")
        plt.ylabel("error metric (lower is better)")
        plt.yscale("log")
        plt.title(f"{prob}: Error vs K  (QAE shots={shots_fixed}, algo={qae_algo})")
        plt.legend()
        fig_path = os.path.join(out_dir, f"{prob}_error_vs_K_sevenway.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        if verbose: print(f"[INFO] Saved plot → {fig_path}")

        # Runtime vs K  (❌ cheb_spectral 제외, y=log)
        fig = plt.figure()
        for var, lab in labels:  # same labels (no cheb_spectral)
            s2 = sub[(sub["variant"] == var)]
            if s2.empty: continue
            grp = s2.groupby("K_nodes", as_index=False)["runtime_sec"].mean()
            plt.plot(grp["K_nodes"], grp["runtime_sec"], marker="o", label=lab)
        plt.xlabel("K (time nodes)")
        plt.ylabel("runtime (sec)")
        plt.yscale("log")  # ▶▶ runtime도 로그 스케일
        plt.title(f"{prob}: Runtime vs K  (QAE shots={shots_fixed}, algo={qae_algo})")
        plt.legend()
        fig_path = os.path.join(out_dir, f"{prob}_runtime_vs_K_sevenway.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        if verbose: print(f"[INFO] Saved plot → {fig_path}")

        # Error vs shots @ K_fixed (QAE-only, y=log)
        sfix = sub[(sub["K_nodes"] == K_fixed)]
        if not sfix.empty:
            fig = plt.figure()
            for var, lab in [
                ("qae_uniform_shots","QAE-Uniform"),
                ("qae_cheb_paper_shots","QAE-Cheb (paper)"),
                ("qae_uniform_paper_shots","QAE-Uniform (paper)"),
                ("qae_cheb_optimal_shots","QAE-Cheb (optimal)"),
                ("qae_cheb_develop_shots","QAE-Cheb (develop)")
            ]:
                s2 = sfix[(sfix["variant"] == var)]
                if s2.empty: continue
                grp = s2.groupby("shots", as_index=False)["metric"].mean()
                plt.plot(grp["shots"], grp["metric"], marker="o", label=lab)
            # 기준선: ❌ cheb_spectral 제외, ✅ classical만 유지
            ref = sfix[(sfix["variant"] == "classical")]
            if not ref.empty:
                ref_val = ref.groupby("K_nodes")["metric"].mean().iloc[0]
                plt.axhline(ref_val, linestyle="--", linewidth=1.0, label="Classical@Kfixed")
            plt.xlabel("shots"); plt.ylabel("error metric")
            plt.yscale("log")
            plt.title(f"{prob}: Error vs shots (K={K_fixed}, algo={qae_algo})")
            plt.legend()
            fig_path = os.path.join(out_dir, f"{prob}_error_vs_shots_Kfixed.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
            if verbose: print(f"[INFO] Saved plot → {fig_path}")

            # Runtime vs shots @ K_fixed  (❌ cheb_spectral 기준선 제거, y=log)
            fig = plt.figure()
            for var, lab in [
                ("qae_uniform_shots","QAE-Uniform"),
                ("qae_cheb_paper_shots","QAE-Cheb (paper)"),
                ("qae_uniform_paper_shots","QAE-Uniform (paper)"),
                ("qae_cheb_optimal_shots","QAE-Cheb (optimal)"),
                ("qae_cheb_develop_shots","QAE-Cheb (develop)")
            ]:
                s2 = sfix[(sfix["variant"] == var)]
                if s2.empty: continue
                grp = s2.groupby("shots", as_index=False)["runtime_sec"].mean()
                plt.plot(grp["shots"], grp["runtime_sec"], marker="o", label=lab)
            # 기준선: ❌ cheb_spectral 제외, ✅ classical만 유지
            ref = sfix[(sfix["variant"] == "classical")]
            if not ref.empty:
                ref_val = ref.groupby("K_nodes")["runtime_sec"].mean().iloc[0]
                plt.axhline(ref_val, linestyle="--", linewidth=1.0, label="Classical@Kfixed")
            plt.xlabel("shots"); plt.ylabel("runtime (sec)")
            plt.yscale("log")  # ▶▶ runtime도 로그 스케일
            plt.title(f"{prob}: Runtime vs shots (K={K_fixed}, algo={qae_algo})")
            plt.legend()
            fig_path = os.path.join(out_dir, f"{prob}_runtime_vs_shots_Kfixed.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
            if verbose: print(f"[INFO] Saved plot → {fig_path}")

    return df, out_dir


# ============================================================
# L. Demo main
# ============================================================
if __name__ == "__main__":
    # 공통 설정
    T = 0.05; n = 40; v = 1.0; D = 0.02; L = 1.0; Nx = 512
    K_list = (9, 17, 33)
    shots_list = (2000, 4000)
    lam_ode = -0.5 - 3j

    # paper/optimal 공통
    Nk, Knf, Kns = 8, 10, 200
    # develop(적응형)
    tol_rel, max_depth, Knf0, Kns0 = 1e-5, 4, 10, 200

    df, out_dir = run_sevenway_and_save(
        out_dir="./results_sevenway",
        problems=("ode", "heat1d", "convdiff1d"),
        T=T, n=n, L=L, D=D, v=v, Nx=Nx, lam_ode=lam_ode,
        K_list=K_list, shots_list=shots_list,
        qae_algo="IAE", eval_schedule=4,
        Nk=Nk, Knf=Knf, Kns=Kns,
        tol_rel=tol_rel, max_depth=max_depth, Knf0=Knf0, Kns0=Kns0,
        seed=123, verbose=True
    )
    print(f"\nSaved results to: {out_dir}")
    print(df.head())