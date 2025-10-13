import numpy as np
from typing import List
from src.phy.utils import lin2db

__all__ = [
    "logistic_bler_wrt_threshold",
    "build_ir_bler_list_from_thresholds",
    "ir_psucc_lists",
    "ir_pmf_from_bler_lists",
    "ir_expected_delay_from_thresholds",
    "ir_throughput_from_bler_lists",
    "ir_residual_after_M",
]

def logistic_bler_wrt_threshold(snr_db: np.ndarray, snr_th_db: np.ndarray, alpha: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(alpha * (snr_db - snr_th_db)))

def build_ir_bler_list_from_thresholds(
    snr_db: np.ndarray, R1: np.ndarray, alpha: float, Mmax: int
) -> List[np.ndarray]:
    bler_list = []
    for m in range(1, int(Mmax) + 1):
        snr_th_lin = np.maximum(2.0 ** (R1 / m) - 1.0, 1e-12)
        snr_th_db = lin2db(snr_th_lin)
        bler_list.append(np.clip(logistic_bler_wrt_threshold(snr_db, snr_th_db, alpha), 0.0, 1.0))
    return bler_list

def ir_psucc_lists(bler_list: List[np.ndarray], erasure_eps: float) -> List[np.ndarray]:
    M = len(bler_list)
    Pe_prev = np.ones_like(bler_list[0])
    Psucc_reg = []
    for m in range(M):
        Pe_m = bler_list[m]
        Psucc_reg.append(np.clip(Pe_prev - Pe_m, 0.0, 1.0))
        Pe_prev = Pe_m
    if erasure_eps <= 0.0:
        return Psucc_reg
    bler_shift = [np.ones_like(bler_list[0])] + bler_list[:-1]
    Pe_prev = np.ones_like(bler_list[0])
    Psucc_shift = []
    for m in range(M):
        Pe_m = bler_shift[m]
        Psucc_shift.append(np.clip(Pe_prev - Pe_m, 0.0, 1.0))
        Pe_prev = Pe_m
    return [(1.0 - erasure_eps) * Psucc_reg[m] + erasure_eps * Psucc_shift[m] for m in range(M)]

def ir_pmf_from_bler_lists(bler_list: List[np.ndarray], erasure_eps: float, snr_idx: int) -> np.ndarray:
    Psucc = ir_psucc_lists(bler_list, erasure_eps)
    M = len(Psucc)
    pmf = np.array([float(Psucc[m][snr_idx]) for m in range(M)], dtype=float)
    peM = float(bler_list[-1][snr_idx])
    peM_s = float(bler_list[-2][snr_idx]) if M >= 2 else 1.0
    pmf[-1] += (1.0 - erasure_eps) * peM + erasure_eps * peM_s
    return pmf

def ir_throughput_from_bler_lists(bler_list: List[np.ndarray], R1_arr: np.ndarray, erasure_eps: float) -> np.ndarray:
    ks = np.arange(1, len(bler_list) + 1, dtype=float)
    Psucc = ir_psucc_lists(bler_list, erasure_eps)
    return np.sum([Psucc[m] * (R1_arr / ks[m]) for m in range(len(bler_list))], axis=0)

def ir_expected_delay_from_thresholds(
    snr_db: np.ndarray, R1: np.ndarray, alpha: float, Mmax: int, erasure_eps: float
) -> np.ndarray:
    bler_list = build_ir_bler_list_from_thresholds(snr_db, R1, alpha, Mmax)
    M = len(bler_list)
    Pe_prev = np.ones_like(bler_list[0])
    Psucc = []
    for m in range(M):
        Pe_m = bler_list[m]
        Psucc.append(np.clip(Pe_prev - Pe_m, 0.0, 1.0))
        Pe_prev = Pe_m
    PeM = bler_list[-1]
    ks = np.arange(1, M + 1)
    D_reg = np.sum([ks[i] * Psucc[i] for i in range(M)], axis=0) + M * PeM
    if erasure_eps <= 0.0:
        return D_reg
    bler_shift = [np.ones_like(bler_list[0])] + bler_list[:-1]
    Pe_prev = np.ones_like(bler_list[0])
    Psucc_s = []
    for m in range(M):
        Pe_m = bler_shift[m]
        Psucc_s.append(np.clip(Pe_prev - Pe_m, 0.0, 1.0))
        Pe_prev = Pe_m
    PeM_s = bler_shift[-1]
    D_erase = np.sum([ks[i] * Psucc_s[i] for i in range(M)], axis=0) + M * PeM_s
    return (1.0 - erasure_eps) * D_reg + erasure_eps * D_erase

def ir_residual_after_M(bler_list: List[np.ndarray], erasure_eps: float) -> np.ndarray:
    peM = bler_list[-1]
    peM_s = bler_list[-2] if len(bler_list) >= 2 else np.ones_like(peM)
    return (1.0 - erasure_eps) * peM + erasure_eps * peM_s
