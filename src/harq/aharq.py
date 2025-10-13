import numpy as np
from typing import List
from src.phy.utils import db2lin
from src.harq.type2_ir import (
    build_ir_bler_list_from_thresholds,
    ir_expected_delay_from_thresholds,
    ir_throughput_from_bler_lists,
    ir_residual_after_M,
    ir_psucc_lists,
)

__all__ = ["choose_rate_amc", "AdaptiveHARQ"]

def choose_rate_amc(snr_db: np.ndarray, beta: float, mcs_set: List[float]) -> np.ndarray:
    cap = np.log2(1.0 + db2lin(snr_db))
    target = beta * cap
    mcs_sorted = np.array(sorted(mcs_set))
    R1 = np.full_like(snr_db, mcs_sorted[0], dtype=float)
    for i in range(snr_db.size):
        feas = mcs_sorted[mcs_sorted <= target[i] + 1e-12]
        R1[i] = feas[-1] if feas.size > 0 else mcs_sorted[0]
    return R1

class AdaptiveHARQ:
    """
    aHARQ = AMC + HARQ Type II(IR) logistic-thresholds.
    """
    def __init__(self, snr_db: np.ndarray, beta: float, mcs_list: List[float],
                 alpha: float, Mmax: int, erasure_eps: float, L_bits: int, Rs: float):
        self.snr_db = snr_db
        self.beta = float(beta)
        self.mcs_list = list(mcs_list)
        self.alpha = float(alpha)
        self.Mmax = int(Mmax)
        self.erasure_eps = float(erasure_eps)
        self.L_bits = int(L_bits)
        self.Rs = float(Rs)

        self.R1 = choose_rate_amc(self.snr_db, self.beta, self.mcs_list)
        self.bler_list = build_ir_bler_list_from_thresholds(self.snr_db, self.R1, self.alpha, self.Mmax)
        self.EN = ir_expected_delay_from_thresholds(self.snr_db, self.R1, self.alpha, self.Mmax, self.erasure_eps)
        self.TH = ir_throughput_from_bler_lists(self.bler_list, self.R1, self.erasure_eps)
        self.residual = ir_residual_after_M(self.bler_list, self.erasure_eps)
        self.psucc_lists = ir_psucc_lists(self.bler_list, self.erasure_eps)

    def latency_seconds(self) -> np.ndarray:
        T_round = (self.L_bits / np.maximum(self.R1, 1e-9)) / self.Rs
        return self.EN * T_round
