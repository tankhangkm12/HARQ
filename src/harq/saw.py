import numpy as np
from typing import List

__all__ = ["SAWGeom", "pmf_truncated_geometric", "psucc_geom_lists", "residual_geom"]

def pmf_truncated_geometric(p_succ_scalar: float, Mmax: int) -> np.ndarray:
    p = float(np.clip(p_succ_scalar, 1e-12, 1 - 1e-12))
    q = 1.0 - p
    Pk = np.array([(q ** (k - 1)) * p for k in range(1, Mmax)], dtype=float)
    Pk = np.append(Pk, q ** (Mmax - 1))
    return Pk

def psucc_geom_lists(p_succ: np.ndarray, Mmax: int) -> List[np.ndarray]:
    p = np.clip(p_succ, 1e-12, 1 - 1e-12)
    q = 1.0 - p
    return [(q ** (k - 1)) * p for k in range(1, Mmax + 1)]

def residual_geom(per: np.ndarray, Mmax: int) -> np.ndarray:
    q = np.clip(per, 0.0, 1.0)
    return q ** int(Mmax)

class SAWGeom:
    """
    SAW (ARQ) với phân phối hình học cắt tại M (Mmax).
    """
    def __init__(self, per: np.ndarray, Mmax: int, L_bits: int, Rs: float):
        self.per = per
        self.Mmax = int(Mmax)
        self.L_bits = int(L_bits)
        self.Rs = float(Rs)

        self.p_succ = 1.0 - self.per
        self.EN = 1.0 / np.clip(self.p_succ, 1e-12, 1.0)           # E[N] hình học (không cắt)
        self.TH = 1.0 / self.EN                                   # bpcu (BPSK 1 bpcu)
        self.residual = residual_geom(self.per, self.Mmax)
        self.psucc_lists = psucc_geom_lists(self.p_succ, self.Mmax)

    def expected_tx_truncated_geometric(self) -> np.ndarray:
        p = np.clip(self.p_succ, 1e-12, 1 - 1e-12)
        q = 1.0 - p
        terms = [ (k+1) * (q**k) * p for k in range(self.Mmax - 1) ]
        sum_terms = np.sum(terms, axis=0) if terms else 0.0
        tail = (q ** (self.Mmax - 1)) * self.Mmax
        return sum_terms + tail

    def latency_seconds(self) -> np.ndarray:
        EN_trunc = self.expected_tx_truncated_geometric()
        T_round = (self.L_bits / self.Rs) * np.ones_like(self.per)   # BPSK 1 bpcu
        return EN_trunc * T_round

    def pmf_at_index(self, idx: int) -> np.ndarray:
        return pmf_truncated_geometric(float(self.p_succ[idx]), self.Mmax)
