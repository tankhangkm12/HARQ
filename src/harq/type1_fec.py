import numpy as np
from typing import List
from src.phy.utils import cw_error_prob_from_pb

__all__ = ["TypeIFEC"]

class TypeIFEC:
    """
    HARQ Type I (FEC block code, hard-decision) với chiến lược ARQ cắt tại M.
    - (n, k, t): mã khối, k/n là code rate.
    - L_bits: chiều dài gói thông tin (bit).
    """
    def __init__(self, ber_uncoded: np.ndarray, n: int, k: int, t: int, L_bits: int, Mmax: int, Rs: float):
        self.ber_uncoded = ber_uncoded
        self.n, self.k, self.t = int(n), int(k), int(t)
        self.L_bits = int(L_bits)
        self.Mmax = int(Mmax)
        self.Rs = float(Rs)

        self.N_cw = int(np.ceil(self.L_bits / self.k))                 # số codeword trong 1 packet
        self.P_cw_err = cw_error_prob_from_pb(self.ber_uncoded, n=self.n, t=self.t)
        self.per = 1.0 - (1.0 - self.P_cw_err) ** self.N_cw
        self.p_succ = 1.0 - self.per

        # E[min(N,M)] cho hình học cắt tại M
        p = np.clip(self.p_succ, 1e-12, 1 - 1e-12)
        q = 1.0 - p
        terms = [ (k+1) * (q**k) * p for k in range(self.Mmax - 1) ]
        sum_terms = np.sum(terms, axis=0) if terms else 0.0
        tail = (q ** (self.Mmax - 1)) * self.Mmax
        self.EN_trunc = sum_terms + tail

        self.rate = float(self.k) / float(self.n)                      # bpcu
        self.TH = self.rate / self.EN_trunc

        # Dùng lại công thức residual hình học theo PER mỗi lần truyền
        self.residual = q ** (self.Mmax)

        # Danh sách P_succ(k) (k=1..M)
        self.psucc_lists = [ (q**(kk-1)) * p for kk in range(1, self.Mmax + 1) ]

    def latency_seconds(self) -> np.ndarray:
        # Mỗi vòng truyền tiêu thụ N_use = N_cw * n symbols
        T_round = (self.N_cw * self.n / self.Rs) * np.ones_like(self.ber_uncoded)
        return self.EN_trunc * T_round

    def pmf_at_index(self, idx: int) -> np.ndarray:
        p = float(np.clip(self.p_succ[idx], 1e-12, 1 - 1e-12))
        q = 1.0 - p
        Pk = np.array([ (q**(k-1)) * p for k in range(1, self.Mmax) ], dtype=float)
        Pk = np.append(Pk, q**(self.Mmax-1))
        return Pk
