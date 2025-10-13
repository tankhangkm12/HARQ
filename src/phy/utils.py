import numpy as np
from math import erfc
from typing import List

__all__ = [
    "db2lin",
    "lin2db",
    "Q_from_snr_bpsk",
    "packet_error_from_bit_error",
    "cw_error_prob_from_pb",
]

def db2lin(x_db: np.ndarray) -> np.ndarray:
    return 10.0 ** (x_db / 10.0)

def lin2db(x_lin: np.ndarray) -> np.ndarray:
    x_lin = np.maximum(x_lin, 1e-30)
    return 10.0 * np.log10(x_lin)

def Q_from_snr_bpsk(snr_db: np.ndarray) -> np.ndarray:
    """BER BPSK trên AWGN: Q(sqrt(2*gamma)) với gamma là SNR tuyến tính."""
    snr_lin = db2lin(snr_db)
    return 0.5 * np.vectorize(erfc)(np.sqrt(snr_lin))

def packet_error_from_bit_error(ber: np.ndarray, L_bits: int) -> np.ndarray:
    """PER ~ 1 - (1 - BER)^L_bits."""
    return 1.0 - (1.0 - ber) ** int(L_bits)

def cw_error_prob_from_pb(pb: np.ndarray, n: int, t: int) -> np.ndarray:
    """
    Xác suất mã khối (codeword) lỗi (hard-decision) với khả năng sửa t lỗi.
    P_ok = sum_{i=0..t} C(n,i) p^i (1-p)^{n-i}; P_err = 1 - P_ok
    """
    from math import comb
    pb = np.clip(pb, 1e-18, 1 - 1e-18)
    i_vals = np.arange(0, t + 1)
    coeffs = np.array([comb(n, i) for i in i_vals], dtype=float)
    terms = np.array([
        coeffs[j] * (pb ** i_vals[j]) * ((1 - pb) ** (n - i_vals[j]))
        for j in range(len(i_vals))
    ])
    P_ok = np.sum(terms, axis=0)
    return 1.0 - np.clip(P_ok, 0.0, 1.0)
