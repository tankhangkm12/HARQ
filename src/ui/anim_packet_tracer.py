# src/ui/packet_tracer_simple.py
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ---------- basic PHY helpers ----------
def db2lin(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)

def bpsk_map(bits: np.ndarray) -> np.ndarray:
    # 0 -> +1, 1 -> -1
    return 1.0 - 2.0 * bits

def awgn(sig: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    snr_lin = db2lin(snr_db)
    sigma2 = 1.0 / (2.0 * snr_lin)  # noise var for BPSK 1D
    noise = rng.normal(0.0, np.sqrt(sigma2), size=sig.shape)
    return sig + noise

def hard_decision(y: np.ndarray) -> np.ndarray:
    return (y < 0.0).astype(np.uint8)

# ---------- tiny linear (n,k) code for demo ----------
@dataclass
class LinearBlockCode:
    n: int
    k: int
    G: np.ndarray  # k x n over GF(2)
    codebook: Optional[np.ndarray] = None

    @staticmethod
    def from_random(n: int, k: int, seed: int = 7) -> "LinearBlockCode":
        rng = np.random.RandomState(seed)
        G = np.zeros((k, n), dtype=np.uint8)
        for i in range(k):
            G[i, i] = 1
        if n > k:
            G[:, k:] = (rng.rand(k, n - k) < 0.5).astype(np.uint8)
        return LinearBlockCode(n=n, k=k, G=G)

    def encode(self, u: np.ndarray) -> np.ndarray:
        return (u @ self.G) % 2

    def _ensure_codebook(self):
        if self.codebook is not None:
            return
        M = 1 << self.k
        cb = np.zeros((M, self.n), dtype=np.uint8)
        for m in range(M):
            u = np.array([(m >> i) & 1 for i in range(self.k)], dtype=np.uint8)
            cb[m] = self.encode(u)
        self.codebook = cb

    def brute_decode(self, r_bits: np.ndarray) -> Tuple[np.ndarray, int]:
        """Nearest-codeword brute force (demo) — phù hợp khi k <= 12."""
        self._ensure_codebook()
        dists = np.sum(np.abs(self.codebook - r_bits[None, :]), axis=1)
        idx = int(np.argmin(dists))
        dmin = int(dists[idx])
        u_hat = np.array([(idx >> i) & 1 for i in range(self.k)], dtype=np.uint8)
        return u_hat, dmin

# ---------- build demo data per scheme ----------
def prepare_tracer_data(
    scheme: str,
    L_bits: int,
    snr_db: float,
    n: int = 7, k: int = 4,
    R1: float = 1.2, alpha: float = 1.8,
    seed: int = 2024,
) -> Dict:
    rng = np.random.RandomState(seed)

    if scheme == "SAW":
        u = rng.randint(0, 2, size=min(L_bits, 32), dtype=np.uint8)
        c = u.copy()
        s = bpsk_map(c)
        y = awgn(s, snr_db, rng)
        r = hard_decision(y)
        return dict(kind="SAW", u=u, c=c, s=s, y=y, r=r, info=dict(L=L_bits, snr=snr_db))

    if scheme == "Type I":
        code = LinearBlockCode.from_random(n=n, k=k, seed=seed)
        u = rng.randint(0, 2, size=k, dtype=np.uint8)
        c = code.encode(u)
        s = bpsk_map(c)
        y = awgn(s, snr_db, rng)
        r = hard_decision(y)
        u_hat, dmin = code.brute_decode(r) if k <= 12 else (u.copy(), 0)
        return dict(kind="T1", u=u, c=c, s=s, y=y, r=r,
                    code=dict(n=n, k=k, dmin=dmin, u_hat=u_hat), info=dict(snr=snr_db))

    if scheme in ("Type II", "aHARQ"):
        # minh hoạ phát dần 3 mảnh
        m_parts = 3
        k_demo, n_demo = 8, 16
        code = LinearBlockCode.from_random(n=n_demo, k=k_demo, seed=seed)
        u = rng.randint(0, 2, size=k_demo, dtype=np.uint8)
        c_full = code.encode(u)
        splits = np.array_split(c_full, m_parts)
        s_parts = [bpsk_map(sp) for sp in splits]
        y_parts = [awgn(sp, snr_db, rng) for sp in s_parts]
        r_parts = [hard_decision(yp) for yp in y_parts]
        cum = []
        for m in range(m_parts):
            c_seen = np.concatenate(splits[:m+1])
            s_seen = np.concatenate(s_parts[:m+1])
            y_seen = np.concatenate(y_parts[:m+1])
            r_seen = np.concatenate(r_parts[:m+1])
            cum.append(dict(c=c_seen, s=s_seen, y=y_seen, r=r_seen))
        return dict(kind="IR", u=u, parts=cum, m_parts=m_parts,
                    info=dict(snr=snr_db, R1=R1, alpha=alpha))

    raise ValueError("scheme không hỗ trợ")

# ---------- figure & text for a step ----------
def make_fig_for_step(data: Dict, step: int, m_part: int = 1, title_suffix: str = "") -> go.Figure:
    """
    step: 1..6  (1=Source, 2=Encoder, 3=BPSK, 4=AWGN, 5=Demod, 6=Decoder)
    m_part: 1..3 (IR/aHARQ) — phần lũy tiến.
    """
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=("Waveform / Symbols", "Bits view"))

    # chọn dữ liệu theo scheme
    if data["kind"] == "SAW":
        u, c = data["u"], data["c"]; s, y, r = data["s"], data["y"], data["r"]
    elif data["kind"] == "T1":
        u, c = data["u"], data["c"]; s, y, r = data["s"], data["y"], data["r"]
    else:  # IR/aHARQ
        u = data["u"]
        part = data["parts"][int(np.clip(m_part, 1, data["m_parts"]))-1]
        c, s, y, r = part["c"], part["s"], part["y"], part["r"]

    # Khung trên: tín hiệu
    if step >= 3:
        fig.add_trace(go.Scatter(x=np.arange(len(s)), y=s, mode="lines+markers", name="s (BPSK)"), row=1, col=1)
    if step >= 4:
        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode="lines+markers", name="y (AWGN)"), row=1, col=1)

    # Khung dưới: bits
    if step >= 1:
        fig.add_trace(go.Scatter(x=np.arange(len(u)), y=u, mode="markers", marker=dict(symbol="square", size=12),
                                 name="u"), row=2, col=1)
    if step >= 2:
        fig.add_trace(go.Scatter(x=np.arange(len(c)), y=c, mode="markers", marker=dict(symbol="square", size=10),
                                 name="c"), row=2, col=1)
    if step >= 5:
        fig.add_trace(go.Scatter(x=np.arange(len(r)), y=r, mode="markers", marker=dict(symbol="square", size=9),
                                 name="r"), row=2, col=1)

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Index", row=1, col=1)
    fig.update_yaxes(title_text="Bit (0/1)", range=[-0.2, 1.2], row=2, col=1)
    fig.update_xaxes(title_text="Bit index", row=2, col=1)

    title = "Packet Tracer"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.update_layout(template="plotly_white", height=560,
                      title=dict(text=title, x=0.5),
                      legend=dict(orientation="h"))
    return fig

def texts_for_step(data: Dict, step: int, m_part: int = 1) -> Tuple[str, str]:
    def fmt_arr(arr, maxlen=64, precision=None):
        if precision is None:
            s = "".join(map(str, arr.tolist())) if arr.dtype.kind in "iu" else np.array2string(arr, separator='')
        else:
            s = np.array2string(arr, precision=precision)
        if len(s) > maxlen: s = s[:maxlen] + " …"
        return s

    if data["kind"] == "SAW":
        u, c, s, y, r = data["u"], data["c"], data["s"], data["y"], data["r"]
        L = data["info"]["L"]; snr = data["info"]["snr"]
        bit_err = int(np.sum(np.abs(r - c)))
        blocks = [
            ("Source (uncoded)", [f"L_bits={L}", f"Preview: {fmt_arr(u)}"]),
            ("No FEC", ["(Bypass encoder)"]),
            ("BPSK mapper", [f"s (±1): {fmt_arr(s)}"]),
            ("AWGN channel", [f"SNR={snr:.2f} dB", f"y: {fmt_arr(y, precision=2)}"]),
            ("Hard decision", [f"r: {fmt_arr(r)}", f"errors vs TX: {bit_err}"]),
            ("Checker / RX", ["Compare r vs TX"]),
        ]
    elif data["kind"] == "T1":
        u, c, s, y, r = data["u"], data["c"], data["s"], data["y"], data["r"]
        snr = data["info"]["snr"]
        n, k, dmin, u_hat = data["code"]["n"], data["code"]["k"], data["code"]["dmin"], data["code"]["u_hat"]
        bit_err = int(np.sum(np.abs(r - c))); u_err = int(np.sum(np.abs(u_hat - u)))
        blocks = [
            ("Source bits (k)", [f"k={k}", f"u: {fmt_arr(u)}"]),
            ("FEC Encoder (n,k)", [f"(n,k)=({n},{k})", f"c: {fmt_arr(c)}"]),
            ("BPSK mapper", [f"s: {fmt_arr(s)}"]),
            ("AWGN channel", [f"SNR={snr:.2f} dB", f"y: {fmt_arr(y, precision=2)}"]),
            ("Hard decision", [f"r: {fmt_arr(r)}", f"bit errors vs c: {bit_err}"]),
            ("FEC Decoder (brute)", [f"dmin={dmin}", f"u_hat: {fmt_arr(u_hat)}", f"errors vs u: {u_err}"]),
        ]
    else:  # IR/aHARQ
        u = data["u"]
        part = data["parts"][int(np.clip(m_part, 1, data["m_parts"]))-1]
        c, s, y, r = part["c"], part["s"], part["y"], part["r"]
        snr = data["info"]["snr"]; R1 = data["info"]["R1"]; alpha = data["info"]["alpha"]
        bit_err = int(np.sum(np.abs(r - c)))
        blocks = [
            ("Source (IR/aHARQ)", [f"u: {fmt_arr(u)}"]),
            ("IR Encoder", [f"R1={R1:.2f} bpcu", f"α={alpha:.2f}"]),
            (f"BPSK (m={m_part})", [f"|s|={s.size}", f"s(m): {fmt_arr(s)}"]),
            ("AWGN channel", [f"SNR={snr:.2f} dB", f"y(m): {fmt_arr(y, precision=2)}"]),
            ("Hard decision", [f"|r| till m={m_part}: {r.size}", f"errors vs sent: {bit_err}"]),
            ("IR Decoder (concept)", [f"mảnh nhận: {m_part}/{data['m_parts']}", "Giải mã lũy tiến (minh hoạ)"]),
        ]

    # step gating
    left = []
    for i, (title, lines) in enumerate(blocks, start=1):
        if i <= step:
            left.append(f"### {title}\n- " + "\n- ".join(lines))
    right = []
    if step < 6:
        pipeline = ["Encoder", "BPSK", "AWGN", "Demod", "Decoder"]
        right.append("### Tiếp theo\n- " + "\n- ".join(pipeline[step-1:]))
    else:
        right.append("### Hoàn tất\n- Gói đã qua toàn bộ pipeline")
    return "\n\n".join(left), "\n\n".join(right)
