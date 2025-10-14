# app2.py
# -*- coding: utf-8 -*-
"""
Streamlit: Năng lượng trung bình mỗi gói thành công theo SNR
- Chỉ vẽ duy nhất 1 biểu đồ năng lượng (Joule) với trục X là SNR (dB)
- E_avg = P0 * Ts * E[N]
- Tận dụng các mô-đun đã có trong thư mục ./src (SAW, Type I, Type II IR, aHARQ)
"""
import os, sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ========= đảm bảo import được module trong ./src =========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ====== import các module sẵn có trong dự án ======
from src.phy.utils import (
    Q_from_snr_bpsk,
    packet_error_from_bit_error,
)
from src.harq.saw import SAWGeom
from src.harq.type1_fec import TypeIFEC
from src.harq.type2_ir import (
    build_ir_bler_list_from_thresholds,
    ir_expected_delay_from_thresholds,
    ir_residual_after_M,
)
from src.harq.aharq import AdaptiveHARQ
from src.presets import CODE_PRESETS, SCENARIO_PROFILES

# ======================== UI ========================
st.set_page_config(page_title="Năng lượng trung bình / gói — HARQ/ARQ", layout="wide")
st.title("⚡ Năng lượng trung bình cho một gói thành công (J) theo SNR")
st.caption("Cố định công suất phát P₀ và thời gian 1 lượt Tₛ. Mô hình AWGN + BPSK, ACK/NACK tức thời.")

with st.sidebar:
    st.header("Thiết lập cơ bản")
    # Chọn hồ sơ (để lấy mặc định đẹp)
    profile_name = st.selectbox("Hồ sơ Preset", list(SCENARIO_PROFILES.keys()), index=0)
    active_profile = SCENARIO_PROFILES.get(profile_name, {}).copy()

    st.subheader("SNR grid (dB)")
    snr_min  = st.number_input("SNR min",  value=float(active_profile.get("snr_min", -8.0)), step=0.25, format="%.2f")
    snr_max  = st.number_input("SNR max",  value=float(active_profile.get("snr_max",  12.0)), step=0.25, format="%.2f")
    snr_step = st.number_input("SNR step", value=float(active_profile.get("snr_step",  0.25)), step=0.05, format="%.2f")

    st.subheader("Thông số gói & lặp")
    Mmax   = st.number_input("Số lần truyền tối đa M", min_value=1, max_value=10, value=int(active_profile.get("Mmax", 4)), step=1)
    L_bits = st.number_input("Độ dài gói thông tin L (bit)", min_value=1, max_value=8192, value=int(active_profile.get("L_bits", 256)), step=1)

    st.subheader("Các sơ đồ sẽ vẽ")
    show_saw   = st.checkbox("SAW (ARQ)", value=True)
    show_t1    = st.checkbox("HARQ Type I (FEC)", value=True)
    show_t2    = st.checkbox("HARQ Type II (IR)", value=True)
    show_aharq = st.checkbox("aHARQ (AMC+IR)", value=True)

    st.subheader("Type I — FEC")
    preset_name = st.selectbox("Bộ mã (n,k,t)", list(CODE_PRESETS.keys()), index=0)
    n, k, t = CODE_PRESETS[preset_name]

    st.subheader("Type II / aHARQ")
    R1_type2 = st.number_input("R1 (bpcu) cho IR", value=float(active_profile.get("R1_type2", 1.25)), step=0.05, format="%.2f")
    alpha    = st.number_input("Độ dốc logistic α", value=float(active_profile.get("alpha", 1.8)), step=0.1, format="%.2f")
    erasure_eps = st.slider("Xác suất mất gói vòng 1 ε", 0.0, 0.5, float(active_profile.get("erasure_eps", 0.20)), 0.01)
    beta     = st.slider("AMC β (≤1)", 0.5, 1.0, float(active_profile.get("beta", 0.90)), 0.01)
    mcs_text = st.text_input("MCS_set [bpcu] (cách/phẩy)", active_profile.get("mcs_text", "0.3 0.6 0.9 1.2 1.5 1.8"))

    st.subheader("Thời gian & năng lượng")
    Rs = st.number_input("Tốc độ ký hiệu Rs (sym/s)", min_value=1e3, max_value=1e9, value=float(active_profile.get("Rs", 1e6)), step=1e5, format="%.0f")
    P0 = st.number_input("Công suất phát P0 (W)", min_value=0.0, value=1.0, step=0.1, format="%.3f")
    Ts = st.number_input("Thời gian 1 lượt Ts (s)", min_value=0.0, value=1e-3, step=1e-4, format="%.6f")
    logy = st.checkbox("Trục Y log", value=False)

# ======================== Tính toán lõi ========================
# SNR trục X
snr_db = np.arange(snr_min, snr_max + 1e-9, snr_step)
Npts = snr_db.size

# BER uncoded + PER cho SAW
ber_uncoded = Q_from_snr_bpsk(snr_db)
per_saw = packet_error_from_bit_error(ber_uncoded, int(L_bits))

# SAW
saw = SAWGeom(per=per_saw, Mmax=int(Mmax), L_bits=int(L_bits), Rs=float(Rs))
EN_saw = saw.EN  # E[N]

# Type I
fec = TypeIFEC(ber_uncoded=ber_uncoded, n=n, k=k, t=t, L_bits=int(L_bits), Mmax=int(Mmax), Rs=float(Rs))
EN_t1  = fec.EN_trunc

# Type II IR (R1 cố định)
R1_vec = np.full_like(snr_db, float(R1_type2))
bler_t2 = build_ir_bler_list_from_thresholds(snr_db, R1_vec, float(alpha), int(Mmax))
EN_t2   = ir_expected_delay_from_thresholds(snr_db, R1_vec, float(alpha), int(Mmax), float(erasure_eps))

# aHARQ (AMC + IR)
try:
    mcs_list = [float(x) for x in mcs_text.replace(",", " ").split()] if mcs_text.strip() else [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
except Exception:
    mcs_list = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
aharq = AdaptiveHARQ(
    snr_db=snr_db, beta=float(beta), mcs_list=mcs_list, alpha=float(alpha),
    Mmax=int(Mmax), erasure_eps=float(erasure_eps), L_bits=int(L_bits), Rs=float(Rs)
)
EN_ah = aharq.EN

# ======================== Năng lượng trung bình ========================
# Theo yêu cầu: constant power P0, thời gian 1 lượt Ts
# Đặt E0 = P0 * Ts, khi đó E_avg = E0 * E[N]
E0 = float(P0) * float(Ts)
Eavg_saw = EN_saw * E0
Eavg_t1  = EN_t1  * E0
Eavg_t2  = EN_t2  * E0
Eavg_ah  = EN_ah  * E0

# ======================== Vẽ biểu đồ ========================
fig, ax = plt.subplots(figsize=(10, 5))
if show_saw:   ax.plot(snr_db, Eavg_saw, label="SAW (ARQ)")
if show_t1:    ax.plot(snr_db, Eavg_t1,  label=f"Type I — (n,k,t)=({n},{k},{t})")
if show_t2:    ax.plot(snr_db, Eavg_t2,  label=f"Type II (IR) — R1={R1_type2:.2f}, ε={erasure_eps:.2f}")
if show_aharq: ax.plot(snr_db, Eavg_ah,  label=f"aHARQ (AMC+IR) — β={beta:.2f}")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Năng lượng trung bình mỗi gói thành công  E_avg  (J)")
if logy: ax.set_yscale("log")
ax.grid(True, which="both" if logy else "major", ls=":", lw=0.6)
ax.set_title("⚡ Năng lượng trung bình cho một gói thành công theo SNR")
ax.legend()
st.pyplot(fig)

# ======================== Hộp công thức gọn ========================
with st.expander("📘 Công thức sử dụng"):
    st.markdown("- **X**: SNR (dB)")
    st.markdown("- **Y**: $E_{avg}$ — năng lượng để *một gói thành công* (Joules)")
    st.latex(r"E_0 = P_0 T_s,\qquad E_{\text{avg}} = P_0 T_s \cdot \mathbb{E}[N]")
    st.markdown("Trong đó $\\mathbb{E}[N]$ là số lượt truyền trung bình của sơ đồ (SAW/Type I/IR/aHARQ).")
