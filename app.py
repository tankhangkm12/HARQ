# app.py – Mô phỏng liên kết HARQ/ARQ (Phiên bản tiếng Việt học thuật)
# -*- coding: utf-8 -*-
import os, sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# ======== Cấu hình đường dẫn module nội bộ ========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.phy.utils import db2lin, lin2db, Q_from_snr_bpsk, packet_error_from_bit_error
from src.harq.saw import SAWGeom
from src.harq.type1_fec import TypeIFEC
from src.harq.type2_ir import (
    build_ir_bler_list_from_thresholds,
    ir_expected_delay_from_thresholds,
    ir_throughput_from_bler_lists,
    ir_residual_after_M,
    ir_psucc_lists,
    ir_pmf_from_bler_lists,
)
from src.harq.aharq import AdaptiveHARQ
from src.presets import CODE_PRESETS, SCENARIO_PROFILES
from src.io.config_loader import parse_config_file

# ============= Giao diện Streamlit =============
st.set_page_config(page_title="HARQ", layout="wide")
st.title("SO SÁNH & ĐÁNH GIÁ HARQ/ARQ")
st.caption("Môi trường AWGN + BPSK. Các sơ đồ: ARQ, HARQ Type I (FEC), HARQ Type II (IR), aHARQ (AMC + IR).")

# ================= Thanh bên =================
with st.sidebar:
    st.header("Cấu hình mô phỏng / Hồ sơ mẫu")
    uploaded = st.file_uploader("Tải cấu hình (.json / .yaml / .csv)", type=["json","yaml","yml","csv"])
    imported_cfg = {}
    if uploaded is not None:
        try:
            imported_cfg, fmt = parse_config_file(uploaded.read(), uploaded.name)
            st.success(f"Đã nạp {uploaded.name} ({fmt.upper()})")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    profile = st.selectbox("Chọn hồ sơ mẫu", list(SCENARIO_PROFILES.keys()), index=0)
    cfg = SCENARIO_PROFILES.get(profile, {}).copy()
    if imported_cfg: cfg.update(imported_cfg)

    st.markdown("---")
    st.header("Thiết lập mô phỏng")

    chart = st.selectbox(
        "Chọn biểu đồ",
        [
            "Độ trễ trung bình",
            "Thông lượng",
            "Độ trễ thời gian",
            "Xác suất lỗi",
            "Xác suất thành công P_succ(k)",
            "Phân phối PMF P[N=k]",
            "Hàm tích lũy CDF P[N≤k]",
            "Độ lệch chuẩn",
            "Đường cong BLER^(m) và PER",
            "Đường cong FEC (BER/PER)"
        ]
    )

    snr_min = st.number_input("Eb/N0 nhỏ nhất (dB)", value=float(cfg.get("snr_min",-8)), step=0.25)
    snr_max = st.number_input("Eb/N0 lớn nhất (dB)", value=float(cfg.get("snr_max",12)), step=0.25)
    snr_step = st.number_input("Bước thay đổi (dB)", value=float(cfg.get("snr_step",0.25)), step=0.05)
    Mmax = st.number_input("Số lần truyền tối đa M", 1, 10, int(cfg.get("Mmax",4)), 1)
    L_bits = st.number_input("Độ dài gói L (bit)", 1, 8192, int(cfg.get("L_bits",256)), 1)

    st.subheader("Hiển thị sơ đồ")
    show_saw   = st.checkbox("ARQ", True)
    show_t1    = st.checkbox("HARQ Type I (FEC)", True)
    show_t2    = st.checkbox("HARQ Type II (IR)", True)
    show_aharq = st.checkbox("aHARQ (AMC + IR)", True)

    st.subheader("Tham số FEC (Type I)")
    preset_name = st.selectbox("Chọn mã (n,k,t)", list(CODE_PRESETS.keys()), 0)
    n, k, t = CODE_PRESETS[preset_name]

    st.subheader("Tham số IR / aHARQ")
    R1_type2 = st.number_input("Tốc độ ban đầu R₁ (bits/s/Hz)", value=float(cfg.get("R1_type2",1.25)), step=0.05)
    alpha = st.number_input("Hệ số logistic α", value=float(cfg.get("alpha",1.8)), step=0.1)
    erasure_eps = st.slider("Xác suất mất gói ε", 0.0, 0.5, float(cfg.get("erasure_eps",0.2)), 0.01)
    beta = st.slider("Hệ số an toàn AMC β", 0.5, 1.0, float(cfg.get("beta",0.9)), 0.01)
    mcs_text = st.text_input("Tập MCS (bits/s/Hz)", cfg.get("mcs_text","0.3 0.6 0.9 1.2 1.5 1.8"))

    st.subheader("Điểm tập trung")
    snr_focus = st.slider("Eb/N0 dùng cho PMF/CDF (dB)", snr_min, snr_max, (snr_min+snr_max)/2, 0.25)
    k_focus = st.number_input("Vòng k", 1, int(Mmax), 1)
    m_focus = st.number_input("m cho BLER^(m)", 1, int(Mmax), 1)

    Rs = st.number_input("Tốc độ ký hiệu Rs (ký hiệu/giây)", 1e3, 1e9, float(cfg.get("Rs",1e6)), 1e5, format="%.0f")
    show_latency_logy = st.checkbox("Dùng trục log cho độ trễ thời gian", False)

# ================= Tính toán =================
snr_db = np.arange(snr_min, snr_max+1e-9, snr_step)
ber_uncoded = Q_from_snr_bpsk(snr_db)
per_saw = packet_error_from_bit_error(ber_uncoded, int(L_bits))

# ============ ARQ ============
saw = SAWGeom(per_saw, int(Mmax), int(L_bits), float(Rs))
EN_saw, TH_saw, res_saw = saw.EN, saw.TH, saw.residual
psucc_saw, LAT_saw = saw.psucc_lists, saw.latency_seconds()

# ============ HARQ Type I ============
fec = TypeIFEC(ber_uncoded, n, k, t, int(L_bits), int(Mmax), float(Rs))
per_t1, EN_t1, TH_t1, res_t1 = fec.per, fec.EN_trunc, fec.TH, fec.residual
psucc_t1, LAT_t1 = fec.psucc_lists, fec.latency_seconds()

# ============ HARQ Type II (IR) ============
R1_vec = np.full_like(snr_db, float(R1_type2))
bler_t2 = build_ir_bler_list_from_thresholds(snr_db, R1_vec, float(alpha), int(Mmax))
EN_t2 = ir_expected_delay_from_thresholds(snr_db, R1_vec, float(alpha), int(Mmax), float(erasure_eps))
TH_t2 = ir_throughput_from_bler_lists(bler_t2, R1_vec, float(erasure_eps))
res_t2 = ir_residual_after_M(bler_t2, float(erasure_eps))
psucc_t2 = ir_psucc_lists(bler_t2, float(erasure_eps))
LAT_t2 = EN_t2 * ((int(L_bits)/np.maximum(R1_vec,1e-9))/Rs)

# ============ aHARQ (AMC + IR) ============
mcs_list = [float(x) for x in mcs_text.replace(",", " ").split() if x]
aharq = AdaptiveHARQ(snr_db, float(beta), mcs_list, float(alpha), int(Mmax),
                     float(erasure_eps), int(L_bits), float(Rs))
R1_ah, bler_ah = aharq.R1, aharq.bler_list
EN_ah, TH_ah, res_ah = aharq.EN, aharq.TH, aharq.residual
psucc_ah, LAT_ah = aharq.psucc_lists, aharq.latency_seconds()

# ============ Hàm trợ giúp PMF + Độ lệch chuẩn ============
def build_pmf_geom(p_succ, Mmax):
    from src.harq.saw import pmf_truncated_geometric
    grid = np.zeros((p_succ.size, Mmax))
    for i in range(p_succ.size):
        grid[i] = pmf_truncated_geometric(float(p_succ[i]), Mmax)
    return grid

def build_pmf_ir(bler_list, eps):
    grid = np.zeros((bler_list[0].size, len(bler_list)))
    for i in range(bler_list[0].size):
        grid[i] = ir_pmf_from_bler_lists(bler_list, eps, i)
    return grid

def std_from_pmf(grid, ks):
    En = (grid * ks).sum(1); En2 = (grid * (ks**2)).sum(1)
    return np.sqrt(np.maximum(En2 - En**2, 0))

ks = np.arange(1, int(Mmax) + 1, dtype=float)

pmf_saw = build_pmf_geom(saw.p_succ, int(Mmax))
pmf_t1  = build_pmf_geom(fec.p_succ, int(Mmax))
pmf_t2  = build_pmf_ir(bler_t2, float(erasure_eps))
pmf_ah  = build_pmf_ir(bler_ah, float(erasure_eps))

STD_saw = std_from_pmf(pmf_saw, ks)
STD_t1  = std_from_pmf(pmf_t1, ks)
STD_t2  = std_from_pmf(pmf_t2, ks)
STD_ah  = std_from_pmf(pmf_ah, ks)

idx_focus = int(np.clip(round((snr_focus - snr_min) / snr_step), 0, snr_db.size - 1))

def pmf_focus():
    from src.harq.saw import pmf_truncated_geometric
    return (pmf_truncated_geometric(float(saw.p_succ[idx_focus]), int(Mmax)),
            pmf_truncated_geometric(float(fec.p_succ[idx_focus]), int(Mmax)),
            ir_pmf_from_bler_lists(bler_t2, float(erasure_eps), idx_focus),
            ir_pmf_from_bler_lists(bler_ah, float(erasure_eps), idx_focus))

xlabel = "Eb/N0 (dB)"

# ===================== VẼ BIỂU ĐỒ =====================
if chart.startswith("Độ trễ trung bình"):
    fig, ax = plt.subplots(figsize=(10,5))
    if show_saw: ax.semilogy(snr_db, EN_saw, label="ARQ")
    if show_t1:  ax.semilogy(snr_db, EN_t1, label=f"Type I ({preset_name})")
    if show_t2:  ax.semilogy(snr_db, EN_t2, label="Type II (IR)")
    if show_aharq: ax.semilogy(snr_db, EN_ah, label="aHARQ (AMC + IR)")
    ax.set_xlabel(xlabel); ax.set_ylabel("Độ trễ trung bình (số lần truyền)")
    ax.grid(True, which="both", ls=":"); ax.set_title("Độ trễ trung bình")
    ax.legend(); st.pyplot(fig)

elif chart.startswith("Thông lượng"):
    fig, ax = plt.subplots(figsize=(10,5))
    if show_saw: ax.plot(snr_db, TH_saw, label="ARQ")
    if show_t1:  ax.plot(snr_db, TH_t1, label=f"Type I ({k}/{n})")
    if show_t2:  ax.plot(snr_db, TH_t2, label="Type II (IR)")
    if show_aharq: ax.plot(snr_db, TH_ah, label="aHARQ (AMC + IR)")
    ax.set_xlabel(xlabel); ax.set_ylabel("Thông lượng (bits/s/Hz)")
    ax.grid(True, ls=":"); ax.set_title("Thông lượng")
    ax.legend(); st.pyplot(fig)

elif chart.startswith("Độ trễ thời gian"):
    fig, ax = plt.subplots(figsize=(10,5))
    if show_saw: ax.plot(snr_db, LAT_saw*1e3, label="ARQ")
    if show_t1:  ax.plot(snr_db, LAT_t1*1e3, label="Type I")
    if show_t2:  ax.plot(snr_db, LAT_t2*1e3, label="Type II (IR)")
    if show_aharq: ax.plot(snr_db, LAT_ah*1e3, label="aHARQ")
    ax.set_xlabel(xlabel); ax.set_ylabel("Độ trễ thời gian (ms)")
    if show_latency_logy: ax.set_yscale("log")
    ax.grid(True, which="both", ls=":"); ax.set_title("Độ trễ thời gian")
    ax.legend(); st.pyplot(fig)

elif chart.startswith("Xác suất lỗi"):
    fig, ax = plt.subplots(figsize=(10,5))
    if show_saw: ax.semilogy(snr_db, res_saw, label="ARQ")
    if show_t1:  ax.semilogy(snr_db, res_t1, label="Type I")
    if show_t2:  ax.semilogy(snr_db, res_t2, label="Type II (IR)")
    if show_aharq: ax.semilogy(snr_db, res_ah, label="aHARQ")
    ax.set_xlabel(xlabel); ax.set_ylabel("Xác suất lỗi sau M lần")
    ax.grid(True, which="both", ls=":"); ax.set_title("Xác suất lỗi")
    ax.legend(); st.pyplot(fig)

elif chart.startswith("Xác suất thành công"):
    k_plot = int(k_focus)
    fig, ax = plt.subplots(figsize=(10,5))
    if show_saw: ax.plot(snr_db, psucc_saw[k_plot-1], label="ARQ")
    if show_t1:  ax.plot(snr_db, psucc_t1[k_plot-1], label="Type I")
    if show_t2:  ax.plot(snr_db, psucc_t2[k_plot-1], label="Type II")
    if show_aharq: ax.plot(snr_db, psucc_ah[k_plot-1], label="aHARQ")
    ax.set_xlabel(xlabel); ax.set_ylabel("Xác suất thành công P_succ(k)")
    ax.grid(True, ls=":"); ax.set_title(f"Xác suất thành công")
    ax.legend(); st.pyplot(fig)

elif chart.startswith("Phân phối PMF"):
    PMF_saw, PMF_t1, PMF_t2, PMF_ah = pmf_focus(); ks = np.arange(1, int(Mmax)+1)
    fig, ax = plt.subplots(figsize=(10,5))
    off = 0.18
    if show_saw: ax.stem(ks-1.5*off, PMF_saw, linefmt='C0-', markerfmt='C0o', basefmt=" ")
    if show_t1:  ax.stem(ks-0.5*off, PMF_t1, linefmt='C1-', markerfmt='C1o', basefmt=" ")
    if show_t2:  ax.stem(ks+0.5*off, PMF_t2, linefmt='C2-', markerfmt='C2o', basefmt=" ")
    if show_aharq: ax.stem(ks+1.5*off, PMF_ah, linefmt='C3-', markerfmt='C3o', basefmt=" ")
    ax.set_xlabel(f"Số lần truyền N tại Eb/N0 = {snr_focus:.2f} dB")
    ax.set_ylabel("PMF  P[N = k]"); ax.grid(True, ls=":"); ax.set_xticks(ks)
    ax.set_title("Phân phối xác suất")
    ax.legend(["ARQ","Type I","Type II","aHARQ"]); st.pyplot(fig)

elif chart.startswith("Hàm tích lũy"):
    PMF_saw, PMF_t1, PMF_t2, PMF_ah = pmf_focus(); ks = np.arange(1, int(Mmax)+1, dtype=float)
    fig, ax = plt.subplots(figsize=(10,5))
    if show_saw: ax.step(ks, np.cumsum(PMF_saw), where='mid', label="ARQ")
    if show_t1:  ax.step(ks, np.cumsum(PMF_t1),np.cumsum(PMF_t2),where='mid',label="Type II")
    if show_aharq: ax.step(ks,np.cumsum(PMF_ah),where='mid',label="aHARQ")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Xác suất tích lũy CDF P[N ≤ k]")
    ax.grid(True,ls=":")
    ax.set_title(f"Hàm tích lũy CDF tại Eb/N0 = {snr_focus:.2f} dB")
    ax.legend()
    st.pyplot(fig)

elif chart.startswith("Độ lệch chuẩn"):
    fig,ax=plt.subplots(figsize=(10,5))
    if show_saw:   ax.plot(snr_db,STD_saw,label="ARQ")
    if show_t1:    ax.plot(snr_db,STD_t1,label="Type I")
    if show_t2:    ax.plot(snr_db,STD_t2,label="Type II")
    if show_aharq: ax.plot(snr_db,STD_ah,label="aHARQ")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Độ lệch chuẩn của số lần truyền σN")
    ax.grid(True,ls=":")
    ax.set_title("Độ lệch chuẩn của số lần truyền")
    ax.legend()
    st.pyplot(fig)

elif chart.startswith("Đường cong BLER"):
    m_plot=int(m_focus)
    fig,ax=plt.subplots(figsize=(10,5))
    if show_t2:    ax.plot(snr_db,bler_t2[m_plot-1],label=f"Type II BLER^(m={m_plot})")
    if show_aharq: ax.plot(snr_db,bler_ah[m_plot-1],label=f"aHARQ BLER^(m={m_plot})")
    if show_saw:   ax.plot(snr_db,per_saw,'--',label="ARQ PER")
    if show_t1:    ax.plot(snr_db,per_t1,'--',label="Type I PER")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Xác suất lỗi khối (BLER/PER)")
    ax.set_yscale("log")
    ax.grid(True,which="both",ls=":")
    ax.set_title("Đường cong BLER và PER")
    ax.legend()
    st.pyplot(fig)

elif chart.startswith("Đường cong FEC"):
    fig,ax=plt.subplots(figsize=(10,5))
    ax.semilogy(snr_db,ber_uncoded,label="BER (BPSK chưa mã hóa)")
    ax.semilogy(snr_db,per_t1,label=f"PER (HARQ Type I - {preset_name})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Xác suất lỗi (log)")
    ax.grid(True,which="both",ls=":")
    ax.set_title("Đường cong FEC")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Chọn một biểu đồ trong danh sách để so sánh các sơ đồ ARQ / HARQ.")

st.markdown("---")
st.caption("© Đỗ Thanh Tân & Nguyễn Tấn Lộc — Mô phỏng liên kết HARQ/ARQ (Phiên bản tiếng Việt học thuật)") 
