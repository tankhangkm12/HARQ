# app.py
# -*- coding: utf-8 -*-
import os, sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# ========= đảm bảo import được module trong ./src =========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ====== import các module bên ngoài (class/utils bạn đã có) ======
from src.phy.utils import (
    db2lin, lin2db, Q_from_snr_bpsk, packet_error_from_bit_error
)
from src.harq.saw import SAWGeom
from src.harq.type1_fec import TypeIFEC  # dùng class TypeIFEC
from src.harq.type2_ir import (
    build_ir_bler_list_from_thresholds,
    ir_expected_delay_from_thresholds,
    ir_throughput_from_bler_lists,
    ir_residual_after_M,
    ir_psucc_lists,
    ir_pmf_from_bler_lists,
)
from src.harq.aharq import AdaptiveHARQ, choose_rate_amc
from src.presets import CODE_PRESETS, SCENARIO_PROFILES
from src.io.config_loader import parse_config_file

# ====== Packet Tracer 2-khung (mới) ======
from src.ui.anim_packet_tracer import (
    prepare_tracer_data, make_fig_for_step, texts_for_step
)

# ========================
# Streamlit UI / Top matter
# ========================
st.set_page_config(page_title="HARQ/ARQ Link-Level — Ứng dụng so sánh", layout="wide")
st.title("🔁 HARQ/ARQ Link-Level — Ứng dụng so sánh trực quan")
st.caption("AWGN + BPSK, ACK/NACK tức thời, không RTT/hàng đợi. Sơ đồ: SAW, Type I (FEC thật), Type II (IR), aHARQ (AMC+IR).")

# ================= Sidebar: Preset + Import + Controls =================
with st.sidebar:
    st.header("Hồ sơ thực tế (Preset) & Import")
    uploaded = st.file_uploader("Tải cấu hình (.json / .yaml / .yml / .csv)", type=["json", "yaml", "yml", "csv"])
    imported_cfg = {}
    if uploaded is not None:
        try:
            imported_cfg, import_fmt = parse_config_file(uploaded.read(), uploaded.name)
            st.success(f"Đã nạp cấu hình từ file **{uploaded.name}** ({import_fmt.upper()}).")
            st.caption("Các ô bên dưới đã được áp theo file. Bạn vẫn có thể chỉnh tiếp.")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    profile_name = st.selectbox("Chọn hồ sơ Preset", list(SCENARIO_PROFILES.keys()), index=0)
    active_profile = SCENARIO_PROFILES.get(profile_name, {}).copy()
    if imported_cfg: active_profile.update(imported_cfg)

    st.markdown("---")
    st.header("Thiết lập chung")
    chart = st.selectbox(
        "Chọn biểu đồ",
        [
            "Độ trễ theo SNR (trục y log)",
            "Thông lượng theo SNR",
            "PMF P[N=k] theo số lần truyền (tại SNR đã chọn)",
            "CDF P[N≤k] theo số lần truyền (tại SNR đã chọn)",
            "Xác suất thất bại sau M lần theo SNR",
            "Xác suất thành công tại vòng k theo SNR",
            "Độ lệch chuẩn số lần truyền theo SNR",
            "Đường cong BLER^(m) của IR/aHARQ và PER mỗi lần truyền",
            "Độ trễ thời gian theo SNR",
            "Đường cong FEC (BER/PER theo SNR)",
        ]
    )

    # SNR grid
    snr_min = st.number_input("SNR min (dB)", value=float(active_profile.get("snr_min", -8.0)), step=0.25, format="%.2f")
    snr_max = st.number_input("SNR max (dB)", value=float(active_profile.get("snr_max", 12.0)), step=0.25, format="%.2f")
    snr_step = st.number_input("SNR step (dB)", value=float(active_profile.get("snr_step", 0.25)), step=0.05, format="%.2f")

    # Common across schemes
    Mmax = st.number_input("Số lần truyền tối đa M", min_value=1, max_value=10, value=int(active_profile.get("Mmax", 4)), step=1)
    L_bits = st.number_input("Độ dài gói thông tin L (bit) [SAW/Type I/IR]", min_value=1, max_value=8192, value=int(active_profile.get("L_bits", 256)), step=1)

    st.subheader("Bật/tắt các sơ đồ")
    show_saw   = st.checkbox("SAW (ARQ)", value=True)
    show_t1    = st.checkbox("HARQ Type I (FEC block code)", value=True)
    show_t2    = st.checkbox("HARQ Type II (IR)", value=True)
    show_aharq = st.checkbox("aHARQ (AMC+IR)", value=True)

    st.subheader("Type I — FEC (hard-decision)")
    fec_override = all(k in active_profile for k in ("fec_n", "fec_k", "fec_t"))
    if fec_override:
        preset_options = ["Custom (từ file)"] + list(CODE_PRESETS.keys())
        preset_name = st.selectbox("Bộ mã (n,k,t)", preset_options, index=0)
        if preset_name.startswith("Custom"):
            n = int(active_profile["fec_n"]); k = int(active_profile["fec_k"]); t = int(active_profile["fec_t"])
        else:
            n, k, t = CODE_PRESETS[preset_name]
    else:
        preset_name = st.selectbox("Bộ mã (n,k,t)", list(CODE_PRESETS.keys()), index=0)
        n, k, t = CODE_PRESETS[preset_name]

    st.subheader("Type II — IR / aHARQ")
    R1_type2 = st.number_input("Tốc độ ban đầu R1 (bpcu)", value=float(active_profile.get("R1_type2", 1.25)), step=0.05, format="%.2f")
    alpha = st.number_input("Độ dốc logistic α", value=float(active_profile.get("alpha", 1.8)), step=0.1, format="%.2f")
    erasure_eps = st.slider("Xác suất mất gói vòng 1 ε", 0.0, 0.5, float(active_profile.get("erasure_eps", 0.20)), 0.01)
    beta = st.slider("Hệ số an toàn AMC β (≤1)", 0.5, 1.0, float(active_profile.get("beta", 0.90)), 0.01)
    mcs_text = st.text_input("MCS_set [bpcu] (phân tách bởi dấu cách/phẩy)", active_profile.get("mcs_text", "0.3 0.6 0.9 1.2 1.5 1.8"))

    st.subheader("Tiêu điểm phân tích")
    snr_focus = st.slider("SNR dùng cho PMF/CDF (dB)", snr_min, snr_max, float(active_profile.get("snr_focus", (snr_min + snr_max)/2)), 0.25)
    k_focus = st.number_input("Vòng k cho biểu đồ P_succ(k)", min_value=1, max_value=int(Mmax), value=1, step=1)
    m_focus = st.number_input("m cho BLER^(m) (IR/aHARQ)", min_value=1, max_value=int(Mmax), value=1, step=1)

    st.subheader("Thông số thời gian")
    Rs = st.number_input("Tốc độ ký hiệu Rs (symbols/second)", min_value=1e3, max_value=1e9, value=float(active_profile.get("Rs", 1e6)), step=1e5, format="%.0f")
    show_latency_logy = st.checkbox("Độ trễ thời gian: dùng trục y log", value=bool(active_profile.get("latlog", False)))

# ================= Build SNR axis =================
snr_db = np.arange(snr_min, snr_max + 1e-9, snr_step)
Npts = snr_db.size

# ================= Base uncoded BER/PER =================
ber_uncoded = Q_from_snr_bpsk(snr_db)
per_saw = packet_error_from_bit_error(ber_uncoded, int(L_bits))

# ================= SAW Model =================
saw = SAWGeom(per=per_saw, Mmax=int(Mmax), L_bits=int(L_bits), Rs=float(Rs))
EN_saw, TH_saw, residual_saw = saw.EN, saw.TH, saw.residual
psucc_saw_lists, LAT_saw = saw.psucc_lists, saw.latency_seconds()

# ================= Type I FEC Model =================
fec = TypeIFEC(ber_uncoded=ber_uncoded, n=n, k=k, t=t, L_bits=int(L_bits), Mmax=int(Mmax), Rs=float(Rs))
per_t1, p_succ_t1 = fec.per, fec.p_succ
EN_t1, TH_t1, residual_t1 = fec.EN_trunc, fec.TH, fec.residual
psucc_t1_lists, LAT_t1 = fec.psucc_lists, fec.latency_seconds()

# ================= Type II IR (fixed R1) =================
R1_vec = np.full_like(snr_db, float(R1_type2))
bler_t2 = build_ir_bler_list_from_thresholds(snr_db, R1_vec, float(alpha), int(Mmax))
EN_t2 = ir_expected_delay_from_thresholds(snr_db, R1_vec, float(alpha), int(Mmax), float(erasure_eps))
TH_t2 = ir_throughput_from_bler_lists(bler_t2, R1_vec, float(erasure_eps))
residual_t2 = ir_residual_after_M(bler_t2, float(erasure_eps))
psucc_t2_lists = ir_psucc_lists(bler_t2, float(erasure_eps))
LAT_t2 = EN_t2 * ((int(L_bits) / np.maximum(R1_vec, 1e-9)) / Rs)

# ================= aHARQ (AMC + IR) =================
try:
    mcs_list = [float(x) for x in mcs_text.replace(",", " ").split()] if mcs_text.strip() else [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
except Exception:
    mcs_list = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
aharq = AdaptiveHARQ(snr_db=snr_db, beta=float(beta), mcs_list=mcs_list, alpha=float(alpha), Mmax=int(Mmax),
                     erasure_eps=float(erasure_eps), L_bits=int(L_bits), Rs=float(Rs))
R1_aharq, bler_aharq = aharq.R1, aharq.bler_list
EN_aharq, TH_aharq, residual_ah = aharq.EN, aharq.TH, aharq.residual
psucc_ah_lists, LAT_aharq = aharq.psucc_lists, aharq.latency_seconds()

# ================= Helpers PMF/STD =================
def build_pmf_grid_geom(p_succ_arr: np.ndarray, Mmax: int) -> np.ndarray:
    from src.harq.saw import pmf_truncated_geometric
    grid = np.zeros((p_succ_arr.size, Mmax), dtype=float)
    for i in range(p_succ_arr.size):
        grid[i, :] = pmf_truncated_geometric(float(p_succ_arr[i]), Mmax)
    return grid

def build_pmf_grid_ir(bler_list, erasure_eps: float) -> np.ndarray:
    grid = np.zeros((bler_list[0].size, len(bler_list)), dtype=float)
    for i in range(bler_list[0].size):
        grid[i, :] = ir_pmf_from_bler_lists(bler_list, erasure_eps, i)
    return grid

def stddev_from_pmf_grid(pmf_grid: np.ndarray, ks: np.ndarray) -> np.ndarray:
    En  = (pmf_grid * ks).sum(axis=1); En2 = (pmf_grid * (ks**2)).sum(axis=1)
    var = np.maximum(En2 - En**2, 0.0); return np.sqrt(var)

ks_vec = np.arange(1, int(Mmax) + 1, dtype=float)
pmf_grid_saw = build_pmf_grid_geom(saw.p_succ, int(Mmax))
pmf_grid_t1  = build_pmf_grid_geom(fec.p_succ,  int(Mmax))
pmf_grid_t2  = build_pmf_grid_ir(bler_t2,      float(erasure_eps))
pmf_grid_ah  = build_pmf_grid_ir(bler_aharq,   float(erasure_eps))
STD_saw = stddev_from_pmf_grid(pmf_grid_saw, ks_vec)
STD_t1  = stddev_from_pmf_grid(pmf_grid_t1,  ks_vec)
STD_t2  = stddev_from_pmf_grid(pmf_grid_t2,  ks_vec)
STD_ah  = stddev_from_pmf_grid(pmf_grid_ah,  ks_vec)

idx_focus = int(np.clip(round((snr_focus - snr_min) / snr_step), 0, max(0, Npts - 1)))
def pmf_all_schemes_at_focus():
    PMF_saw = saw.pmf_at_index(idx_focus)
    PMF_t1  = fec.pmf_at_index(idx_focus)
    PMF_t2  = ir_pmf_from_bler_lists(bler_t2,   float(erasure_eps), idx_focus)
    PMF_ah  = ir_pmf_from_bler_lists(bler_aharq, float(erasure_eps), idx_focus)
    return PMF_saw, PMF_t1, PMF_t2, PMF_ah

# ================= Formula panel =================
def render_formulas(chart_name: str, enabled: dict):
    with st.expander("📘 Công thức áp dụng"):
        st.markdown("### Khối cơ bản")
        st.latex(r"\gamma = 10^{\mathrm{SNR_{dB}}/10},\quad Q(x)=\tfrac12\,\mathrm{erfc}\!\big(x/\sqrt{2}\big)")
        st.latex(r"\mathrm{BER}_{\text{BPSK}} = Q\!\big(\sqrt{2\gamma}\big),\quad \mathrm{PER} = 1 - (1-\mathrm{BER})^{L}")

        if enabled.get("t1", False):
            st.markdown("#### Type I (FEC thật; hard-decision)")
            st.latex(r"P_{\text{cw,err}} = 1 - \sum_{i=0}^{t}\binom{n}{i}p_b^i(1-p_b)^{n-i}")
            st.latex(r"\mathrm{PER}_{\text{pkt}} = 1 - \big(1 - P_{\text{cw,err}}\big)^{\lceil L/k\rceil}")

        if enabled.get("t2", False) or enabled.get("aharq", False):
            st.markdown("#### Type II / aHARQ (IR, logistic thresholds)")
            st.latex(r"\text{SNR}_{\text{th}}(m)=10\log_{10}\!\big(2^{R_1/m}-1\big)")
            st.latex(r"P_e^{(m)}=\frac{1}{1+\exp\{\alpha(\mathrm{SNR_{dB}}-\text{SNR}_{\text{th}}(m))\}}")
            st.latex(r"P_{\text{succ}}(k)=P_e^{(k-1)}-P_e^{(k)},\quad P_e^{(0)}=1")
            st.latex(r"P_{\text{residual}}=(1-\epsilon)\,P_e^{(M)}+\epsilon\,P_e^{(M-1)}")

        if enabled.get("aharq", False):
            st.markdown("#### AMC (cho aHARQ)")
            st.latex(r"R_{\text{target}}=\beta\log_2(1+\gamma),\quad R_1=\max\{r\in\text{MCS}\mid r\le R_{\text{target}}\}")

        st.markdown("### Đại lượng vẽ ở biểu đồ này")
        if chart_name.startswith("Độ trễ theo SNR"):
            st.latex(r"D=\mathbb{E}[N]")
            st.markdown("- SAW/Type I (hình học cắt tại \(M\)):")
            st.latex(r"\mathbb{E}[\min(N,M)]=\sum_{k=1}^{M-1}k(1-p)^{k-1}p+M(1-p)^{M-1}")
        elif chart_name.startswith("Thông lượng"):
            st.markdown("- SAW: \(T=1/\mathbb{E}[N]\) bpcu;  Type I: \(T=(k/n)/\mathbb{E}[N]\) bpcu;  IR/aHARQ:")
            st.latex(r"T=\sum_{k=1}^{M}P_{\text{succ}}(k)\cdot \frac{R_1}{k}")
        elif chart_name.startswith("PMF"):
            st.latex(r"\text{PMF: }P[N=k]")
        elif chart_name.startswith("CDF"):
            st.latex(r"\text{CDF: }F(k)=\sum_{i=1}^{k}P[N=i]")
        elif chart_name.startswith("Xác suất thất bại"):
            st.latex(r"P_{\text{residual}}=\Pr[N=M,\ \text{vẫn lỗi}]")
        elif chart_name.startswith("Xác suất thành công"):
            st.latex(r"P_{\text{succ}}(k)")
        elif chart_name.startswith("Độ lệch chuẩn"):
            st.latex(r"\sigma_N=\sqrt{\mathbb{E}[N^2]-\mathbb{E}[N]^2]")
        elif chart_name.startswith("Đường cong FEC"):
            st.latex(r"\text{Hiển thị } \mathrm{BER}_{\text{uncoded}},\ \mathrm{PER}_{\text{pkt}}^{\text{Type I}}")
        else:
            st.markdown("#### Quy đổi thời gian")
            st.latex(r"T_{\text{round}}=\frac{N_{\text{use}}}{R_s}")
            st.markdown("- SAW: \(N_{use}=L\) (BPSK 1 bpcu)")
            st.markdown("- Type I: \(N_{use}= \lceil L/k\rceil \cdot n\)")
            st.markdown("- IR/aHARQ: \(N_{use}= L/R_1\)")
            st.latex(r"\text{Latency}=\mathbb{E}[N]\cdot T_{\text{round}}")

# ================= Charts =================
if chart.startswith("Độ trễ theo SNR"):
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_saw:   ax.semilogy(snr_db, EN_saw,   label="SAW (ARQ)")
    if show_t1:    ax.semilogy(snr_db, EN_t1,    label=f"Type I — {preset_name}")
    if show_t2:    ax.semilogy(snr_db, EN_t2,    label=f"Type II (IR) — R1={R1_type2:.2f}, ε={erasure_eps:.2f}")
    if show_aharq: ax.semilogy(snr_db, EN_aharq, label=f"aHARQ (AMC+IR) — β={beta:.2f}")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Độ trễ trung bình (số lần truyền/gói)")
    ax.grid(True, which="both", ls=":", lw=0.6); ax.set_title("Độ trễ trung bình theo SNR")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("Thông lượng"):
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_saw:   ax.plot(snr_db, TH_saw,   label="SAW (rate = 1 bpcu)")
    if show_t1:    ax.plot(snr_db, TH_t1,    label=f"Type I (rate = {k}/{n} bpcu)")
    if show_t2:    ax.plot(snr_db, TH_t2,    label=f"Type II (IR) — R1={R1_type2:.2f}")
    if show_aharq: ax.plot(snr_db, TH_aharq, label="aHARQ (AMC)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Thông lượng (bpcu)")
    ax.grid(True, ls=":", lw=0.6); ax.set_title("Thông lượng theo SNR")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("PMF"):
    PMF_saw, PMF_t1, PMF_t2, PMF_ah = pmf_all_schemes_at_focus()
    ks = np.arange(1, int(Mmax) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    offset = 0.18
    if show_saw:
        ml, sl, bl = ax.stem(ks - 1.5*offset, PMF_saw, linefmt='C0-', markerfmt='C0o', basefmt=" ")
        plt.setp(sl, linewidth=1.5)
    if show_t1:
        ml, sl, bl = ax.stem(ks - 0.5*offset, PMF_t1,  linefmt='C1-', markerfmt='C1o', basefmt=" ")
        plt.setp(sl, linewidth=1.5)
    if show_t2:
        ml, sl, bl = ax.stem(ks + 0.5*offset, PMF_t2,  linefmt='C2-', markerfmt='C2o', basefmt=" ")
        plt.setp(sl, linewidth=1.5)
    if show_aharq:
        ml, sl, bl = ax.stem(ks + 1.5*offset, PMF_ah,  linefmt='C3-', markerfmt='C3o', basefmt=" ")
        plt.setp(sl, linewidth=1.5)
    ax.set_xlabel(f"Số lần truyền N   (SNR tiêu điểm = {snr_focus:.2f} dB)")
    ax.set_ylabel("PMF  P[N = k]")
    ax.set_xticks(ks); ax.set_ylim(0, 1.0); ax.grid(True, ls=":", lw=0.6)
    ax.set_title("Phân phối PMF của số lần truyền tại SNR đã chọn")
    labels = []
    if show_saw: labels.append("SAW")
    if show_t1:  labels.append("Type I")
    if show_t2:  labels.append("Type II")
    if show_aharq: labels.append("aHARQ")
    if labels: ax.legend(labels)
    st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("CDF"):
    PMF_saw, PMF_t1, PMF_t2, PMF_ah = pmf_all_schemes_at_focus()
    ks = np.arange(1, int(Mmax) + 1)
    CDF_saw = np.cumsum(PMF_saw); CDF_t1 = np.cumsum(PMF_t1)
    CDF_t2  = np.cumsum(PMF_t2);  CDF_ah  = np.cumsum(PMF_ah)
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_saw:   ax.step(ks, CDF_saw, where='mid', label="SAW")
    if show_t1:    ax.step(ks, CDF_t1,  where='mid', label="Type I")
    if show_t2:    ax.step(ks, CDF_t2,  where='mid', label="Type II")
    if show_aharq: ax.step(ks, CDF_ah,  where='mid', label="aHARQ")
    ax.set_xlabel(f"N (SNR tiêu điểm = {snr_focus:.2f} dB)")
    ax.set_ylabel("CDF  P[N ≤ k]"); ax.set_xticks(ks); ax.set_ylim(0, 1.0)
    ax.grid(True, ls=":", lw=0.6); ax.set_title("Hàm phân phối tích lũy CDF của số lần truyền tại SNR đã chọn")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("Xác suất thất bại"):
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_saw:   ax.semilogy(snr_db, residual_saw, label="SAW")
    if show_t1:    ax.semilogy(snr_db, residual_t1,  label="Type I")
    if show_t2:    ax.semilogy(snr_db, residual_t2,  label="Type II (IR)")
    if show_aharq: ax.semilogy(snr_db, residual_ah,  label="aHARQ")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel(f"Xác suất thất bại sau M = {int(Mmax)}")
    ax.grid(True, which="both", ls=":", lw=0.6); ax.set_title("Xác suất thất bại sau số lần truyền tối đa")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("Xác suất thành công"):
    k_plot = int(np.clip(int(k_focus), 1, int(Mmax)))
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_saw:   ax.plot(snr_db, psucc_saw_lists[k_plot-1], label=f"SAW: P_succ(k={k_plot})")
    if show_t1:    ax.plot(snr_db, psucc_t1_lists[k_plot-1], label=f"Type I: P_succ(k={k_plot})")
    if show_t2:    ax.plot(snr_db, psucc_t2_lists[k_plot-1], label=f"Type II: P_succ(k={k_plot})")
    if show_aharq: ax.plot(snr_db, psucc_ah_lists[k_plot-1], label=f"aHARQ: P_succ(k={k_plot})")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel(f"P_succ tại vòng k = {k_plot}")
    ax.grid(True, ls=":", lw=0.6); ax.set_title("Xác suất thành công tại một vòng truyền cụ thể")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("Độ lệch chuẩn"):
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_saw:   ax.plot(snr_db, STD_saw, label="SAW")
    if show_t1:    ax.plot(snr_db, STD_t1,  label="Type I")
    if show_t2:    ax.plot(snr_db, STD_t2,  label="Type II (IR)")
    if show_aharq: ax.plot(snr_db, STD_ah,  label="aHARQ")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Độ lệch chuẩn của số lần truyền N")
    ax.grid(True, ls=":", lw=0.6); ax.set_title("Độ dao động của số lần truyền theo SNR")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("Đường cong BLER"):
    m_plot = int(np.clip(int(m_focus), 1, int(Mmax)))
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_t2:    ax.plot(snr_db, bler_t2[m_plot-1],    label=f"Type II: BLER^(m={m_plot})")
    if show_aharq: ax.plot(snr_db, bler_aharq[m_plot-1], label=f"aHARQ: BLER^(m={m_plot})")
    if show_saw:   ax.plot(snr_db, per_saw, '--', label="SAW: PER mỗi lần truyền")
    if show_t1:    ax.plot(snr_db, per_t1,  '--', label=f"Type I: PER mỗi lần truyền")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Xác suất lỗi / BLER"); ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", lw=0.6); ax.set_title("Đường cong BLER^(m) của IR/aHARQ và PER mỗi lần truyền")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

elif chart.startswith("Đường cong FEC"):
    # Hiển thị: BER_uncoded và PER Type I (packet)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(snr_db, np.clip(ber_uncoded, 1e-18, 1), label="BER (uncoded) — BPSK AWGN")
    ax.semilogy(snr_db, per_t1, label=f"PER (Type I) — {preset_name}")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Xác suất lỗi (log)")
    ax.grid(True, which="both", ls=":", lw=0.6); ax.set_title("Đường cong FEC (BER/PER) theo SNR")
    ax.legend(); st.pyplot(fig); render_formulas(chart, {"t1": True})

else:
    # Latency (seconds) vs SNR
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_saw:   ax.plot(snr_db, LAT_saw*1e3,   label="SAW (ARQ)")
    if show_t1:    ax.plot(snr_db, LAT_t1*1e3,    label=f"Type I — {preset_name}")
    if show_t2:    ax.plot(snr_db, LAT_t2*1e3,    label=f"Type II (IR) — R1={R1_type2:.2f}, ε={erasure_eps:.2f}")
    if show_aharq: ax.plot(snr_db, LAT_aharq*1e3, label=f"aHARQ (AMC+IR) — β={beta:.2f}")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Độ trễ thời gian (ms)")
    if show_latency_logy: ax.set_yscale("log")
    ax.grid(True, which="both" if show_latency_logy else "major", ls=":", lw=0.6)
    ax.set_title("Độ trễ thời gian theo SNR")
    ax.legend(); st.pyplot(fig); render_formulas("Độ trễ thời gian theo SNR", {"t1": show_t1, "t2": show_t2, "aharq": show_aharq})

# ====================== Packet Tracer (2 khung: trái chạy, phải info) ======================
st.markdown("---")
st.subheader("📦 Packet Tracer (Cisco-style) — 1 khung chạy • 1 khung thông tin")

c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.0])
with c1:
    tracer_scheme = st.selectbox("Chọn sơ đồ", ["SAW", "Type I", "Type II", "aHARQ"], index=1)
with c2:
    tracer_snr = st.number_input("SNR cho tracer (dB)", value=float(snr_focus), step=0.25, format="%.2f")
with c3:
    tracer_n = st.number_input("n (demo cho Type I)", value=7, min_value=3, max_value=64, step=1)
with c4:
    tracer_k = st.number_input("k (demo cho Type I)", value=4, min_value=2, max_value=tracer_n-1, step=1)

cc1, cc2, cc3 = st.columns([1.2, 1.0, 1.0])
with cc1:
    tracer_step = st.slider("Bước hiển thị", 1, 6, 1)
with cc2:
    tracer_m = st.slider("m (IR/aHARQ)", 1, 3, 1, help="Chỉ dùng cho Type II/aHARQ")
with cc3:
    st.caption("1=Source, 2=Encoder, 3=BPSK, 4=AWGN, 5=Demod, 6=Decoder")

# Chuẩn bị dữ liệu theo scheme
if tracer_scheme == "SAW":
    data = prepare_tracer_data("SAW", L_bits=int(L_bits), snr_db=float(tracer_snr))
    title_suffix = "SAW"
elif tracer_scheme == "Type I":
    data = prepare_tracer_data("Type I", L_bits=int(L_bits), snr_db=float(tracer_snr),
                               n=int(tracer_n), k=int(tracer_k))
    title_suffix = f"Type I — (n,k)=({int(tracer_n)},{int(tracer_k)})"
elif tracer_scheme == "Type II":
    data = prepare_tracer_data("Type II", L_bits=int(L_bits), snr_db=float(tracer_snr),
                               R1=float(R1_type2), alpha=float(alpha))
    title_suffix = f"Type II — R1={R1_type2:.2f}, α={alpha:.2f}"
else:
    # aHARQ: lấy R1 AMC tại snr focus nếu có; fallback dùng R1_type2
    R1_sel = float(R1_aharq[idx_focus] if len(snr_db) > 0 else R1_type2)
    data = prepare_tracer_data("aHARQ", L_bits=int(L_bits), snr_db=float(tracer_snr),
                               R1=R1_sel, alpha=float(alpha))
    title_suffix = f"aHARQ — R1={R1_sel:.2f}, α={alpha:.2f}"

# Hai khung: trái (figure), phải (text)
left_col, right_col = st.columns([0.7, 0.3])

with left_col:
    fig_pt = make_fig_for_step(data, step=int(tracer_step), m_part=int(tracer_m), title_suffix=title_suffix)
    st.plotly_chart(fig_pt, use_container_width=True, theme="streamlit")

with right_col:
    left_text, right_text = texts_for_step(data, step=int(tracer_step), m_part=int(tracer_m))
    st.markdown(left_text)
    st.markdown("---")
    st.markdown(right_text)

# ================= Quick readouts =================
with st.expander("🧪 Thông tin nhanh"):
    st.markdown(f"Lưới SNR: {snr_min} → {snr_max} dB, bước {snr_step} dB ({Npts} điểm)")
    st.markdown(f"Type I code: (n,k,t) = ({n},{k},{t}),  số codeword/packet N_cw = {fec.N_cw}")
    st.markdown(f"Tốc độ ký hiệu Rs = {Rs:.0f} sym/s → 1 sym = {1/Rs*1e6:.2f} µs")
    st.markdown(f"Hồ sơ đang chọn: **{profile_name}**")
    st.markdown("aHARQ (AMC) — R1 mẫu (bpcu):")
    if Npts > 0:
        idx = np.linspace(0, max(0, Npts - 1), num=min(8, max(1, Npts)), dtype=int)
        st.dataframe({"SNR (dB)": snr_db[idx], "R1": R1_aharq[idx]})
