from typing import Dict, Any, List

__all__ = ["CODE_PRESETS", "SCENARIO_PROFILES"]

# Mã FEC mẫu (hard-decision), thuận tiện cho demo/so sánh
CODE_PRESETS: Dict[str, tuple] = {
    "Hamming(7,4,t=1)": (7, 4, 1),
    "BCH(63,51,t=2)": (63, 51, 2),
    "BCH(127,106,t=3)": (127, 106, 3),
    "BCH(255,239,t=2)": (255, 239, 2),
}

# Hồ sơ “gần với thực tế” để cấu hình nhanh (minh hoạ)
SCENARIO_PROFILES: Dict[str, Dict[str, Any]] = {
    "Tùy chỉnh": {},
    "LTE-like (QPSK/16QAM) – lưu lượng trung bình": {
        "snr_min": -6.0, "snr_max": 16.0, "snr_step": 0.25,
        "Mmax": 4, "L_bits": 336, "Rs": 1.0e6,
        "R1_type2": 1.2, "alpha": 1.8, "erasure_eps": 0.15,
        "beta": 0.85, "mcs_text": "0.3 0.6 0.9 1.2 1.5 1.8 2.4",
    },
    "5G NR – eMBB (MCS phong phú)": {
        "snr_min": -8.0, "snr_max": 24.0, "snr_step": 0.25,
        "Mmax": 5, "L_bits": 1024, "Rs": 2.0e6,
        "R1_type2": 1.5, "alpha": 2.2, "erasure_eps": 0.10,
        "beta": 0.90, "mcs_text": "0.3 0.6 0.9 1.2 1.5 1.8 2.4 3.0 3.6",
    },
    "5G NR – URLLC (bảo thủ, độ trễ thấp)": {
        "snr_min": -4.0, "snr_max": 14.0, "snr_step": 0.25,
        "Mmax": 3, "L_bits": 192, "Rs": 1.5e6,
        "R1_type2": 0.9, "alpha": 2.0, "erasure_eps": 0.05,
        "beta": 0.80, "mcs_text": "0.3 0.6 0.9 1.2",
    },
}
from typing import Dict, Any, List

__all__ = ["CODE_PRESETS", "SCENARIO_PROFILES"]

# Mã FEC mẫu (hard-decision), thuận tiện cho demo/so sánh
CODE_PRESETS: Dict[str, tuple] = {
    "Hamming(7,4,t=1)": (7, 4, 1),
    "BCH(63,51,t=2)": (63, 51, 2),
    "BCH(127,106,t=3)": (127, 106, 3),
    "BCH(255,239,t=2)": (255, 239, 2),
}

# Hồ sơ “gần với thực tế” để cấu hình nhanh (minh hoạ)
SCENARIO_PROFILES: Dict[str, Dict[str, Any]] = {
    "Tùy chỉnh": {},
    "LTE-like (QPSK/16QAM) – lưu lượng trung bình": {
        "snr_min": -6.0, "snr_max": 16.0, "snr_step": 0.25,
        "Mmax": 4, "L_bits": 336, "Rs": 1.0e6,
        "R1_type2": 1.2, "alpha": 1.8, "erasure_eps": 0.15,
        "beta": 0.85, "mcs_text": "0.3 0.6 0.9 1.2 1.5 1.8 2.4",
    },
    "5G NR – eMBB (MCS phong phú)": {
        "snr_min": -8.0, "snr_max": 24.0, "snr_step": 0.25,
        "Mmax": 5, "L_bits": 1024, "Rs": 2.0e6,
        "R1_type2": 1.5, "alpha": 2.2, "erasure_eps": 0.10,
        "beta": 0.90, "mcs_text": "0.3 0.6 0.9 1.2 1.5 1.8 2.4 3.0 3.6",
    },
    "5G NR – URLLC (bảo thủ, độ trễ thấp)": {
        "snr_min": -4.0, "snr_max": 14.0, "snr_step": 0.25,
        "Mmax": 3, "L_bits": 192, "Rs": 1.5e6,
        "R1_type2": 0.9, "alpha": 2.0, "erasure_eps": 0.05,
        "beta": 0.80, "mcs_text": "0.3 0.6 0.9 1.2",
    },
}
