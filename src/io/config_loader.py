import json, csv, io
from typing import Dict, Any, Tuple

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# Các key hợp lệ và kiểu dữ liệu mong đợi
_ALLOWED_KEYS = {
    "snr_min": float, "snr_max": float, "snr_step": float,
    "Mmax": int, "L_bits": int, "Rs": float,
    "R1_type2": float, "alpha": float, "erasure_eps": float,
    "beta": float, "mcs_text": str,
    # Cho phép override mã FEC Type I qua file:
    "fec_n": int, "fec_k": int, "fec_t": int,
    # Tuỳ chọn thêm:
    "snr_focus": float, "latlog": bool,
}

def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k not in _ALLOWED_KEYS:
            continue
        typ = _ALLOWED_KEYS[k]
        try:
            if typ is int:
                out[k] = int(v)
            elif typ is float:
                out[k] = float(v)
            elif typ is bool:
                # chấp nhận cả "true"/"false" chuỗi
                if isinstance(v, str):
                    out[k] = v.strip().lower() in ("1", "true", "yes", "y", "on")
                else:
                    out[k] = bool(v)
            elif typ is str:
                out[k] = str(v)
        except Exception:
            # bỏ qua key sai kiểu
            pass
    return out

def load_json(content: bytes) -> Dict[str, Any]:
    data = json.loads(content.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON phải là object {key:value}.")
    return _coerce_types(data)

def load_yaml(content: bytes) -> Dict[str, Any]:
    if not _HAVE_YAML:
        raise RuntimeError("Thiếu thư viện PyYAML (pyyaml). Hãy cài đặt để đọc YAML.")
    data = yaml.safe_load(content.decode("utf-8"))  # type: ignore
    if not isinstance(data, dict):
        raise ValueError("YAML phải là mapping key: value.")
    return _coerce_types(data)

def load_csv(content: bytes) -> Dict[str, Any]:
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    d: Dict[str, Any] = {}
    # CSV dạng 2 cột: key,value (theo template)
    for row in reader:
        k = row.get("key")
        v = row.get("value")
        if k is None:
            vals = list(row.values())
            if len(vals) >= 2:
                k, v = vals[0], vals[1]
        if k is not None:
            d[str(k).strip()] = v
    return _coerce_types(d)

def parse_config_file(file_bytes: bytes, filename: str) -> Tuple[dict, str]:
    """
    Trả về (config_dict, fmt) với fmt ∈ {"json","yaml","csv"}
    """
    lower = filename.lower()
    if lower.endswith(".json"):
        return load_json(file_bytes), "json"
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        return load_yaml(file_bytes), "yaml"
    if lower.endswith(".csv"):
        return load_csv(file_bytes), "csv"
    raise ValueError("Định dạng không hỗ trợ. Hãy dùng .json / .yaml / .yml / .csv")
