from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Block:
    name: str
    xywh: Tuple[float, float, float, float]
    hover: str

def _rounded_rect(x, y, w, h, radius=14, line_width=2, fill="rgba(255,255,255,0.0)"):
    r = radius
    path = (f"M {x+r},{y} L {x+w-r},{y} Q {x+w},{y} {x+w},{y+r} L {x+w},{y+h-r} "
            f"Q {x+w},{y+h} {x+w-r},{y+h} L {x+r},{y+h} Q {x},{y+h} {x},{y+h-r} "
            f"L {x},{y+r} Q {x},{y} {x+r},{y} Z")
    return dict(type="path", path=path, line=dict(width=line_width, color="rgba(30,30,30,1)"),
                fillcolor=fill, layer="below")

def _arrow(x0, y0, x1, y1, dash=False, width=2):
    return dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(width=width, color="rgba(30,30,30,1)", dash="dash" if dash else "solid"))

def db2lin(x_db): return 10.0 ** (np.asarray(x_db, float) / 10.0)

def ber_bpsk_awgn(snr_db):
    from math import erfc
    g = db2lin(snr_db)
    return float(0.5 * erfc(np.sqrt(g)))

def _base_pipeline_blocks(L_bits, Rs, snr_db, extra_hover: Dict[str, str]):
    row_y, h, gap = 0.0, 1.6, 1.2
    x = -5.8
    blocks: List[Block] = []

    ho_tx   = f"<b>TX</b><br>L_bits={L_bits}<br>Rs={Rs:,.0f} sym/s"
    ho_enc  = extra_hover.get("encoder", "<b>Encoder</b>")
    ho_bpsk = "<b>BPSK Mapper</b><br>1 bpcu"
    sigma2  = 1.0 / (2.0 * db2lin(snr_db))
    ho_awgn = f"<b>AWGN Channel</b><br>SNR={snr_db:.2f} dB<br>σ²≈{sigma2:.3e}"
    ho_dem  = f"<b>Demod (Hard)</b><br>BER≈{ber_bpsk_awgn(snr_db):.3e}"
    ho_dec  = extra_hover.get("decoder", "<b>Decoder</b>")

    def add(name, w, hover):
        nonlocal x
        blocks.append(Block(name, (x, row_y - h/2, w, h), hover)); x += w + gap

    add("TX",            2.4, ho_tx)
    add("FEC Encoder",   3.2, ho_enc)
    add("BPSK",          2.3, ho_bpsk)
    add("AWGN Channel",  3.5, ho_awgn)
    add("Demod",         2.3, ho_dem)
    add("Decoder",       3.2, ho_dec)

    shapes = []
    for b in blocks:
        x0, y0, w0, h0 = b.xywh
        shapes.append(_rounded_rect(x0, y0, w0, h0))
    arrow_y = 0.0
    for i in range(len(blocks)-1):
        x0 = blocks[i].xywh[0] + blocks[i].xywh[2]
        x1 = blocks[i+1].xywh[0]
        shapes.append(_arrow(x0+0.06, arrow_y, x1-0.06, arrow_y))
    xL = blocks[0].xywh[0]; xR = blocks[-1].xywh[0] + blocks[-1].xywh[2]
    fb_y = arrow_y - 1.05
    shapes.append(_arrow(xR-0.2, fb_y, xL+0.2, fb_y, dash=True))

    hover_traces: List[go.Scatter] = []
    for b in blocks:
        x0, y0, w0, h0 = b.xywh
        xs = [x0, x0+w0, x0+w0, x0, x0]
        ys = [y0, y0,   y0+h0, y0+h0, y0]
        hover_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(width=0),
            fill="toself", fillcolor="rgba(0,0,0,0)", hoveron="fills",
            hoverinfo="text", text=b.hover, showlegend=False
        ))
    return blocks, shapes, hover_traces, xL, xR

def _packet_frames(x_start, x_end, fb_back, round_idx, Mmax, per_text,
                   pkt_y=0.0, steps_forward=56, steps_dwell=8, steps_back=12):
    x_fwd = np.linspace(x_start, x_end, steps_forward)
    x_ack = np.full(steps_dwell, x_end)
    x_ret = np.linspace(x_end, x_start, steps_back)
    frames: List[go.Frame] = []
    if not fb_back:
        traj = np.concatenate([x_fwd, x_ack]); tags = ["FWD"]*len(x_fwd) + ["ACK"]*len(x_ack)
    else:
        traj = np.concatenate([x_fwd, x_ret]); tags = ["FWD"]*len(x_fwd) + ["NACK"]*len(x_ret)
    for j, xx in enumerate(traj):
        badge = "ACK ✔" if tags[j] == "ACK" else ("NACK ✖" if tags[j] == "NACK" else "")
        frames.append(go.Frame(
            name=f"r{round_idx}_f{j}",
            data=[go.Scatter(
                x=[xx], y=[pkt_y], mode="markers", marker=dict(size=18, symbol="circle"),
                hoverinfo="text", text=f"<b>Packet</b><br>Round={round_idx}/{Mmax}{per_text}", showlegend=False
            )],
            layout=go.Layout(
                annotations=[
                    dict(x=x_start-0.2, y=pkt_y+1.2, xanchor="left", yanchor="bottom",
                         text=f"<b>Round {round_idx}/{Mmax}</b>", showarrow=False, font=dict(size=18)),
                    dict(x=x_end+0.2, y=pkt_y-1.2, xanchor="right", yanchor="top",
                         text=f"<b>{badge}</b>", showarrow=False, font=dict(size=18))
                ]
            )
        ))
    return frames

def _assemble_figure(blocks, shapes, hover_traces, frames, width, height, xL, xR, title):
    arrow_y = 0.0; fb_y = arrow_y - 1.05
    fig = go.Figure(
        data=[
            go.Scatter(x=[blocks[0].xywh[0] + 0.6], y=[arrow_y], mode="markers",
                       marker=dict(size=18, symbol="circle"), hoverinfo="skip", showlegend=False),
            *hover_traces,
            go.Scatter(
                x=[b.xywh[0] + b.xywh[2]/2 for b in blocks],
                y=[b.xywh[1] + b.xywh[3]/2 for b in blocks],
                mode="text", text=[f"<b>{b.name}</b>" for b in blocks],
                textposition="middle center", hoverinfo="skip", showlegend=False
            ),
            go.Scatter(x=[(xL+xR)/2], y=[fb_y - 0.25], mode="text",
                       text=["Feedback (ACK/NACK)"], hoverinfo="skip", showlegend=False)
        ],
        layout=go.Layout(
            width=width, height=height, template="plotly_white",
            title=dict(text=title, x=0.5),
            xaxis=dict(visible=False, range=[xL-0.8, xR+0.8]),
            yaxis=dict(visible=False, range=[-2.5, 2.5]),
            shapes=shapes, margin=dict(l=20, r=20, t=60, b=20),
            updatemenus=[dict(
                type="buttons", x=0.5, y=1.12, xanchor="center", yanchor="top",
                direction="left", showactive=False,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=90, redraw=True),
                                          fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                x=0.08, y=-0.06, len=0.84,
                pad=dict(t=10, b=10, l=0, r=0),   # ← sửa ở đây (thay padding -> pad)
                currentvalue=dict(prefix="Frame: ", visible=True),
                steps=[dict(method="animate",
                            args=[[fr.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                            label=str(i+1)) for i, fr in enumerate(frames)]
            )] if frames else []
        ),
        frames=frames
    )
    return fig

# ==== Public builders ====
def build_anim_saw(snr_db, L_bits, Rs, per_geom, Mmax, width=1100, height=520, seed=7):
    rng = np.random.RandomState(seed)
    extra_hover = {"encoder": "<b>Encoder</b><br>(No FEC – ARQ/SAW)",
                   "decoder": f"<b>Checker</b><br>PER (each try) ≈ {per_geom:.3e}"}
    blocks, shapes, hover_traces, xL, xR = _base_pipeline_blocks(L_bits, Rs, snr_db, extra_hover)
    x_start = blocks[0].xywh[0] + 0.6; x_end = blocks[-1].xywh[0] + blocks[-1].xywh[2] - 0.25
    p_succ = max(min(1.0 - per_geom, 1 - 1e-12), 1e-12)
    frames: List[go.Frame] = []
    for r in range(1, Mmax+1):
        succ = rng.rand() < p_succ
        frames.extend(_packet_frames(x_start, x_end, fb_back=(not succ), round_idx=r, Mmax=Mmax,
                                     per_text=f"<br>p_succ≈{p_succ:.3e}"))
        if succ: break
    title = f"SAW / ARQ — SNR={snr_db:.2f} dB, PER≈{per_geom:.2e}"
    return _assemble_figure(blocks, shapes, hover_traces, frames, width, height, xL, xR, title)

def build_anim_type1(snr_db, L_bits, Rs, n, k, t, per_packet, Mmax, width=1100, height=520, seed=7):
    rng = np.random.RandomState(seed)
    rate = k / n
    ho_enc = f"<b>FEC Encoder</b><br>(n,k,t)=({n},{k},{t})<br>Rate={rate:.3f}"
    ho_dec = f"<b>FEC Decoder</b><br>PER(packet)≈{per_packet:.3e}"
    blocks, shapes, hover_traces, xL, xR = _base_pipeline_blocks(L_bits, Rs, snr_db, {"encoder": ho_enc, "decoder": ho_dec})
    x_start = blocks[0].xywh[0] + 0.6; x_end = blocks[-1].xywh[0] + blocks[-1].xywh[2] - 0.25
    p_succ = max(min(1.0 - per_packet, 1 - 1e-12), 1e-12)
    frames: List[go.Frame] = []
    for r in range(1, Mmax+1):
        succ = rng.rand() < p_succ
        frames.extend(_packet_frames(x_start, x_end, fb_back=(not succ), round_idx=r, Mmax=Mmax,
                                     per_text=f"<br>PER≈{per_packet:.3e}, p_succ≈{p_succ:.3e}"))
        if succ: break
    title = f"Type I (hard-decision) — (n,k,t)=({n},{k},{t}), SNR={snr_db:.2f} dB"
    return _assemble_figure(blocks, shapes, hover_traces, frames, width, height, xL, xR, title)

def build_anim_ir(snr_db, L_bits, Rs, R1, alpha, bler_per_round, erasure_eps, Mmax,
                  width=1100, height=520, seed=7):
    rng = np.random.RandomState(seed)
    ho_enc = f"<b>IR Encoder</b><br>R1={R1:.2f} bpcu"
    ho_dec = f"<b>IR Decoder</b><br>α={alpha:.2f}, ε={erasure_eps:.2f}<br>BLER^(m) theo round"
    blocks, shapes, hover_traces, xL, xR = _base_pipeline_blocks(L_bits, Rs, snr_db, {"encoder": ho_enc, "decoder": ho_dec})
    x_start = blocks[0].xywh[0] + 0.6; x_end = blocks[-1].xywh[0] + blocks[-1].xywh[2] - 0.25

    pe = np.array(bler_per_round[:Mmax], dtype=float)
    pe_prev = 1.0; psucc = []
    for m in range(len(pe)):
        psucc.append(max(0.0, min(1.0, pe_prev - pe[m]))); pe_prev = pe[m]
    residual = (1.0 - erasure_eps) * pe[-1] + (erasure_eps * (pe[-2] if len(pe) >= 2 else 1.0))

    masses = np.array(psucc + [residual], float); masses /= max(np.sum(masses), 1e-12)
    u = rng.rand(); cum = np.cumsum(masses); chosen = int(np.searchsorted(cum, u)) + 1  # 1..M or M+1
    frames: List[go.Frame] = []
    if chosen <= Mmax:
        for r in range(1, chosen):
            frames.extend(_packet_frames(x_start, x_end, fb_back=True, round_idx=r, Mmax=Mmax,
                                         per_text=f"<br>BLER^(m={r})≈{pe[r-1]:.2e}, Psucc≈{psucc[r-1]:.2e}"))
        frames.extend(_packet_frames(x_start, x_end, fb_back=False, round_idx=chosen, Mmax=Mmax,
                                     per_text=f"<br>BLER^(m={chosen})≈{pe[chosen-1]:.2e}, Psucc≈{psucc[chosen-1]:.2e}"))
    else:
        for r in range(1, Mmax+1):
            frames.extend(_packet_frames(x_start, x_end, fb_back=True, round_idx=r, Mmax=Mmax,
                                         per_text=f"<br>BLER^(m={r})≈{pe[r-1]:.2e}, Psucc≈{psucc[r-1]:.2e}"))
    title = f"Type II (IR) — R1={R1:.2f}, α={alpha:.2f}, ε={erasure_eps:.2f}, SNR={snr_db:.2f} dB"
    return _assemble_figure(blocks, shapes, hover_traces, frames, width, height, xL, xR, title)

def build_anim_aharq(snr_db, L_bits, Rs, R1_sel, alpha, bler_per_round, erasure_eps, Mmax,
                     width=1100, height=520, seed=7):
    return build_anim_ir(snr_db, L_bits, Rs, R1_sel, alpha, bler_per_round, erasure_eps, Mmax,
                         width=width, height=height, seed=seed)
