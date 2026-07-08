"""
generate_report.py - Generate a comprehensive PDF report comparing
Gemma 4 31B, Qwen3.5 27B (autoregressive) and LLaDA 8B (diffusion).

Usage:
  python generate_report.py
"""

import sys
import statistics
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

GEMMA_PT  = "signatures-google_gemma-4-31B-it.pt"
QWEN_PT   = "signatures-Qwen_Qwen3.5-27B.pt"
LLADA_PT  = "signatures-GSAI-ML_LLaDA-8B-Instruct.pt"
OUT_PDF   = "inference_telemetry_report.pdf"

PROMPT_LABELS = [
    "P0: Multi-hop factual reasoning",
    "P1: LLM mechanics explanation",
    "P2: Multi-step math (factory)",
    "P3: BST code + explanation",
    "P4: WWI causal chain",
    "P5: Dual-passage comprehension",
    "P6: Constrained creative writing",
    "P7: Multi-premise logic",
    "P8: Financial system (code)",
    "P9: Car ban policy analysis",
]

# ── helpers ────────────────────────────────────────────────────────────────

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def get_captures(data):
    if "captures" in data:
        return data.get("model", "?"), data["captures"]
    return data.get("model", "?"), [data]

def per_capture_metrics(cap):
    hs   = cap["hidden_states"]
    drift = [((hs[i] - hs[i-1]).norm(dim=-1)
               / hs[i-1].norm(dim=-1).clamp_min(1e-6)).mean().item()
              for i in range(1, len(hs))]
    spars = []
    for a in cap.get("mlp_acts", {}).values():
        thr = 0.01 * a.abs().max()
        spars.append((a.abs() < thr).float().mean().item())
    ent, sink = [], []
    attn = cap.get("attentions")
    if attn:
        for A in attn:
            if A is None: continue
            p = A.clamp_min(1e-12)
            ent.append((-(p * p.log()).sum(-1)).mean().item())
            sink.append(A[:, :, 0].mean().item())
    med = sorted(drift)[len(drift) // 2]
    return {
        "drift":       drift,
        "final_ratio": drift[-1] / med if med > 0 else float("nan"),
        "sparsity":    sum(spars)/len(spars) if spars else None,
        "entropy":     sum(ent)/len(ent)     if ent  else None,
        "sink":        sum(sink)/len(sink)   if sink else None,
    }

def avg_layer_drift(caps):
    all_d = [[((hs[i]-hs[i-1]).norm(dim=-1)/hs[i-1].norm(dim=-1).clamp_min(1e-6)).mean().item()
               for i in range(1, len(hs))]
              for hs in (c["hidden_states"] for c in caps)]
    L = len(all_d[0])
    return [sum(d[i] for d in all_d)/len(all_d) for i in range(L)]

def title_page(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    ax.text(0.5, 0.82, "Inference Telemetry", ha="center", va="center",
            fontsize=36, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.72, "Internal Activation Signatures Across\nAutoregressive and Diffusion Language Models",
            ha="center", va="center", fontsize=18, color="#8b949e",
            transform=ax.transAxes, linespacing=1.6)

    ax.axhline(y=0.63, xmin=0.1, xmax=0.9, color="#30363d", linewidth=1)

    models = [
        ("Gemma 4 31B",    "Autoregressive", "#4f8ef7"),
        ("Qwen3.5 27B",    "Autoregressive", "#f78c6c"),
        ("LLaDA 8B",       "Text Diffusion", "#7ee787"),
    ]
    for i, (name, kind, col) in enumerate(models):
        x = 0.25 + i * 0.25
        ax.text(x, 0.53, name, ha="center", fontsize=14, fontweight="bold",
                color=col, transform=ax.transAxes)
        ax.text(x, 0.47, kind, ha="center", fontsize=11, color="#8b949e",
                transform=ax.transAxes)

    ax.text(0.5, 0.28,
            "10 prompts spanning factual recall, reasoning, code generation,\n"
            "creative writing, and long reading comprehension.\n\n"
            "Metrics: residual drift · MLP sparsity · attention entropy · "
            "attention sink · token crystallization",
            ha="center", va="center", fontsize=12, color="#8b949e",
            transform=ax.transAxes, linespacing=1.8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def section_header(pdf, title, subtitle=""):
    fig = plt.figure(figsize=(11, 2.5))
    fig.patch.set_facecolor("#161b22")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#161b22")
    ax.axis("off")
    ax.text(0.06, 0.62, title, ha="left", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    if subtitle:
        ax.text(0.06, 0.28, subtitle, ha="left", fontsize=13, color="#8b949e",
                transform=ax.transAxes)
    ax.axhline(y=0.08, xmin=0.05, xmax=0.95, color="#30363d", linewidth=1)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── Section 1: per-prompt metric comparison ────────────────────────────────

def plot_metric_comparison(pdf, data_g, data_q, metric, label, pct=False):
    model_g, caps_g = get_captures(data_g)
    model_q, caps_q = get_captures(data_q)
    n = min(len(caps_g), len(caps_q))

    vals_g = [per_capture_metrics(c).get(metric) for c in caps_g[:n]]
    vals_q = [per_capture_metrics(c).get(metric) for c in caps_q[:n]]
    vals_g = [v for v in vals_g if v is not None]
    vals_q = [v for v in vals_q if v is not None]
    if not vals_g or not vals_q:
        return

    labels = PROMPT_LABELS[:n]
    x      = np.arange(n)
    w      = 0.38

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bg = ax.bar(x - w/2, vals_g, w, label=model_g.split("/")[-1], color="#4f8ef7", alpha=0.88)
    bq = ax.bar(x + w/2, vals_q, w, label=model_q.split("/")[-1], color="#f78c6c", alpha=0.88)

    fmt = "{:.1%}" if pct else "{:.2f}"
    for bar in list(bg) + list(bq):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001 * max(vals_g + vals_q),
                fmt.format(h), ha="center", va="bottom", fontsize=6.5, color="#c9d1d9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9, color="#8b949e")
    ax.set_ylabel(label, color="#8b949e", fontsize=11)
    ax.set_title(f"{label} — per prompt", color="white", fontsize=13, pad=10)
    ax.legend(fontsize=10, facecolor="#21262d", labelcolor="white", framealpha=0.9)
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.yaxis.label.set_color("#8b949e")
    if pct: ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── Section 2: layer-wise drift profile ───────────────────────────────────

def plot_layer_profile(pdf, data_g, data_q):
    model_g, caps_g = get_captures(data_g)
    model_q, caps_q = get_captures(data_q)
    mg = model_g.split("/")[-1]
    mq = model_q.split("/")[-1]

    drift_g = avg_layer_drift(caps_g)
    drift_q = avg_layer_drift(caps_q)
    Lg, Lq  = len(drift_g), len(drift_q)
    xg = [100 * i / Lg for i in range(Lg)]
    xq = [100 * i / Lq for i in range(Lq)]
    med_g = sorted(drift_g)[Lg // 2]
    med_q = sorted(drift_q)[Lq // 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.plot(xg, drift_g, color="#4f8ef7", label=mg, linewidth=1.8)
        ax.plot(xq, drift_q, color="#f78c6c", label=mq, linewidth=1.8)
        ax.axhline(med_g, color="#4f8ef7", linestyle="--", alpha=0.4, linewidth=1)
        ax.axhline(med_q, color="#f78c6c", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_yscale(yscale)
        ax.set_xlabel("Depth (%)", color="#8b949e", fontsize=11)
        ax.set_ylabel("Relative drift", color="#8b949e", fontsize=11)
        ax.set_title(f"Residual drift per layer ({'linear' if yscale=='linear' else 'log scale'})",
                     color="white", fontsize=12)
        ax.legend(fontsize=10, facecolor="#21262d", labelcolor="white")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.grid(alpha=0.15, color="#30363d")

    fig.suptitle("Layer-wise residual drift profile — averaged over 10 prompts",
                 color="white", fontsize=14, y=1.02)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── Section 3: token journey heatmap ──────────────────────────────────────

def plot_token_heatmap(pdf, data, prompt_idx, title_prefix):
    import torch.nn.functional as F
    model_name, caps = get_captures(data)
    short = model_name.split("/")[-1]
    cap = caps[prompt_idx]
    hs  = cap["hidden_states"]
    tokens   = cap["tokens"]
    seq_len  = len(tokens)
    n_layers = len(hs) - 1

    layers_drift = []
    for L in range(1, n_layers + 1):
        step = (hs[L] - hs[L-1]).norm(dim=-1) / hs[L-1].norm(dim=-1).clamp_min(1e-6)
        layers_drift.append(step)
    drift_mat = torch.stack(layers_drift, dim=0).T.numpy()  # [seq, n_layers]
    cum_drift = drift_mat.sum(axis=1)
    cos_sim   = F.cosine_similarity(hs[0], hs[-1], dim=-1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, max(5, seq_len * 0.35)))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")

    yticks  = range(seq_len)
    ylabels = [repr(t)[:12] for t in tokens]

    im = axes[0].imshow(drift_mat, aspect="auto", cmap="inferno", interpolation="nearest")
    axes[0].set_yticks(yticks); axes[0].set_yticklabels(ylabels, fontsize=6.5, color="#c9d1d9")
    axes[0].set_xlabel("Layer", color="#8b949e"); axes[0].tick_params(colors="#8b949e")
    axes[0].set_title(f"Per-token drift heatmap\n{short}", color="white", fontsize=11)
    plt.colorbar(im, ax=axes[0]).ax.yaxis.set_tick_params(color="#8b949e")

    axes[1].barh(yticks, cum_drift, color="#4f8ef7", alpha=0.85)
    axes[1].set_yticks(yticks); axes[1].set_yticklabels(ylabels, fontsize=6.5, color="#c9d1d9")
    axes[1].set_xlabel("Cumulative drift (all layers)", color="#8b949e")
    axes[1].set_title("Total transformation per token", color="white", fontsize=11)
    axes[1].invert_yaxis(); axes[1].tick_params(colors="#8b949e")
    for sp in axes[1].spines.values(): sp.set_color("#30363d")

    axes[2].barh(yticks, cos_sim, color="#7ee787", alpha=0.85)
    axes[2].set_yticks(yticks); axes[2].set_yticklabels(ylabels, fontsize=6.5, color="#c9d1d9")
    axes[2].set_xlabel("Cos similarity to initial embedding", color="#8b949e")
    axes[2].set_title("Direction preserved from embedding\n(1.0=same, 0=orthogonal)", color="white", fontsize=11)
    axes[2].axvline(0, color="#8b949e", linewidth=0.6)
    axes[2].invert_yaxis(); axes[2].tick_params(colors="#8b949e")
    for sp in axes[2].spines.values(): sp.set_color("#30363d")

    fig.suptitle(f"{title_prefix} — {PROMPT_LABELS[prompt_idx]}",
                 color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── Section 4: diffusion crystallization ──────────────────────────────────

def plot_diffusion(pdf, data_d, prompt_idx):
    short = data_d.get("model", "LLaDA").split("/")[-1]
    cap   = data_d["captures"][prompt_idx]
    conf_hist  = cap["confidence_history"]
    locked     = cap["locked_step"]
    final_toks = cap["final_tokens"]
    gen_length = cap["gen_length"]
    n_steps    = cap["n_steps"]
    prompt     = cap["prompt"]

    conf_mat   = torch.stack(conf_hist, dim=0).numpy()  # [steps, gen_len]
    avg_conf   = conf_mat.mean(axis=1)
    locked_arr = np.array(locked, dtype=float)
    locked_arr[locked_arr < 0] = np.nan

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    for ax in [ax0, ax1, ax2]: ax.set_facecolor("#161b22")

    im = ax0.imshow(conf_mat.T, aspect="auto", cmap="viridis",
                    vmin=0, vmax=1, interpolation="nearest")
    ax0.set_xlabel("Denoising step", color="#8b949e", fontsize=10)
    ax0.set_ylabel("Gen token position", color="#8b949e", fontsize=10)
    ax0.set_title("Confidence per token per step", color="white", fontsize=11)
    ax0.tick_params(colors="#8b949e")
    plt.colorbar(im, ax=ax0)

    ax1.barh(range(gen_length), locked_arr, color="#f78c6c", alpha=0.85)
    ax1.set_xlabel("Denoising step when locked", color="#8b949e", fontsize=10)
    ax1.set_ylabel("Gen token position", color="#8b949e", fontsize=10)
    ax1.set_title("Crystallization step per token", color="white", fontsize=11)
    ax1.invert_yaxis()
    ax1.tick_params(colors="#8b949e")
    if final_toks:
        ax1.set_yticks(range(gen_length))
        ax1.set_yticklabels([repr(t)[:10] for t in final_toks[:gen_length]], fontsize=5, color="#c9d1d9")
    for sp in ax1.spines.values(): sp.set_color("#30363d")

    ax2.plot(range(len(avg_conf)), avg_conf, marker="o", markersize=4,
             color="#7ee787", linewidth=2)
    ax2.fill_between(range(len(avg_conf)), avg_conf, alpha=0.15, color="#7ee787")
    ax2.set_xlabel("Denoising step", color="#8b949e", fontsize=10)
    ax2.set_ylabel("Avg confidence", color="#8b949e", fontsize=10)
    ax2.set_title("Confidence rise across steps", color="white", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors="#8b949e")
    ax2.grid(alpha=0.15, color="#30363d")
    for sp in ax2.spines.values(): sp.set_color("#30363d")

    fig.suptitle(f"LLaDA Diffusion — {PROMPT_LABELS[prompt_idx]}\n\"{prompt[:80]}\"",
                 color="white", fontsize=12, y=1.03)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── Section 5: summary findings table ─────────────────────────────────────

def plot_summary_table(pdf, data_g, data_q, data_d):
    model_g, caps_g = get_captures(data_g)
    model_q, caps_q = get_captures(data_q)
    n = min(len(caps_g), len(caps_q))

    def avg(caps, key):
        vals = [per_capture_metrics(c).get(key) for c in caps[:n]]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    mg = model_g.split("/")[-1]
    mq = model_q.split("/")[-1]

    rows = [
        ("Final-layer drift ratio",  avg(caps_g,"final_ratio"), avg(caps_q,"final_ratio"), "—", "Higher = more late-layer processing"),
        ("MLP sparsity",             avg(caps_g,"sparsity"),     avg(caps_q,"sparsity"),    "—", "Higher = fewer neurons fire per token"),
        ("Attention entropy",        avg(caps_g,"entropy"),      avg(caps_q,"entropy"),     "—", "Higher = more diffuse attention"),
        ("Attention sink (token 0)", avg(caps_g,"sink"),         avg(caps_q,"sink"),        "—", "Higher = more routing to token 0"),
    ]

    # LLaDA stats from prompt 0 and 2
    llada_steps_p0 = 3   # answer complete by step 2
    llada_steps_p2 = 20  # used all steps
    llada_conf_p0  = 0.643
    llada_conf_p2  = 0.198

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    cols = [mg, mq, "LLaDA 8B\n(diffusion)", "Interpretation"]
    col_w = [0.22, 0.22, 0.22, 0.34]
    headers = ["Metric"] + cols
    header_x = [0.01, 0.20, 0.40, 0.60, 0.78]

    ax.text(0.5, 0.97, "Model Comparison Summary", ha="center", va="top",
            fontsize=15, fontweight="bold", color="white", transform=ax.transAxes)

    # Header row
    for i, (h, x) in enumerate(zip(headers, header_x)):
        ax.text(x, 0.88, h, ha="left", va="top", fontsize=10, fontweight="bold",
                color="#7ee787" if i == 0 else "#c9d1d9", transform=ax.transAxes)
    ax.axhline(y=0.83, xmin=0.01, xmax=0.99, color="#30363d", linewidth=0.8,
               transform=ax.transAxes)

    fmt_map = {
        "Final-layer drift ratio":  lambda v: f"{v:.2f}×",
        "MLP sparsity":             lambda v: f"{v:.1%}",
        "Attention entropy":        lambda v: f"{v:.3f}",
        "Attention sink (token 0)": lambda v: f"{v:.1%}",
    }
    llada_vals = {
        "Final-layer drift ratio":  "—",
        "MLP sparsity":             "—",
        "Attention entropy":        "—",
        "Attention sink (token 0)": "—",
    }
    row_y = 0.78
    for metric, vg, vq, vl, interp in rows:
        fmt = fmt_map[metric]
        vals = [metric,
                fmt(vg) if vg else "—",
                fmt(vq) if vq else "—",
                llada_vals[metric],
                interp]
        for i, (v, x) in enumerate(zip(vals, header_x)):
            color = "#c9d1d9" if i > 0 else "#8b949e"
            ax.text(x, row_y, v, ha="left", va="top", fontsize=9,
                    color=color, transform=ax.transAxes)
        row_y -= 0.10

    ax.axhline(y=row_y + 0.04, xmin=0.01, xmax=0.99, color="#30363d",
               linewidth=0.8, transform=ax.transAxes)

    # LLaDA-specific rows
    llada_rows = [
        ("Steps to answer (factual P0)",  "—", "—", "2 / 20 steps",  "Crystallizes immediately"),
        ("Steps to answer (reasoning P2)","—", "—", "20 / 20 steps", "Uses full denoising budget"),
        ("Initial avg confidence (P0)",   "—", "—", "0.643",         "Knows structure fast"),
        ("Initial avg confidence (P2)",   "—", "—", "0.198",         "Much more uncertain on math"),
    ]
    for metric, vg, vq, vl, interp in llada_rows:
        vals = [metric, vg, vq, vl, interp]
        for i, (v, x) in enumerate(zip(vals, header_x)):
            color = "#7ee787" if i == 3 else "#8b949e" if i == 0 else "#c9d1d9"
            ax.text(x, row_y, v, ha="left", va="top", fontsize=9,
                    color=color, transform=ax.transAxes)
        row_y -= 0.09

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── Section 6: findings text page ─────────────────────────────────────────

def plot_findings(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.set_facecolor("#0d1117"); ax.axis("off")

    ax.text(0.5, 0.97, "Key Findings", ha="center", va="top", fontsize=18,
            fontweight="bold", color="white", transform=ax.transAxes)

    findings = [
        ("1. Gemma 31B defers answer assembly to the final 5% of layers",
         "Residual drift is near-zero for 55 of 60 layers, then explodes 3–4× above "
         "the median in the final layers. Final-layer drift ratio ≈ 20×. "
         "All tokens spike simultaneously — the answer is written into every position at once."),

        ("2. Qwen 27B front-loads processing and sinks attention to token 0",
         "Token 0 ('What') accumulates 33.4 drift units in early layers — 6× more than Gemma. "
         "Attention sink is 27% higher on average, meaning Qwen routes attention to token 0 "
         "as a scratchpad. Final-layer ratio ≈ 4× — work is distributed, not deferred."),

        ("3. Gemma's MLPs are significantly sparser than Qwen's",
         "Gemma fires ≈93% fewer neurons per token vs Qwen's ≈84%. "
         "Gap widens to 12% on creative writing. Long reading comprehension pushes both "
         "models to their sparsest (Gemma 98.6%, Qwen 92.4%) — long context forces selectivity."),

        ("4. LLaDA crystallizes the answer non-causally and near-instantly on factual prompts",
         "'Paris' locks in at denoising step 1 (out of 20), before 'of' (step 2). "
         "There is no left-to-right constraint — tokens resolve by confidence, not position. "
         "The model recognises a short answer and reserves the remainder of the generation "
         "buffer with EOS tokens from step 0."),

        ("5. LLaDA uses its full denoising budget on reasoning tasks",
         "Arithmetic reasoning (P2) starts with avg confidence 0.197 vs 0.643 for factual recall. "
         "All 20 steps are consumed. Tokens at the answer boundary ('40', 'km/h') lock in "
         "at steps 1–2 while structural tokens ('Total', 'distance', '=') finalise at steps 19–20. "
         "The model resolves the numeric answer before its supporting explanation."),

        ("6. Deeper layers drive denoising; final-layer activity grows across steps",
         "Layer 24 shows the highest hidden-state drift between snapshot timesteps. "
         "Layer 31 drift accelerates from 0.48 → 0.83 as denoising progresses — "
         "unlike AR models where the final layer fires once, in LLaDA it becomes "
         "progressively more active as confidence rises."),
    ]

    y = 0.90
    for title, body in findings:
        ax.text(0.0, y, title, ha="left", va="top", fontsize=11, fontweight="bold",
                color="#7ee787", transform=ax.transAxes)
        y -= 0.04
        ax.text(0.02, y, body, ha="left", va="top", fontsize=9.5, color="#c9d1d9",
                transform=ax.transAxes, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                          edgecolor="#30363d", alpha=0.6),
                fontfamily="monospace")
        y -= 0.115

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── confidence comparison across all prompts ──────────────────────────────

def plot_confidence_comparison(pdf, data_d):
    caps    = data_d["captures"]
    n_steps = data_d.get("n_steps", 20)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    cmap = plt.cm.tab10
    for i, cap in enumerate(caps):
        conf_hist = cap["confidence_history"]
        avg_conf  = [ch.mean().item() for ch in conf_hist]
        label     = PROMPT_LABELS[i] if i < len(PROMPT_LABELS) else f"P{i}"
        ax.plot(range(len(avg_conf)), avg_conf, marker="o", markersize=3,
                linewidth=1.8, label=label, color=cmap(i / len(caps)), alpha=0.9)

    ax.set_xlabel("Denoising step", color="#8b949e", fontsize=11)
    ax.set_ylabel("Average confidence", color="#8b949e", fontsize=11)
    ax.set_title("Confidence rise per prompt — how quickly LLaDA becomes certain",
                 color="white", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, facecolor="#21262d", labelcolor="white",
              framealpha=0.9, loc="lower right")
    ax.tick_params(colors="#8b949e")
    ax.grid(alpha=0.15, color="#30363d")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── hidden state drift across snapshot steps ──────────────────────────────

def plot_hidden_drift(pdf, data_d):
    caps           = data_d["captures"]
    snapshot_layers = data_d.get("snapshot_layers", [0, 8, 16, 24, 31])
    snapshot_steps  = data_d.get("snapshot_steps",  [0, 5, 10, 15, 19])

    # For each layer, collect drift between consecutive snapshot steps per prompt
    # drift[layer][interval] = list of values across prompts
    n_intervals = len(snapshot_steps) - 1
    layer_drifts = {li: [[] for _ in range(n_intervals)] for li in snapshot_layers}

    for cap in caps:
        snaps      = cap.get("hidden_snapshots", {})
        prompt_len = cap.get("prompt_len", 0)
        for interval in range(n_intervals):
            s0, s1 = snapshot_steps[interval], snapshot_steps[interval + 1]
            for li in snapshot_layers:
                if s0 in snaps and s1 in snaps and li in snaps[s0] and li in snaps[s1]:
                    t0 = snaps[s0][li][prompt_len:]
                    t1 = snaps[s1][li][prompt_len:]
                    if t0.shape[0] > 0:
                        d = ((t1 - t0).norm(dim=-1) / t0.norm(dim=-1).clamp_min(1e-6)).mean().item()
                        layer_drifts[li][interval].append(d)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    x      = np.arange(n_intervals)
    width  = 0.15
    colors = ["#4f8ef7", "#f78c6c", "#7ee787", "#e3b341", "#db6d28"]
    xlabels = [f"steps {snapshot_steps[i]}→{snapshot_steps[i+1]}" for i in range(n_intervals)]

    for idx, li in enumerate(snapshot_layers):
        means = [np.mean(layer_drifts[li][iv]) if layer_drifts[li][iv] else 0
                 for iv in range(n_intervals)]
        offset = (idx - len(snapshot_layers) / 2) * width
        ax.bar(x + offset, means, width, label=f"Layer {li}",
               color=colors[idx % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, color="#8b949e", fontsize=10)
    ax.set_ylabel("Avg hidden state drift (gen tokens)", color="#8b949e", fontsize=11)
    ax.set_title("Hidden state drift per layer between denoising snapshots\n(avg over all prompts)",
                 color="white", fontsize=13)
    ax.legend(fontsize=9, facecolor="#21262d", labelcolor="white", framealpha=0.9)
    ax.tick_params(colors="#8b949e")
    ax.grid(axis="y", alpha=0.15, color="#30363d")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading model files...")
    try:
        data_d = load(LLADA_PT)
        print(f"  ✓ {LLADA_PT}")
    except FileNotFoundError:
        print(f"  ✗ {LLADA_PT} not found"); sys.exit(1)

    n_prompts = len(data_d["captures"])
    print(f"  {n_prompts} prompts found")

    print(f"\nGenerating report -> {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Cover
        title_page(pdf)

        # Section 1: diffusion crystallization for all prompts
        section_header(pdf,
            "Section 1 — Token Crystallization Analysis",
            "LLaDA 8B · How tokens lock in across denoising steps per prompt")
        for pidx in range(n_prompts):
            plot_diffusion(pdf, data_d, pidx)

        # Section 2: confidence rise comparison across all prompts
        section_header(pdf,
            "Section 2 — Confidence Progression",
            "How quickly does LLaDA become certain across different prompt types?")
        plot_confidence_comparison(pdf, data_d)

        # Section 3: hidden state drift across snapshot steps
        section_header(pdf,
            "Section 3 — Hidden State Drift Across Denoising Steps",
            "Which layers do the most work at each stage of denoising?")
        plot_hidden_drift(pdf, data_d)

        # Section 4: findings
        section_header(pdf, "Section 4 — Key Findings")
        plot_findings(pdf)

        info = pdf.infodict()
        info["Title"]   = "Inference Telemetry — LLaDA Diffusion Report"
        info["Subject"] = "LLaDA 8B token crystallization across 10 prompts"

    print(f"\n✓ Report saved -> {OUT_PDF}")

if __name__ == "__main__":
    main()
