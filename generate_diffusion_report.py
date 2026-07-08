"""
generate_diffusion_report.py
Comprehensive layer-by-layer analysis report for the LLaDA text diffusion model.
Explains observed patterns in terms of LLaDA's architecture.

Usage:
  python generate_diffusion_report.py
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch

LLADA_PT = "signatures-GSAI-ML_LLaDA-8B-Instruct.pt"
OUT_PDF  = "diffusion_layer_analysis_report.pdf"

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

ARCH_NOTES = {
    0: "Bidirectional attn: all tokens visible simultaneously",
    8: "Early MLP: local feature extraction, token-type encoding",
    16: "Mid-depth: cross-token reasoning, semantic binding",
    24: "Deep layers: highest drift — abstract concept formation",
    31: "Final layer: output projection prep, accelerates late",
}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ── helpers ────────────────────────────────────────────────────────────────

def get_layer_step_drift(cap, snapshot_layers, snapshot_steps):
    """Returns a [n_layers, n_steps-1] matrix of drift between consecutive snapshots."""
    snaps      = cap.get("hidden_snapshots", {})
    prompt_len = cap.get("prompt_len", 0)
    n_intervals = len(snapshot_steps) - 1
    mat = np.zeros((len(snapshot_layers), n_intervals))
    for li_idx, li in enumerate(snapshot_layers):
        for iv in range(n_intervals):
            s0, s1 = snapshot_steps[iv], snapshot_steps[iv + 1]
            if s0 in snaps and s1 in snaps and li in snaps.get(s0, {}) and li in snaps.get(s1, {}):
                t0 = snaps[s0][li][prompt_len:]
                t1 = snaps[s1][li][prompt_len:]
                if t0.shape[0] > 0:
                    mat[li_idx, iv] = ((t1 - t0).norm(dim=-1)
                                       / t0.norm(dim=-1).clamp_min(1e-6)).mean().item()
    return mat


def style_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_color("#30363d")


# ── pages ──────────────────────────────────────────────────────────────────

def page_cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d1117"); ax.axis("off")

    ax.text(0.5, 0.85, "LLaDA Text Diffusion Model",
            ha="center", fontsize=34, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.75, "Layer-by-Layer Internal Analysis\nand Architectural Interpretation",
            ha="center", fontsize=18, color="#8b949e", transform=ax.transAxes, linespacing=1.7)
    ax.axhline(0.66, xmin=0.1, xmax=0.9, color="#30363d", linewidth=1)

    bullets = [
        ("Model",      "GSAI-ML/LLaDA-8B-Instruct · 8B params · 32 layers · bidirectional attention"),
        ("Paradigm",   "Masked diffusion — iterative denoising over N steps, not left-to-right generation"),
        ("Captures",   "10 diverse prompts · 20 denoising steps · hidden states at 5 layer depths"),
        ("Analysis",   "Token crystallization · layer drift heatmaps · confidence progression · arch explanation"),
    ]
    y = 0.58
    for key, val in bullets:
        ax.text(0.12, y, f"{key}:", ha="left", fontsize=12, fontweight="bold",
                color="#7ee787", transform=ax.transAxes)
        ax.text(0.26, y, val, ha="left", fontsize=11, color="#c9d1d9", transform=ax.transAxes)
        y -= 0.07

    ax.text(0.5, 0.18,
            "Key question: how do LLaDA's bidirectional attention and masked-diffusion training\n"
            "shape what each layer does at each denoising step?",
            ha="center", fontsize=12, color="#8b949e", transform=ax.transAxes, linespacing=1.7,
            style="italic")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_architecture(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.set_facecolor("#0d1117"); ax.axis("off")

    ax.text(0.5, 0.97, "LLaDA Architecture — Why It Behaves Differently from AR Models",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color="white", transform=ax.transAxes)

    sections = [
        ("1. Bidirectional Self-Attention  (vs causal mask in GPT/Gemma/Qwen)",
         "#4f8ef7",
         "LLaDA removes the causal mask so every token can attend to every other token "
         "at every layer. This is why token crystallization is non-causal: 'Paris' "
         "(position 5) can lock in before 'of' (position 2) because there is no left-to-right "
         "ordering constraint. The model resolves tokens by confidence, not by sequence order."),

        ("2. Masked Diffusion Training  (vs next-token prediction)",
         "#f78c6c",
         "During training, a random fraction of tokens are replaced with [MASK] and the model "
         "learns to predict all of them simultaneously. This creates a different inductive bias: "
         "the model learns to reason about the full output shape before committing to any token. "
         "This is why it immediately reserves EOS tokens at the end of the sequence — it 'knows' "
         "how long the answer will be before generating any content."),

        ("3. Iterative Denoising  (vs single forward pass)",
         "#7ee787",
         "Inference runs N forward passes (steps). At each step, the model sees the partially "
         "unmasked sequence and predicts remaining tokens with updated context. Early steps "
         "resolve high-confidence tokens (answer entities, structural markers). Later steps "
         "resolve lower-confidence tokens (connective reasoning, qualifications). "
         "Deep layers (24, 31) show increasing drift over steps — they do more work as context clarifies."),

        ("4. LLaMA-3 Backbone  (same MLP/attention blocks as AR models)",
         "#e3b341",
         "LLaDA reuses the LLaMA-3 8B architecture for its transformer blocks: RoPE positional "
         "encoding, SwiGLU MLP, grouped-query attention. The key change is removing the causal "
         "mask and changing the training objective. This means layer-depth patterns are similar "
         "to LLaMA: early layers (0-8) do local feature extraction, middle layers (8-24) do "
         "cross-token reasoning, deep layers (24-32) do high-level abstraction. But the "
         "TIMING of when each layer activates now also varies across denoising steps."),

        ("5. Confidence-Based Unmasking  (the crystallization scheduler)",
         "#db6d28",
         "At each denoising step, tokens are unmasked in order of prediction confidence. "
         "This means the model's own certainty drives the generation order. High-confidence "
         "tokens (factual entities, EOS, punctuation) crystallize first. Low-confidence tokens "
         "(causal connectives, qualifications, step-by-step reasoning) crystallize last. "
         "On hard prompts (math, policy analysis), initial avg confidence is ~0.2 vs ~0.64 "
         "on factual prompts — the model 'knows' when it is uncertain."),
    ]

    y = 0.89
    for title, color, body in sections:
        ax.text(0.0, y, title, ha="left", va="top", fontsize=11, fontweight="bold",
                color=color, transform=ax.transAxes)
        y -= 0.038
        ax.text(0.02, y, body, ha="left", va="top", fontsize=9, color="#c9d1d9",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                          edgecolor="#30363d", alpha=0.7))
        y -= 0.125

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_layer_step_heatmap(pdf, data_d, prompt_idx):
    cap             = data_d["captures"][prompt_idx]
    snapshot_layers = data_d.get("snapshot_layers", [0, 8, 16, 24, 31])
    snapshot_steps  = data_d.get("snapshot_steps",  [0, 5, 10, 15, 19])
    prompt          = cap["prompt"][:80]

    mat         = get_layer_step_drift(cap, snapshot_layers, snapshot_steps)
    n_intervals = mat.shape[1]
    xlabels     = [f"S{snapshot_steps[i]}→S{snapshot_steps[i+1]}" for i in range(n_intervals)]
    ylabels     = [f"L{li}" for li in snapshot_layers]

    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.35)

    # Left: heatmap
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("#161b22")
    im  = ax0.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax0.set_xticks(range(n_intervals)); ax0.set_xticklabels(xlabels, color="#8b949e", fontsize=10)
    ax0.set_yticks(range(len(snapshot_layers)))
    ax0.set_yticklabels(ylabels, color="#8b949e", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Denoising interval", color="#8b949e", fontsize=11)
    ax0.set_ylabel("Layer", color="#8b949e", fontsize=11)
    ax0.set_title("Hidden state drift: layer × denoising interval\n(brighter = more change)",
                  color="white", fontsize=12)
    plt.colorbar(im, ax=ax0).ax.yaxis.set_tick_params(color="#8b949e")

    # Annotate cells
    for r in range(len(snapshot_layers)):
        for c in range(n_intervals):
            val = mat[r, c]
            ax0.text(c, r, f"{val:.3f}", ha="center", va="center",
                     fontsize=8, color="black" if val > mat.max() * 0.6 else "white",
                     fontweight="bold")

    # Right: architectural interpretation
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor("#0d1117"); ax1.axis("off")

    ax1.text(0.05, 0.97, "Why each layer behaves this way",
             ha="left", va="top", fontsize=11, fontweight="bold",
             color="white", transform=ax1.transAxes)

    interp = [
        ("L0 — Embedding",
         "Stable early, rises late.\nEmbeddings shift as masked\ntokens resolve context."),
        ("L8 — Early MLP",
         "Lowest drift throughout.\nLocal features already set\nafter step 0."),
        ("L16 — Middle",
         "Moderate, peaks mid-denoise.\nCross-token reasoning\nmost active here."),
        ("L24 — Deep",
         "Highest drift overall.\nAbstract concept binding;\nmost sensitive to new info."),
        ("L31 — Final",
         "Accelerates over steps.\nOutput projection prep;\nmore active as answer forms."),
    ]
    colors = ["#4f8ef7", "#8b949e", "#7ee787", "#f78c6c", "#e3b341"]
    y = 0.88
    for (layer, text), col in zip(interp, colors):
        ax1.text(0.05, y, layer, ha="left", va="top", fontsize=9.5, fontweight="bold",
                 color=col, transform=ax1.transAxes)
        y -= 0.05
        ax1.text(0.07, y, text, ha="left", va="top", fontsize=8.5, color="#c9d1d9",
                 transform=ax1.transAxes, linespacing=1.4)
        y -= 0.135

    fig.suptitle(f"{PROMPT_LABELS[prompt_idx]}\n\"{prompt}...\"",
                 color="white", fontsize=11, y=1.02)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_crystallization(pdf, data_d, prompt_idx):
    cap        = data_d["captures"][prompt_idx]
    conf_hist  = cap["confidence_history"]
    locked     = cap["locked_step"]
    final_toks = cap["final_tokens"]
    gen_length = cap["gen_length"]
    n_steps    = cap["n_steps"]
    prompt     = cap["prompt"][:80]

    conf_mat   = torch.stack(conf_hist, dim=0).numpy()
    avg_conf   = conf_mat.mean(axis=1)
    locked_arr = np.array(locked, dtype=float)
    locked_arr[locked_arr < 0] = np.nan

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    # Panel 1: confidence heatmap
    ax0 = fig.add_subplot(gs[0]); style_ax(ax0)
    im  = ax0.imshow(conf_mat.T, aspect="auto", cmap="viridis",
                     vmin=0, vmax=1, interpolation="nearest")
    ax0.set_xlabel("Denoising step", color="#8b949e", fontsize=10)
    ax0.set_ylabel("Gen token position", color="#8b949e", fontsize=10)
    ax0.set_title("Confidence per token per step\n(yellow=certain, purple=uncertain)",
                  color="white", fontsize=10)
    plt.colorbar(im, ax=ax0)

    # Panel 2: crystallization step per token
    ax1 = fig.add_subplot(gs[1]); style_ax(ax1)
    colors_bar = plt.cm.RdYlGn_r(locked_arr / n_steps)
    ax1.barh(range(gen_length), locked_arr, color=colors_bar, alpha=0.9)
    ax1.set_xlabel("Step locked in", color="#8b949e", fontsize=10)
    ax1.set_title("Crystallization order\n(left=early=confident, right=late=uncertain)",
                  color="white", fontsize=10)
    ax1.invert_yaxis()
    ax1.axvline(n_steps // 3, color="#f78c6c", linestyle="--", alpha=0.6, linewidth=1,
                label="Early zone")
    ax1.axvline(2 * n_steps // 3, color="#e3b341", linestyle="--", alpha=0.6, linewidth=1,
                label="Late zone")
    ax1.legend(fontsize=7, facecolor="#21262d", labelcolor="white")
    if final_toks:
        ax1.set_yticks(range(gen_length))
        ax1.set_yticklabels([repr(t)[:10] for t in final_toks[:gen_length]],
                            fontsize=4.5, color="#c9d1d9")

    # Panel 3: avg confidence + annotation
    ax2 = fig.add_subplot(gs[2]); style_ax(ax2)
    ax2.plot(range(len(avg_conf)), avg_conf, marker="o", markersize=4,
             color="#7ee787", linewidth=2)
    ax2.fill_between(range(len(avg_conf)), avg_conf, alpha=0.15, color="#7ee787")

    # Mark the inflection (steepest rise)
    diffs = np.diff(avg_conf)
    peak_step = int(np.argmax(diffs))
    ax2.axvline(peak_step, color="#f78c6c", linestyle="--", alpha=0.7, linewidth=1.2)
    ax2.text(peak_step + 0.3, avg_conf[peak_step],
             f"steepest rise\nat step {peak_step}", fontsize=7.5, color="#f78c6c")

    ax2.set_xlabel("Denoising step", color="#8b949e", fontsize=10)
    ax2.set_ylabel("Avg confidence", color="#8b949e", fontsize=10)
    ax2.set_title("Confidence rise — architectural reason:\nbidirectional context improves each step",
                  color="white", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.15, color="#30363d")

    # Annotate start/end confidence
    ax2.text(0.5, avg_conf[0] + 0.03, f"start: {avg_conf[0]:.3f}",
             fontsize=8, color="#8b949e")
    ax2.text(len(avg_conf) - 3, avg_conf[-1] - 0.06, f"end: {avg_conf[-1]:.3f}",
             fontsize=8, color="#8b949e")

    fig.suptitle(f"{PROMPT_LABELS[prompt_idx]} — Token Crystallization\n\"{prompt}...\"",
                 color="white", fontsize=11, y=1.02)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_layer_drift_all_prompts(pdf, data_d):
    """One chart: layer drift aggregated over all prompts, per denoising interval."""
    caps            = data_d["captures"]
    snapshot_layers = data_d.get("snapshot_layers", [0, 8, 16, 24, 31])
    snapshot_steps  = data_d.get("snapshot_steps",  [0, 5, 10, 15, 19])
    n_intervals     = len(snapshot_steps) - 1

    # Average drift matrix over all prompts
    avg_mat = np.zeros((len(snapshot_layers), n_intervals))
    count   = np.zeros_like(avg_mat)
    for cap in caps:
        m = get_layer_step_drift(cap, snapshot_layers, snapshot_steps)
        mask = m > 0
        avg_mat += m
        count   += mask.astype(float)
    count[count == 0] = 1
    avg_mat /= count

    xlabels = [f"S{snapshot_steps[i]}→S{snapshot_steps[i+1]}" for i in range(n_intervals)]
    colors  = ["#4f8ef7", "#8b949e", "#7ee787", "#f78c6c", "#e3b341"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")

    # Left: line chart — each layer's drift over time
    ax = axes[0]; style_ax(ax)
    for li_idx, li in enumerate(snapshot_layers):
        ax.plot(range(n_intervals), avg_mat[li_idx], marker="o", markersize=6,
                linewidth=2.2, color=colors[li_idx],
                label=f"Layer {li} — {ARCH_NOTES.get(li, '')[:30]}")
    ax.set_xticks(range(n_intervals)); ax.set_xticklabels(xlabels, color="#8b949e", fontsize=10)
    ax.set_ylabel("Avg hidden state drift (gen tokens)", color="#8b949e", fontsize=11)
    ax.set_title("Layer drift across denoising intervals\n(avg over all 10 prompts)",
                 color="white", fontsize=12)
    ax.legend(fontsize=8, facecolor="#21262d", labelcolor="white", framealpha=0.9)
    ax.grid(alpha=0.15, color="#30363d")

    # Right: heatmap
    ax2 = axes[1]; ax2.set_facecolor("#161b22")
    im  = ax2.imshow(avg_mat, aspect="auto", cmap="hot", interpolation="nearest")
    ax2.set_xticks(range(n_intervals)); ax2.set_xticklabels(xlabels, color="#8b949e", fontsize=10)
    ax2.set_yticks(range(len(snapshot_layers)))
    ax2.set_yticklabels([f"L{li}" for li in snapshot_layers], color="#8b949e", fontsize=11)
    ax2.set_title("Same data as heatmap\n(brighter = more drift = more work done here)",
                  color="white", fontsize=12)
    for sp in ax2.spines.values(): sp.set_color("#30363d")
    ax2.tick_params(colors="#8b949e")
    for r in range(len(snapshot_layers)):
        for c in range(n_intervals):
            ax2.text(c, r, f"{avg_mat[r,c]:.3f}", ha="center", va="center",
                     fontsize=9, color="black" if avg_mat[r,c] > avg_mat.max()*0.6 else "white",
                     fontweight="bold")
    plt.colorbar(im, ax=ax2)

    fig.suptitle("Aggregate Layer-by-Layer Drift — LLaDA 8B — All 10 Prompts",
                 color="white", fontsize=14, y=1.02)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_confidence_comparison(pdf, data_d):
    caps   = data_d["captures"]
    cmap   = plt.cm.tab10

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")

    # Left: all prompts on one chart
    ax = axes[0]; style_ax(ax)
    for i, cap in enumerate(caps):
        avg_conf = [ch.mean().item() for ch in cap["confidence_history"]]
        ax.plot(range(len(avg_conf)), avg_conf, marker="o", markersize=3,
                linewidth=1.8, color=cmap(i / len(caps)),
                label=PROMPT_LABELS[i] if i < len(PROMPT_LABELS) else f"P{i}", alpha=0.9)
    ax.set_xlabel("Denoising step", color="#8b949e", fontsize=11)
    ax.set_ylabel("Avg confidence", color="#8b949e", fontsize=11)
    ax.set_title("Confidence rise per prompt\n(steeper = model resolves faster)",
                 color="white", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, facecolor="#21262d", labelcolor="white",
              framealpha=0.9, loc="lower right")
    ax.grid(alpha=0.15, color="#30363d")

    # Right: starting confidence bar chart (how uncertain at step 0)
    ax2 = axes[1]; style_ax(ax2)
    start_conf = [cap["confidence_history"][0].mean().item() for cap in caps]
    labels     = [f"P{i}" for i in range(len(caps))]
    bar_colors = [cmap(i / len(caps)) for i in range(len(caps))]
    bars = ax2.bar(labels, start_conf, color=bar_colors, alpha=0.85)
    ax2.set_ylabel("Avg confidence at step 0", color="#8b949e", fontsize=11)
    ax2.set_title("Initial uncertainty per prompt\n(lower = model must work harder)",
                  color="white", fontsize=12)
    ax2.axhline(np.mean(start_conf), color="white", linestyle="--",
                alpha=0.5, linewidth=1, label=f"Mean: {np.mean(start_conf):.3f}")
    ax2.legend(fontsize=9, facecolor="#21262d", labelcolor="white")
    for bar, val in zip(bars, start_conf):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#c9d1d9")
    ax2.set_xticklabels(labels, color="#8b949e", fontsize=9)

    fig.suptitle("LLaDA Confidence Analysis — Architectural Interpretation:\n"
                 "Bidirectional attention means each step refines ALL tokens using full context",
                 color="white", fontsize=13, y=1.04)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_findings(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#0d1117")
    ax  = fig.add_axes([0.05, 0.03, 0.90, 0.94])
    ax.set_facecolor("#0d1117"); ax.axis("off")

    ax.text(0.5, 0.98, "Key Findings — LLaDA Layer Analysis",
            ha="center", va="top", fontsize=17, fontweight="bold",
            color="white", transform=ax.transAxes)

    findings = [
        ("Layer 24 does the most work at every denoising stage",
         "#f78c6c",
         "Across all 10 prompts, Layer 24 consistently shows the highest hidden-state drift "
         "between snapshot steps. This is the deep abstract reasoning layer in LLaMA-style architectures. "
         "WHY: At this depth, the residual stream has accumulated local and mid-level features from "
         "earlier layers. Layer 24 uses bidirectional attention to bind concepts across the entire "
         "sequence — it is most sensitive to new information as masked tokens are resolved."),

        ("Layer 31 (final) accelerates over denoising steps",
         "#7ee787",
         "Unlike earlier layers whose drift is roughly constant or decreasing, Layer 31 drift "
         "increases from early to late denoising intervals. "
         "WHY: The final layer prepares hidden states for the output projection (unembedding matrix). "
         "Early in denoising most positions are still masked — the final layer has little to do. "
         "As tokens crystallize, the final layer increasingly transforms representations into "
         "output-ready vectors, causing its drift to accelerate."),

        ("Layer 8 is the quietest throughout denoising",
         "#4f8ef7",
         "Early layers (L8) show the lowest drift across all denoising intervals and all prompts. "
         "WHY: Local syntactic and token-type features are established in the first few forward "
         "passes and do not change much as content tokens crystallize. The embedding layer (L0) "
         "shows slightly more drift because the embedding of masked tokens is updated as context "
         "grows, but L8 has already extracted stable low-level features."),

        ("Non-causal crystallization is a direct consequence of bidirectional attention",
         "#e3b341",
         "Tokens crystallize in confidence order, not sequence order. On the factual prompt, "
         "'Paris' (position 5) locked at step 1 while 'of' (position 2) locked at step 2. "
         "WHY: With no causal mask, every attention head can attend to every position. "
         "The model predicts all positions simultaneously at each step. It commits to "
         "whichever position it is most confident about, regardless of where it sits in the sequence."),

        ("Hard prompts show lower initial confidence AND flatter confidence curves",
         "#db6d28",
         "Multi-step math, policy analysis, and code generation start at ~0.2 avg confidence "
         "vs ~0.65 for factual prompts. The confidence curve also rises more gradually. "
         "WHY: The masked-diffusion training objective means the model has learned to calibrate "
         "its confidence. When a prompt requires multi-hop reasoning or knowledge synthesis, "
         "the model genuinely distributes uncertainty across more tokens and more steps. "
         "This is in contrast to AR models which cannot express 'how hard' a generation is."),

        ("The crystallization order reveals the model's implicit reasoning priority",
         "#8b949e",
         "On the factory math problem, '40 km/h' (the final answer) crystallized at steps 1-2, "
         "while 'Total distance', '=', and 'hours' crystallized at steps 19-20. "
         "WHY: The model's confidence in the numeric answer is higher than its confidence in "
         "the explanatory scaffold around it. This suggests LLaDA has internalized the answer "
         "structure early but defers the step-by-step explanation to later denoising passes — "
         "an inversion of how humans typically write (scaffold first, answer second)."),
    ]

    y = 0.91
    for title, color, body in findings:
        ax.text(0.0, y, f"▶  {title}", ha="left", va="top", fontsize=10.5,
                fontweight="bold", color=color, transform=ax.transAxes)
        y -= 0.04
        ax.text(0.03, y, body, ha="left", va="top", fontsize=8.8, color="#c9d1d9",
                transform=ax.transAxes, linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#161b22",
                          edgecolor="#30363d", alpha=0.6))
        y -= 0.118

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {LLADA_PT} ...")
    try:
        data_d = load(LLADA_PT)
    except FileNotFoundError:
        print(f"ERROR: {LLADA_PT} not found."); sys.exit(1)

    n_prompts = len(data_d["captures"])
    print(f"  {n_prompts} prompts, "
          f"{data_d.get('n_steps', '?')} steps, "
          f"snapshot layers: {data_d.get('snapshot_layers', '?')}")

    print(f"\nGenerating {OUT_PDF} ...")
    with PdfPages(OUT_PDF) as pdf:

        page_cover(pdf)
        page_architecture(pdf)

        # Aggregate layer drift over all prompts
        page_layer_drift_all_prompts(pdf, data_d)

        # Confidence comparison
        page_confidence_comparison(pdf, data_d)

        # Per-prompt: layer×step heatmap + crystallization
        for pidx in range(n_prompts):
            print(f"  prompt {pidx}/{n_prompts-1} ...")
            page_layer_step_heatmap(pdf, data_d, pidx)
            page_crystallization(pdf, data_d, pidx)

        # Final findings
        page_findings(pdf)

        info = pdf.infodict()
        info["Title"]   = "LLaDA Diffusion Model — Layer Analysis Report"
        info["Subject"] = "Layer-by-layer internal analysis with architectural interpretation"

    print(f"\n✓ Saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
