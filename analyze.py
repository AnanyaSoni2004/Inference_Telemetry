"""
analyze.py - compute signatures and their stability across prompts.
Usage:  python analyze.py file1.pt [file2.pt ...]
Handles both single-prompt and multi-prompt capture files.
"""
import sys
import statistics
import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def get_captures(data):
    if "captures" in data:
        return data.get("model", "?"), data["captures"]
    return data.get("model", "?"), [data]


def per_capture(cap):
    hs = cap["hidden_states"]
    drift = [((hs[i] - hs[i - 1]).norm(dim=-1)
              / hs[i - 1].norm(dim=-1).clamp_min(1e-6)).mean().item()
             for i in range(1, len(hs))]
    max_act = [hs[i].abs().max().item() for i in range(len(hs))]
    spars = []
    for a in cap.get("mlp_acts", {}).values():
        thr = 0.01 * a.abs().max()
        spars.append((a.abs() < thr).float().mean().item())
    ent, sink = [], []
    attn = cap.get("attentions")
    if attn:
        for A in attn:
            if A is None:
                continue
            p = A.clamp_min(1e-12)
            ent.append((-(p * p.log()).sum(-1)).mean().item())
            sink.append(A[:, :, 0].mean().item())
    med = sorted(drift)[len(drift) // 2]
    return {"drift": drift, "max_act": max_act,
            "final_ratio": (drift[-1] / med) if med > 0 else float("nan"),
            "sparsity": sum(spars) / len(spars) if spars else None,
            "entropy": sum(ent) / len(ent) if ent else None,
            "sink": sum(sink) / len(sink) if sink else None}


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def analyze_file(path):
    model, caps = get_captures(load(path))
    pcs = [per_capture(c) for c in caps]
    n = len(pcs)
    L = len(pcs[0]["drift"])
    avg_drift = [sum(pc["drift"][i] for pc in pcs) / n for i in range(L)]
    nA = len(pcs[0]["max_act"])
    avg_maxact = [sum(pc["max_act"][i] for pc in pcs) / n for i in range(nA)]

    print(f"\n=== {model} === ({L} layers, {n} prompt(s))")

    def line(name, key, pct=False):
        ms = mean_sd([pc[key] for pc in pcs])
        if ms is None:
            return
        m, sd = ms
        if pct:
            print(f"  {name:26s}: {m:.1%}  (+/- {sd:.1%} across prompts)")
        else:
            print(f"  {name:26s}: {m:.2f}  (+/- {sd:.2f} across prompts)")

    line("final-layer drift ratio", "final_ratio")
    line("MLP sparsity", "sparsity", pct=True)
    line("attention entropy", "entropy")
    line("attention sink (token 0)", "sink", pct=True)
    return {"model": model, "L": L, "avg_drift": avg_drift, "avg_maxact": avg_maxact}


def plot(sigs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(for plots: pip install matplotlib)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for s in sigs:
        ax[0].plot(range(1, s["L"] + 1), s["avg_drift"], marker=".", label=s["model"])
        ax[1].plot(range(len(s["avg_maxact"])), s["avg_maxact"], marker=".", label=s["model"])
    ax[0].set(title="relative drift per layer (avg over prompts)", xlabel="layer", yscale="log")
    ax[1].set(title="peak |activation| per layer (avg over prompts)", xlabel="layer", yscale="log")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("analysis.png", dpi=120)
    print("\nsaved plot -> analysis.png")


def compare(path_a, path_b):
    """Per-prompt side-by-side comparison of two model capture files."""
    data_a = load(path_a)
    data_b = load(path_b)
    model_a, caps_a = get_captures(data_a)
    model_b, caps_b = get_captures(data_b)

    prompts_a = data_a.get("prompts", [c.get("prompt", f"prompt {i}") for i, c in enumerate(caps_a)])
    prompts_b = data_b.get("prompts", [c.get("prompt", f"prompt {i}") for i, c in enumerate(caps_b)])

    n = min(len(caps_a), len(caps_b))
    if len(caps_a) != len(caps_b):
        print(f"warning: prompt counts differ ({len(caps_a)} vs {len(caps_b)}), comparing first {n}")

    METRICS = [("final_ratio", "drift ratio", False),
               ("sparsity",    "MLP sparsity", True),
               ("entropy",     "attn entropy", False),
               ("sink",        "attn sink",    True)]

    col = 28
    ma_short = model_a.split("/")[-1]
    mb_short = model_b.split("/")[-1]
    header = f"  {'prompt':40s}  {ma_short:>12}  {mb_short:>12}  {'delta':>10}"
    print(f"\n{'='*len(header)}")
    print(f"  Side-by-side comparison: {ma_short}  vs  {mb_short}")
    print(f"{'='*len(header)}")

    for metric, label, pct in METRICS:
        print(f"\n--- {label} ---")
        print(header)
        deltas = []
        for i in range(n):
            pc_a = per_capture(caps_a[i])
            pc_b = per_capture(caps_b[i])
            va, vb = pc_a.get(metric), pc_b.get(metric)
            if va is None or vb is None:
                continue
            delta = vb - va
            deltas.append(delta)
            prompt_short = prompts_a[i][:40] if i < len(prompts_a) else f"prompt {i}"
            fmt = "{:.1%}" if pct else "{:.3f}"
            print(f"  {prompt_short:40s}  {fmt.format(va):>12}  {fmt.format(vb):>12}  {fmt.format(delta):>10}")
        if deltas:
            avg = sum(deltas) / len(deltas)
            fmt = "{:.1%}" if pct else "{:.3f}"
            print(f"  {'AVERAGE':40s}  {'':>12}  {'':>12}  {fmt.format(avg):>10}")


def avg_layer_drift(caps):
    all_drifts = []
    for cap in caps:
        hs = cap["hidden_states"]
        drift = [((hs[i] - hs[i - 1]).norm(dim=-1)
                  / hs[i - 1].norm(dim=-1).clamp_min(1e-6)).mean().item()
                 for i in range(1, len(hs))]
        all_drifts.append(drift)
    n_layers = len(all_drifts[0])
    return [sum(d[i] for d in all_drifts) / len(all_drifts) for i in range(n_layers)]


def layer_profile(path_a, path_b):
    """Layer-wise residual drift: where does each model converge vs process?"""
    data_a, data_b = load(path_a), load(path_b)
    model_a, caps_a = get_captures(data_a)
    model_b, caps_b = get_captures(data_b)
    ma = model_a.split("/")[-1]
    mb = model_b.split("/")[-1]

    drift_a = avg_layer_drift(caps_a)
    drift_b = avg_layer_drift(caps_b)
    L_a, L_b = len(drift_a), len(drift_b)

    med_a = sorted(drift_a)[L_a // 2]
    med_b = sorted(drift_b)[L_b // 2]

    min_a, min_b = min(drift_a), min(drift_b)
    max_a, max_b = max(drift_a), max(drift_b)
    min_layer_a = drift_a.index(min_a) + 1
    min_layer_b = drift_b.index(min_b) + 1
    max_layer_a = drift_a.index(max_a) + 1
    max_layer_b = drift_b.index(max_b) + 1
    conv_a = sum(1 for d in drift_a if d < med_a)
    conv_b = sum(1 for d in drift_b if d < med_b)

    def zone(layer, n):
        p = layer / n
        return "early" if p < 0.33 else "middle" if p < 0.67 else "late"

    print(f"\n{'='*72}")
    print(f"  Layer-wise residual drift profile")
    print(f"  {ma} ({L_a} layers)  vs  {mb} ({L_b} layers)")
    print(f"{'='*72}")
    print(f"\n  {'Metric':38s}  {ma:>16}  {mb:>16}")
    print(f"  {'-'*72}")
    print(f"  {'Peak convergence (min drift layer)':38s}  "
          f"{f'L{min_layer_a}/{L_a} [{zone(min_layer_a,L_a)}]':>16}  "
          f"{f'L{min_layer_b}/{L_b} [{zone(min_layer_b,L_b)}]':>16}")
    print(f"  {'Peak processing  (max drift layer)':38s}  "
          f"{f'L{max_layer_a}/{L_a} [{zone(max_layer_a,L_a)}]':>16}  "
          f"{f'L{max_layer_b}/{L_b} [{zone(max_layer_b,L_b)}]':>16}")
    print(f"  {'Convergence zone (below-median layers)':38s}  "
          f"{f'{conv_a}/{L_a} ({100*conv_a//L_a}%)':>16}  "
          f"{f'{conv_b}/{L_b} ({100*conv_b//L_b}%)':>16}")
    print(f"  {'Min drift (quietest layer)':38s}  {min_a:>16.4f}  {min_b:>16.4f}")
    print(f"  {'Max drift (busiest layer)':38s}  {max_a:>16.4f}  {max_b:>16.4f}")
    print(f"  {'Final layer drift':38s}  {drift_a[-1]:>16.4f}  {drift_b[-1]:>16.4f}")
    print(f"  {'Final / median ratio':38s}  {drift_a[-1]/med_a:>16.2f}x  {drift_b[-1]/med_b:>16.2f}x")

    # ASCII bar chart sampled at every 10% of depth
    BAR = 30
    scale = max(max_a, max_b)
    print(f"\n  Layer-by-layer drift (sampled every 10% of depth, bar = relative drift)")
    print(f"\n  {'%':>4}  {'layer':>6}  {ma[:18]:18s}  {'layer':>6}  {mb[:18]:18s}")
    print(f"  {'-'*72}")
    for pct in range(0, 101, 10):
        idx_a = min(int(pct / 100 * L_a), L_a - 1)
        idx_b = min(int(pct / 100 * L_b), L_b - 1)
        da, db = drift_a[idx_a], drift_b[idx_b]
        bar_a = "█" * int(da / scale * BAR)
        bar_b = "█" * int(db / scale * BAR)
        conv_marker_a = " <-- converge" if da < med_a else ""
        conv_marker_b = " <-- converge" if db < med_b else ""
        print(f"  {pct:>3}%  L{idx_a+1:>5}  {da:6.3f} {bar_a:<{BAR}}{conv_marker_a}")
        print(f"  {'':>4}  L{idx_b+1:>5}  {db:6.3f} {bar_b:<{BAR}}{conv_marker_b}")
        print()

    # Save plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Normalized x-axis so both models align at 0%–100% depth
        xa = [100 * i / L_a for i in range(L_a)]
        xb = [100 * i / L_b for i in range(L_b)]

        axes[0].plot(xa, drift_a, label=ma, color="royalblue")
        axes[0].plot(xb, drift_b, label=mb, color="tomato")
        axes[0].axhline(med_a, color="royalblue", linestyle="--", alpha=0.4, label=f"{ma} median")
        axes[0].axhline(med_b, color="tomato",    linestyle="--", alpha=0.4, label=f"{mb} median")
        axes[0].set(title="Residual drift per layer (avg over prompts)",
                    xlabel="depth (%)", ylabel="relative drift")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        axes[1].plot(xa, drift_a, label=ma, color="royalblue")
        axes[1].plot(xb, drift_b, label=mb, color="tomato")
        axes[1].set_yscale("log")
        axes[1].set(title="Same — log scale (convergence zones more visible)",
                    xlabel="depth (%)", ylabel="relative drift (log)")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig("layer_profile.png", dpi=130)
        print("  saved plot -> layer_profile.png")
    except ImportError:
        print("  (pip install matplotlib for plots)")


def token_journey(path, prompt_idx=0):
    """Track how each token's hidden state transforms across layers."""
    import torch.nn.functional as F

    data = load(path)
    model_name, caps = get_captures(data)
    short = model_name.split("/")[-1]

    if prompt_idx >= len(caps):
        print(f"error: prompt_idx {prompt_idx} out of range (0-{len(caps)-1})")
        return

    cap = caps[prompt_idx]
    hs     = cap["hidden_states"]   # list of [seq, d_model], len = n_layers+1
    tokens = cap["tokens"]          # list of str
    seq_len  = len(tokens)
    n_layers = len(hs) - 1
    prompt_text = cap.get("prompt", f"prompt {prompt_idx}")

    print(f"\n{'='*74}")
    print(f"  Token journey: {short}")
    print(f"  Prompt [{prompt_idx}]: {prompt_text[:70]}")
    print(f"  {seq_len} tokens, {n_layers} layers")
    print(f"{'='*74}")

    # Per-token per-layer drift [seq, n_layers]
    layers_drift = []
    for L in range(1, n_layers + 1):
        step = (hs[L] - hs[L-1]).norm(dim=-1) / hs[L-1].norm(dim=-1).clamp_min(1e-6)
        layers_drift.append(step)
    drift_mat = torch.stack(layers_drift, dim=0).T  # [seq, n_layers]

    # Cumulative drift per token (sum over all layers)
    cum_drift = drift_mat.sum(dim=1)  # [seq]

    # Cosine similarity: final repr vs initial embedding, per token
    cos_sim = F.cosine_similarity(hs[0], hs[-1], dim=-1)  # [seq]

    # Zone boundaries
    early_end  = n_layers // 3
    mid_end    = 2 * n_layers // 3

    def top_tokens_in_zone(start, end, n=5):
        zone_drift = drift_mat[:, start:end].sum(dim=1)
        idxs = zone_drift.argsort(descending=True)[:n]
        return [(tokens[i], zone_drift[i].item(), i) for i in idxs]

    def print_zone(label, toks):
        print(f"\n  {label}")
        for tok, val, idx in toks:
            bar = "█" * min(int(val * 4), 40)
            print(f"    [{idx:>3}] {repr(tok):>22}  drift={val:.3f}  {bar}")

    print(f"\n  Zones:  early=L1-L{early_end}  "
          f"mid=L{early_end+1}-L{mid_end}  "
          f"late=L{mid_end+1}-L{n_layers}")

    print_zone(f"Most active tokens in EARLY layers (0-33% depth):",
               top_tokens_in_zone(0, early_end))
    print_zone(f"Most active tokens in MIDDLE layers (33-67% depth):",
               top_tokens_in_zone(early_end, mid_end))
    print_zone(f"Most active tokens in LATE layers (67-100% depth):",
               top_tokens_in_zone(mid_end, n_layers))

    # Overall most / least transformed
    print(f"\n  Most transformed tokens (highest cumulative drift across ALL layers):")
    for idx in cum_drift.argsort(descending=True)[:5]:
        val = cum_drift[idx].item()
        bar = "█" * int(val / cum_drift.max().item() * 30)
        print(f"    [{idx:>3}] {repr(tokens[idx]):>22}  total={val:.3f}  {bar}")

    print(f"\n  Most stable tokens (representation closest to initial embedding at final layer):")
    for idx in cos_sim.argsort(descending=True)[:5]:
        print(f"    [{idx:>3}] {repr(tokens[idx]):>22}  cos_sim={cos_sim[idx].item():.4f}")

    print(f"\n  Most semantically shifted (cos_sim lowest — direction flipped most):")
    for idx in cos_sim.argsort(descending=False)[:5]:
        print(f"    [{idx:>3}] {repr(tokens[idx]):>22}  cos_sim={cos_sim[idx].item():.4f}")

    # ASCII heatmap (sampled at 10% depth intervals, per-token normalized)
    sample_layers = [min(int(n_layers * p / 10), n_layers - 1) for p in range(1, 11)]
    print(f"\n  Drift heatmap per token (per-token scale: █=token's own peak, ·=quiet)")
    header_layers = "  ".join(f"L{l+1:<3}" for l in sample_layers)
    print(f"  {'token':>16}  {header_layers}")
    print(f"  {'-'*74}")
    for t in range(seq_len):
        tok_str = repr(tokens[t])[:14]
        row_max = drift_mat[t].max().item() + 1e-9
        cells = ""
        for l in sample_layers:
            v = drift_mat[t][l].item() / row_max
            if   v > 0.75: cells += "  ███ "
            elif v > 0.50: cells += "  ██· "
            elif v > 0.25: cells += "  █·· "
            else:          cells += "  ··· "
        print(f"  {tok_str:>16}  {cells}")

    # Save visual plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(20, max(5, seq_len * 0.28)))
        yticks = range(seq_len)
        ylabels = [repr(t)[:12] for t in tokens]

        im = axes[0].imshow(drift_mat.numpy(), aspect="auto", cmap="inferno",
                            interpolation="nearest")
        axes[0].set_yticks(yticks); axes[0].set_yticklabels(ylabels, fontsize=6)
        axes[0].set_xlabel("layer"); axes[0].set_title(f"Per-token drift heatmap\n{short}")
        plt.colorbar(im, ax=axes[0])

        axes[1].barh(yticks, cum_drift.numpy(), color="steelblue")
        axes[1].set_yticks(yticks); axes[1].set_yticklabels(ylabels, fontsize=6)
        axes[1].set_xlabel("cumulative drift (all layers)")
        axes[1].set_title("Total transformation per token")
        axes[1].invert_yaxis()

        axes[2].barh(yticks, cos_sim.numpy(), color="darkorange")
        axes[2].set_yticks(yticks); axes[2].set_yticklabels(ylabels, fontsize=6)
        axes[2].set_xlabel("cosine similarity to initial embedding")
        axes[2].set_title("Direction preserved from embedding\n(1.0=unchanged, 0=orthogonal)")
        axes[2].axvline(0, color="black", linewidth=0.5)
        axes[2].invert_yaxis()

        fig.tight_layout()
        out = f"token_journey_{short}_p{prompt_idx}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n  saved plot -> {out}")
    except ImportError:
        print("  (pip install matplotlib for the visual plot)")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: python analyze.py file1.pt [file2.pt ...]")
        print("       python analyze.py --compare gemma.pt qwen.pt")
        print("       python analyze.py --layers  gemma.pt qwen.pt")
        print("       python analyze.py --tokens  model.pt [prompt_idx]")
        sys.exit(1)
    if paths[0] == "--compare" and len(paths) == 3:
        compare(paths[1], paths[2])
    elif paths[0] == "--layers" and len(paths) == 3:
        layer_profile(paths[1], paths[2])
    elif paths[0] == "--tokens" and len(paths) >= 2:
        idx = int(paths[2]) if len(paths) == 3 else 0
        token_journey(paths[1], idx)
    else:
        sigs = [analyze_file(p) for p in paths]
        plot(sigs)
