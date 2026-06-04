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


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: python analyze.py file1.pt [file2.pt ...]"); sys.exit(1)
    sigs = [analyze_file(p) for p in paths]
    plot(sigs)
