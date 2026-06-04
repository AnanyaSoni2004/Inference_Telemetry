"""Save full signature data + per-prompt spreads to a JSON file."""
import json, sys, statistics
from analyze import load, get_captures, per_capture

if len(sys.argv) < 2:
    print("usage: python save_results.py file1.pt [file2.pt ...]"); sys.exit(1)

out = {}
for p in sys.argv[1:]:
    model, caps = get_captures(load(p))
    pcs = [per_capture(c) for c in caps]
    def col(key):
        xs = [pc[key] for pc in pcs if pc[key] is not None]
        if not xs: return None
        return {"mean": sum(xs)/len(xs),
                "spread": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
                "per_prompt": xs}
    out[model] = {
        "n_prompts": len(pcs),
        "n_layers": len(pcs[0]["drift"]),
        "final_ratio": col("final_ratio"),
        "sparsity": col("sparsity"),
        "entropy": col("entropy"),
        "sink": col("sink"),
        "avg_drift_per_layer": [sum(pc["drift"][i] for pc in pcs)/len(pcs)
                                for i in range(len(pcs[0]["drift"]))],
    }

name = "results_e4b_vs_qwen08_6prompts.json"
with open(name, "w") as f:
    json.dump(out, f, indent=2)
print(f"saved {name}  ({len(out)} models)")
