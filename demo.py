"""
demo.py - run BOTH models on one prompt and print the four signatures side by side.
Usage:  modal run demo.py --prompt "their prompt here"
"""
import modal

app = modal.App("signature-demo")
image = (modal.Image.debian_slim()
         .pip_install("torch", "transformers>=4.50", "accelerate", "safetensors", "huggingface_hub"))
volume = modal.Volume.from_name("signatures-vol", create_if_missing=True)
CACHE = "/cache"
MODELS = ["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"]


@app.function(gpu="A100-80GB", image=image, volumes={CACHE: volume},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def sig(model_name: str, prompt: str):
    import os
    os.environ["HF_HOME"] = CACHE
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager").eval()

    cand = [m for _, m in model.named_modules()
            if isinstance(m, torch.nn.ModuleList) and len(m) > 0
            and (hasattr(m[0], "mlp") or hasattr(m[0], "self_attn"))]
    layers = max(cand, key=len)
    tcfg = getattr(model.config, "text_config", model.config)

    mlp_acts, handles = {}, []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            def mh(m, inp, out, i=i):
                mlp_acts[i] = inp[0].detach().float().cpu().squeeze(0)
            handles.append(mlp.down_proj.register_forward_hook(mh))

    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model(**enc, output_hidden_states=True, output_attentions=True, use_cache=False)
    for h in handles:
        h.remove()

    hs = [h.detach().float().cpu().squeeze(0) for h in o.hidden_states]
    drift = [((hs[i]-hs[i-1]).norm(dim=-1)/hs[i-1].norm(dim=-1).clamp_min(1e-6)).mean().item()
             for i in range(1, len(hs))]
    med = sorted(drift)[len(drift)//2]
    spars = []
    for a in mlp_acts.values():
        spars.append((a.abs() < 0.01*a.abs().max()).float().mean().item())
    sinks = [A[:, :, 0].mean().item() for A in (o.attentions or []) if A is not None]
    return {
        "model": model_name, "tokens": enc["input_ids"].shape[1],
        "final_ratio": drift[-1]/med if med > 0 else 0,
        "sparsity": sum(spars)/len(spars) if spars else 0,
        "sink": sum(sinks)/len(sinks) if sinks else 0,
        "attn_layers": f"{len(sinks)} of {len(layers)}",
    }


@app.local_entrypoint()
def main(prompt: str):
    results = list(sig.map(MODELS, kwargs={"prompt": prompt}))
    print("\n" + "=" * 64)
    print(f"PROMPT: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print("=" * 64)
    print(f"{'signature':<26}{'Gemma 4 E4B':>16}{'Qwen3.5 0.8B':>18}")
    g, q = (results if 'gemma' in results[0]['model'].lower() else results[::-1])
    print(f"{'tokens':<26}{g['tokens']:>16}{q['tokens']:>18}")
    print(f"{'final-layer drift ratio':<26}{g['final_ratio']:>16.2f}{q['final_ratio']:>18.2f}")
    print(f"{'MLP sparsity':<26}{g['sparsity']:>15.1%}{q['sparsity']:>18.1%}")
    print(f"{'attention sink (token 0)':<26}{g['sink']:>15.1%}{q['sink']:>18.1%}")
    print(f"{'attention layers':<26}{g['attn_layers']:>16}{q['attn_layers']:>18}")
    print("=" * 64)
