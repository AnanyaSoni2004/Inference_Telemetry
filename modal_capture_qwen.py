"""
modal_capture_qwen.py - Qwen 3.5 activation capture on Modal.
Identical harness to modal_capture.py; only the model differs.
Outputs go to a per-model file in the same Volume, so this never
collides with the Gemma capture.
"""

import modal

app = modal.App("signature-capture")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers>=4.50", "accelerate", "safetensors", "huggingface_hub")
)

volume = modal.Volume.from_name("signatures-vol", create_if_missing=True)
CACHE = "/cache"
OUT = "/cache/out"

MODEL_NAME = "Qwen/Qwen3.5-0.8B"   # cheap test model; switch to "Qwen/Qwen3.5-27B" for the real run
PROMPT = "Explain why the sky appears blue, step by step."
CAPTURE_ATTN = True


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def capture(prompt: str = PROMPT, model_name: str = MODEL_NAME,
            capture_attention: bool = CAPTURE_ATTN):
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
        attn_implementation="eager" if capture_attention else "sdpa",
    )
    model.eval()

    enc = tok(prompt, return_tensors="pt").to(model.device)
    token_strs = tok.convert_ids_to_tokens(enc["input_ids"][0])

    mlp_acts, head_out, handles = {}, {}, []
    candidates = [m for _, m in model.named_modules()
                  if isinstance(m, torch.nn.ModuleList) and len(m) > 0
                  and (hasattr(m[0], "mlp") or hasattr(m[0], "self_attn"))]
    layers = max(candidates, key=len)
    tcfg = getattr(model.config, "text_config", model.config)
    n_heads = getattr(tcfg, "num_attention_heads", None)
    print(f"found {len(layers)} decoder layers, {n_heads} attention heads")

    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            def mlp_hook(m, inp, out, i=i):
                mlp_acts[i] = inp[0].detach().float().cpu().squeeze(0)
            handles.append(mlp.down_proj.register_forward_hook(mlp_hook))
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "o_proj") and n_heads:
            def head_hook(m, inp, out, i=i):
                x = inp[0].detach().float().cpu().squeeze(0)
                head_out[i] = x.view(x.shape[0], n_heads, x.shape[1] // n_heads)
            handles.append(sa.o_proj.register_forward_hook(head_hook))

    with torch.no_grad():
        o = model(**enc, output_hidden_states=True,
                  output_attentions=capture_attention, use_cache=False)
    for h in handles:
        h.remove()

    hidden = [h.detach().float().cpu().squeeze(0) for h in o.hidden_states]
    attn = None
    if capture_attention and getattr(o, "attentions", None):
        attn = [None if a is None else a.detach().float().cpu().squeeze(0)
                for a in o.attentions]

    safe = model_name.replace("/", "_")
    os.makedirs(OUT, exist_ok=True)
    out_path = f"{OUT}/signatures-{safe}.pt"
    torch.save(
        {"model": model_name, "prompt": prompt, "tokens": token_strs,
         "hidden_states": hidden, "attentions": attn,
         "mlp_acts": mlp_acts, "head_outputs": head_out},
        out_path,
    )
    volume.commit()

    drift = [(hidden[L] - hidden[L - 1]).norm(dim=-1).mean().item()
             for L in range(1, len(hidden))]
    return {
        "model": model_name,
        "file": f"out/signatures-{safe}.pt",
        "n_tokens": len(token_strs),
        "n_layers": len(hidden) - 1,
        "d_model": hidden[0].shape[-1],
        "head_layers_captured": len(head_out),
        "attn_layers": None if attn is None else sum(a is not None for a in attn),
        "residual_drift_per_layer": drift,
    }


@app.local_entrypoint()
def main():
    s = capture.remote()
    print("Capture summary:")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"\nDownload it with:\n  modal volume get signatures-vol {s['file']} ./{s['file'].split('/')[-1]}")
