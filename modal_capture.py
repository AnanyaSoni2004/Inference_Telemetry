"""
modal_capture.py - multi-prompt activation capture on Modal.
Run:
  modal run modal_capture.py --model "google/gemma-4-E4B-it"
  modal run modal_capture.py --model "Qwen/Qwen3.5-0.8B"
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

MODEL_NAME = "google/gemma-4-E4B-it"
CAPTURE_ATTN = True

PROMPTS = [
    # 1. factual recall (short)
    "What is the capital of France?",
    # 2. scientific explanation
    "Explain why the sky appears blue, step by step.",
    # 3. arithmetic reasoning
    "If a train travels 60 km in 1.5 hours, what is its average speed? Show your reasoning.",
    # 4. code generation
    "Write a Python function that returns the nth Fibonacci number.",
    # 5. historical summarization
    "Summarize the main causes of the First World War in a few sentences.",
    # 6. long reading comprehension
    "Read the following passage carefully and then explain, in your own words, what it is describing. The water cycle is the continuous movement of water within the Earth and atmosphere. It begins when the sun heats water in oceans, lakes, and rivers, causing it to evaporate and rise into the air as water vapor. As this vapor rises, it cools and condenses into tiny droplets, forming clouds in a process called condensation. When the droplets in a cloud grow large and heavy enough, they fall back to the surface as precipitation, which can take the form of rain, snow, sleet, or hail. Some of this water soaks into the ground and is stored as groundwater, while some flows across the land as runoff, gradually making its way back into streams, rivers, and eventually the ocean. Plants also play a role: they absorb water through their roots and release it back into the air through their leaves in a process called transpiration. Together, evaporation, condensation, precipitation, runoff, and transpiration form a closed loop that recycles the same water over and over again across the entire planet. Because the total amount of water on Earth stays roughly constant, the water you drink today may have fallen as rain thousands of years ago, or even passed through a dinosaur long before humans existed. After reading this, summarize the five main stages of the water cycle and explain how they connect to one another in a single continuous process.",
    # 7. creative writing
    "Write a short poem about the ocean at night.",
    # 8. logical / syllogistic reasoning
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain your reasoning step by step.",
    # 9. instruction following / prioritized list
    "List 5 things you would need to survive on a deserted island and briefly explain why each is important.",
    # 10. commonsense / physical reasoning
    "If you place a sealed plastic bottle full of water in the freezer overnight, what will happen to it and why?",
]


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def capture(model_name: str = MODEL_NAME, prompts: list = PROMPTS,
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

    candidates = [m for _, m in model.named_modules()
                  if isinstance(m, torch.nn.ModuleList) and len(m) > 0
                  and (hasattr(m[0], "mlp") or hasattr(m[0], "self_attn"))]
    layers = max(candidates, key=len)
    tcfg = getattr(model.config, "text_config", model.config)
    n_heads = getattr(tcfg, "num_attention_heads", None)
    print(f"found {len(layers)} decoder layers, {n_heads} attention heads")

    def run_one(prompt):
        mlp_acts, head_out, handles = {}, {}, []
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "down_proj"):
                def mh(m, inp, out, i=i):
                    mlp_acts[i] = inp[0].detach().float().cpu().squeeze(0)
                handles.append(mlp.down_proj.register_forward_hook(mh))
            sa = getattr(layer, "self_attn", None)
            if sa is not None and hasattr(sa, "o_proj") and n_heads:
                def hh(m, inp, out, i=i):
                    x = inp[0].detach().float().cpu().squeeze(0)
                    head_out[i] = x.view(x.shape[0], n_heads, x.shape[1] // n_heads)
                handles.append(sa.o_proj.register_forward_hook(hh))
        enc = tok(prompt, return_tensors="pt").to(model.device)
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
        return {"prompt": prompt,
                "tokens": tok.convert_ids_to_tokens(enc["input_ids"][0]),
                "hidden_states": hidden, "attentions": attn,
                "mlp_acts": mlp_acts, "head_outputs": head_out}

    captures = []
    for p in prompts:
        print("capturing:", p[:50])
        captures.append(run_one(p))

    safe = model_name.replace("/", "_")
    os.makedirs(OUT, exist_ok=True)
    out_path = f"{OUT}/signatures-{safe}.pt"
    torch.save({"model": model_name, "prompts": prompts, "captures": captures}, out_path)
    volume.commit()

    return {"model": model_name, "file": f"out/signatures-{safe}.pt",
            "n_prompts": len(prompts),
            "n_layers": len(captures[0]["hidden_states"]) - 1,
            "tokens_per_prompt": [len(c["tokens"]) for c in captures]}


@app.local_entrypoint()
def main(model: str = MODEL_NAME):
    s = capture.remote(model_name=model)
    print("Capture summary:")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"\nDownload it with:\n  modal volume get signatures-vol {s['file']} ./{s['file'].split('/')[-1]}")
