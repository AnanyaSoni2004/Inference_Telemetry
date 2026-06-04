"""
capture_signatures.py
----------------------
A minimal, correct harness for capturing internal activations ("signatures")
from decoder-only LLMs (Gemma 4, Qwen 3.5, Llama, etc.) during inference.

It taps the five points described in the diagram:
  (1) residual stream in   -> outputs.hidden_states
  (2) attention weights     -> outputs.attentions      (needs eager attention)
  (3) per-head attn output  -> forward hook on o_proj input
  (4) MLP activations       -> forward hook on down_proj input (SwiGLU intermediate)
  (5) residual stream out   -> outputs.hidden_states (last entry)

This file is deliberately the *infrastructure* only. The analysis / hypothesis
logic (correlations, signature definitions) is left to you -- see the stubs
at the bottom.

Run:  python capture_signatures.py
Edit the CONFIG block first. You need a GPU with enough VRAM for your chosen
model size, and (for gated models) `huggingface-cli login` or HF_TOKEN set.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# CONFIG -- edit this
# ----------------------------------------------------------------------------
MODEL_NAME   = "google/gemma-4-E4B-it"      # e.g. gemma-4-{e2b,e4b,26b,31b} or Qwen/Qwen3.5-...
PROMPT       = "Explain why the sky appears blue, step by step."  # use LONG prompts in practice
LAYERS_OF_INTEREST = None    # None = all layers; or e.g. [0, 5, 10, 15, 20]
CAPTURE_ATTENTION  = True     # WARNING: O(layers * heads * seq^2). See notes below.
CAPTURE_MLP        = True
CAPTURE_HEADS      = True
SAVE_PATH    = "signatures.pt"   # set to None to skip saving
DTYPE        = torch.bfloat16    # bf16 for the model; activations are cast to fp32 for analysis
SEED         = 0
# ----------------------------------------------------------------------------

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        # eager is REQUIRED to get attention weights back. The default (SDPA /
        # flash-attention) never materializes the full attention matrix, so
        # output_attentions=True silently returns None with those backends.
        attn_implementation="eager" if CAPTURE_ATTENTION else "sdpa",
    )
    model.eval()
    return tok, model


def layer_modules(model):
    """Return the list of decoder layers. Works for Gemma/Qwen/Llama HF layouts."""
    return model.model.layers


class HookManager:
    """Registers forward hooks for the taps that aren't exposed by the forward
    call (per-head output ③ and MLP activations ④) and stashes them on CPU."""

    def __init__(self, model, layers_of_interest, capture_mlp, capture_heads):
        self.handles = []
        self.mlp_acts = {}     # layer_idx -> [seq, intermediate]
        self.head_out = {}     # layer_idx -> [seq, n_heads, head_dim]
        layers = layer_modules(model)
        idxs = range(len(layers)) if layers_of_interest is None else layers_of_interest

        for i in idxs:
            layer = layers[i]
            if capture_mlp:
                # input to down_proj == act(gate(x)) * up(x) == the SwiGLU neuron activations
                self.handles.append(
                    layer.mlp.down_proj.register_forward_hook(self._mlp_hook(i))
                )
            if capture_heads:
                # input to o_proj == concatenated per-head outputs, pre-merge
                self.handles.append(
                    layer.self_attn.o_proj.register_forward_hook(self._head_hook(i, layer))
                )

    def _mlp_hook(self, i):
        def hook(module, inputs, output):
            self.mlp_acts[i] = inputs[0].detach().float().cpu().squeeze(0)
        return hook

    def _head_hook(self, i, layer):
        n_heads = layer.self_attn.config.num_attention_heads
        def hook(module, inputs, output):
            x = inputs[0].detach().float().cpu().squeeze(0)   # [seq, n_heads*head_dim]
            seq, hidden = x.shape
            head_dim = hidden // n_heads
            self.head_out[i] = x.view(seq, n_heads, head_dim)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def capture():
    tok, model = load_model()

    # Gemma adds a BOS token automatically; keep the offsets so you can align
    # captured positions back to readable tokens.
    enc = tok(PROMPT, return_tensors="pt").to(model.device)
    token_strs = tok.convert_ids_to_tokens(enc["input_ids"][0])
    seq_len = enc["input_ids"].shape[1]
    print(f"prompt tokens: {seq_len}")

    hooks = HookManager(model, LAYERS_OF_INTEREST, CAPTURE_MLP, CAPTURE_HEADS)

    out = model(
        **enc,
        output_hidden_states=True,
        output_attentions=CAPTURE_ATTENTION,
        use_cache=False,
    )
    hooks.remove()

    # ① + ⑤ residual stream: tuple of (n_layers + 1) tensors, index 0 = embeddings
    hidden_states = [h.detach().float().cpu().squeeze(0) for h in out.hidden_states]
    # ② attention weights: tuple of n_layers tensors [heads, seq, seq]
    attentions = (
        [a.detach().float().cpu().squeeze(0) for a in out.attentions]
        if CAPTURE_ATTENTION else None
    )

    captured = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "tokens": token_strs,
        "hidden_states": hidden_states,   # ①⑤
        "attentions": attentions,         # ②
        "head_outputs": hooks.head_out,   # ③
        "mlp_acts": hooks.mlp_acts,       # ④
    }

    # quick shape report so you can sanity-check before analysing
    print(f"hidden_states: {len(hidden_states)} layers, each {tuple(hidden_states[0].shape)}")
    if attentions is not None:
        print(f"attentions:    {len(attentions)} layers, each {tuple(attentions[0].shape)}")
    if hooks.mlp_acts:
        k = next(iter(hooks.mlp_acts)); print(f"mlp_acts[{k}]:  {tuple(hooks.mlp_acts[k].shape)}")
    if hooks.head_out:
        k = next(iter(hooks.head_out)); print(f"head_out[{k}]:  {tuple(hooks.head_out[k].shape)}")

    if SAVE_PATH:
        torch.save(captured, SAVE_PATH)
        print(f"saved -> {os.path.abspath(SAVE_PATH)}")
    return captured


# ----------------------------------------------------------------------------
# YOUR LOGIC GOES HERE. A couple of stubs to show the data shapes -- the actual
# signature definitions and hypotheses are yours to design.
# ----------------------------------------------------------------------------
def example_residual_drift(captured):
    """How much does each token's representation change from layer to layer?
    A simple 'signature' of where in the network the heavy lifting happens."""
    hs = captured["hidden_states"]                       # list of [seq, d_model]
    deltas = []
    for L in range(1, len(hs)):
        step = (hs[L] - hs[L - 1]).norm(dim=-1)          # per-token change at layer L
        deltas.append(step.mean().item())
    return deltas  # plot this; the bumps tell you which layers do the work


def example_head_correlation(captured, layer):
    """Correlation between the per-head output vectors at one layer -- heads that
    behave similarly may form a circuit. (One idea among many; build your own.)"""
    h = captured["head_outputs"][layer]                  # [seq, n_heads, head_dim]
    flat = h.permute(1, 0, 2).reshape(h.shape[1], -1)    # [n_heads, seq*head_dim]
    return torch.corrcoef(flat)                          # [n_heads, n_heads]


if __name__ == "__main__":
    data = capture()
    print("residual drift per layer:", example_residual_drift(data))
