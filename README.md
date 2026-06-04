# Inference Signatures

Observing two large language models — **Gemma 4** and **Qwen 3.5** — during inference, and measuring patterns in their internal activations ("signatures") to compare how each model processes information layer by layer.

The model runs happen on cloud GPUs (via [Modal](https://modal.com)); the analysis runs locally on CPU.

---

## What this project does

While each model reads a prompt, it captures four families of internal signal at every layer:

1. **Residual stream** — the running representation of each token as it moves through the network.
2. **Attention weights** — how much each token attends to every other token.
3. **Per-head attention outputs** — what each individual attention head contributes.
4. **MLP activations** — which feed-forward neurons fire.

From these it computes summary *signatures* (final-layer drift, MLP sparsity, attention entropy, attention sink) and compares the two models on identical prompts. Running over several prompts and reporting the spread separates stable signatures from prompt-dependent ones.

---

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `modal_capture.py` | Modal (GPU) | Captures activations for one model over a list of prompts. Pick the model with `--model`. |
| `analyze.py` | Local (CPU) | Computes signatures from a capture file and reports mean ± spread across prompts. Compares multiple files. |
| `show_attention.py` | Local (CPU) | Prints and plots a single attention matrix (heatmap of token-to-token attention). |
| `add_long_prompt.py` | Local | Appends a long (~300-token) prompt to the prompt list in `modal_capture.py`. |
| `save_results.py` | Local (CPU) | Saves full signature data + per-prompt values to a JSON file. |
| `signatures_long_prompt_report.pdf` / `.docx` | — | Findings report. |
| `report_figure.png` | — | Comparison figure used in the report. |

Capture files are named `signatures-<model>.pt` and hold the raw recordings.

---

## Setup

Requires Python 3.11+. Only `modal` is needed locally for capture; `torch` and `matplotlib` are needed locally for analysis.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install modal torch matplotlib

modal setup                          # authenticate Modal (opens a browser)
modal secret create huggingface HF_TOKEN=hf_your_token   # store HF token as a Modal secret
```

A HuggingFace account and read token are required to download the models. The torch/transformers stack itself runs inside the Modal container, not locally.

---

## Usage

**1. Capture a model** (loops over all prompts, saves to a per-model file in the Modal Volume):

```bash
modal run modal_capture.py --model "google/gemma-4-E4B-it"
modal run modal_capture.py --model "Qwen/Qwen3.5-0.8B"
```

Each run prints a `modal volume get ...` line at the end.

**2. Download the capture files** to your machine (add `--force` to overwrite an existing copy):

```bash
modal volume get signatures-vol out/signatures-google_gemma-4-E4B-it.pt ./signatures-google_gemma-4-E4B-it.pt --force
modal volume get signatures-vol out/signatures-Qwen_Qwen3.5-0.8B.pt ./signatures-Qwen_Qwen3.5-0.8B.pt --force
```

**3. Analyze and compare:**

```bash
python analyze.py signatures-google_gemma-4-E4B-it.pt signatures-Qwen_Qwen3.5-0.8B.pt
```

Output shows each signature as `mean (+/- spread across prompts)`. Small spread = stable; large spread = prompt-dependent. Also writes `analysis.png`.

**4. Look at one attention matrix** (`<file> <attn_layer_index> <prompt_index>`):

```bash
python show_attention.py signatures-Qwen_Qwen3.5-0.8B.pt 0 5
```

**5. Save results to JSON:**

```bash
python save_results.py signatures-google_gemma-4-E4B-it.pt signatures-Qwen_Qwen3.5-0.8B.pt
```

---

## Key choices and gotchas

- **Attention capture cost grows with the square of prompt length.** Hidden states and MLP activations grow only linearly. For long prompts, capture attention on a few selected layers, not all of them.
- **Attention weights require eager attention.** The capture sets `attn_implementation="eager"`; the default fast paths never expose the attention matrix.
- **Qwen uses linear attention on most layers.** It produces a real attention matrix on only ~1 in 4 layers; the harness detects and skips the rest rather than crashing.
- **Cross-model magnitudes are not comparable directly.** The models operate at different internal scales, so representation-change is normalised by representation magnitude before comparing.
- **Compute on GPU, analyze on CPU.** Heavy capture happens on Modal; only the small analysis runs locally. Keep the `.pt` capture files — they are the raw data, and new signatures can be computed from them without re-capturing.

---

## Findings (current, on the small development models)

Measured on a single long (~290-token) prompt:

- **Stable, model-distinguishing:** Qwen performs a dramatic final-layer transformation (~27x a typical layer) while Gemma changes gradually (~2.7x). Gemma's feed-forward layers are much sparser (92.5% vs 69.8% inactive).
- **Short-context only:** the attention sink (attention piling onto the first token) is strong on short prompts but collapses on long input, at every depth.
- **Structural:** Qwen uses full attention on only 6 of 24 layers vs Gemma's 42 of 42; their largest internal activations form at different depths (Gemma mid-network, Qwen at the final layer).

These use the small models (Gemma 4 E4B, Qwen3.5 0.8B) for fast iteration; the full-size models (Gemma 4 31B, Qwen3.5 27B) are the intended confirmation step.
