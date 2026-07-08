"""
modal_capture_diffusion.py - LLaDA text diffusion model capture on Modal.

LLaDA works by iteratively denoising a fully-masked sequence over N steps.
At each step the model does a full bidirectional forward pass and unmasks
the most-confident tokens. This captures:
  - Per-token confidence at every denoising step  (crystallization timeline)
  - Hidden states at 5 sampled layers at key snapshot steps
  - Which token got locked in at which step
  - Final generated text

Run:
  modal run modal_capture_diffusion.py
"""

import modal

app = modal.App("signature-capture-diffusion")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers==4.48.3", "accelerate",
                 "safetensors", "huggingface_hub", "einops")
)

volume = modal.Volume.from_name("signatures-vol", create_if_missing=True)
CACHE = "/cache"
OUT   = "/cache/out"

MODEL_NAME      = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID         = 126336   # LLaDA [MASK] token id
GEN_LENGTH      = 64       # tokens to generate per prompt
N_STEPS         = 20       # denoising steps
SNAPSHOT_LAYERS = [0, 8, 16, 24, 31]   # layers to snapshot hidden states at
SNAPSHOT_STEPS  = [0, 5, 10, 15, 19]   # denoising steps to save hidden states at

PROMPTS = [
    # 1. multi-hop factual reasoning
    (
        "A traveller starts in the capital of the country that won the 2018 FIFA World Cup. "
        "She takes a train to the largest city in the country directly to the east of that nation. "
        "From there she flies to the capital of the country whose currency is the Yen. "
        "Finally she takes a ferry to the island nation located to the south of that country "
        "whose name means 'eastern sea' in Korean. "
        "List each stop in order, name the city and country at each step, and explain your "
        "reasoning for every leg of the journey."
    ),
    # 2. deep scientific explanation
    (
        "Explain in detail how a modern large language model processes a sentence. "
        "Start from the raw text string and walk through tokenisation, embedding lookup, "
        "positional encoding, the self-attention mechanism including the role of queries, "
        "keys and values, the feed-forward sublayer, residual connections and layer normalisation. "
        "Then explain what 'next-token prediction' means, how the model is trained with "
        "cross-entropy loss, and why scaling the number of parameters tends to improve performance. "
        "Use concrete numerical examples where helpful."
    ),
    # 3. multi-step math word problem
    (
        "A factory produces widgets on three shifts. The morning shift (8 h) produces 120 widgets "
        "per hour but has a 5% defect rate. The afternoon shift (6 h) produces 95 widgets per hour "
        "with a 3% defect rate. The night shift (10 h) produces 80 widgets per hour with a 7% "
        "defect rate. "
        "Answer all of the following, showing every calculation step: "
        "(a) How many total widgets are produced in one full day? "
        "(b) How many defective widgets are produced per day? "
        "(c) What is the overall defect rate for the day? "
        "(d) If the factory needs to ship 3,000 non-defective widgets tomorrow and currently has "
        "500 in stock, how many full days of production are needed?"
    ),
    # 4. code generation with explanation
    (
        "Write a Python class called BinarySearchTree that supports the following operations: "
        "insert(value), search(value) returning True/False, delete(value), and "
        "inorder_traversal() returning a sorted list of all values. "
        "Handle edge cases such as deleting a node with two children correctly using the "
        "in-order successor strategy. "
        "After the code, explain in plain English how the delete operation works for each of "
        "the three cases: leaf node, node with one child, and node with two children. "
        "Include a short usage example at the bottom."
    ),
    # 5. historical causal chain analysis
    (
        "Trace the causal chain that led from the assassination of Archduke Franz Ferdinand "
        "in June 1914 to the United States entering the First World War in April 1917. "
        "Your answer should cover: the alliance system and why a local conflict escalated to a "
        "continental war within six weeks; the Western Front stalemate and its effect on German "
        "strategy; the decision to resume unrestricted submarine warfare and its risks; the "
        "Zimmermann Telegram; and the domestic political pressures on President Wilson. "
        "For each factor explain both what happened and why it mattered causally."
    ),
    # 6. long dual-passage reading comprehension
    (
        "Read the following two passages and answer the questions below.\n\n"
        "PASSAGE A: Photosynthesis is the process by which green plants, algae, and some bacteria "
        "convert light energy into chemical energy stored in glucose. It occurs mainly in the "
        "chloroplasts, organelles that contain the pigment chlorophyll. The overall reaction can "
        "be summarised as: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Photosynthesis has two "
        "main stages. The light-dependent reactions occur in the thylakoid membranes and capture "
        "solar energy to produce ATP and NADPH while splitting water molecules and releasing "
        "oxygen. The light-independent reactions (Calvin cycle) occur in the stroma and use ATP "
        "and NADPH to fix carbon dioxide into glucose.\n\n"
        "PASSAGE B: Cellular respiration is the process by which organisms break down glucose "
        "to release energy in the form of ATP. In aerobic respiration, glucose is broken down "
        "through glycolysis into pyruvate, yielding 2 ATP. Pyruvate enters the mitochondria "
        "where the Krebs cycle and electron transport chain produce approximately 36-38 ATP total. "
        "The overall reaction is: C6H12O6 + 6O2 → 6CO2 + 6H2O + ATP.\n\n"
        "Questions: (1) How are photosynthesis and cellular respiration complementary? "
        "(2) Where in the cell does each process occur? "
        "(3) Is it accurate to say they are simply the reverse of each other? "
        "(4) What happens to a plant's glucose stores if kept in complete darkness for several days?"
    ),
    # 7. constrained creative writing
    (
        "Write a short story of exactly four paragraphs set on the last operating lighthouse "
        "on a remote coastline. The story must satisfy all of the following constraints: "
        "the lighthouse keeper is named Mara and has worked there for thirty years; "
        "a ship appears in a storm in paragraph two; "
        "the third paragraph must include a memory from Mara's childhood; "
        "the story must end ambiguously — the reader should not know whether the ship was saved. "
        "Use vivid sensory detail in every paragraph and vary your sentence length deliberately."
    ),
    # 8. multi-premise logical reasoning
    (
        "Consider the following set of statements:\n"
        "1. All mammals are warm-blooded.\n"
        "2. All warm-blooded animals have a four-chambered heart OR are birds.\n"
        "3. Dolphins are mammals.\n"
        "4. No cold-blooded animal can regulate its own body temperature independently.\n"
        "5. Sharks are not mammals and are not birds.\n"
        "6. Some animals with four-chambered hearts can live in the ocean.\n\n"
        "Answer each question showing your full reasoning chain: "
        "(a) Is a dolphin warm-blooded? "
        "(b) Does a dolphin have a four-chambered heart or is it a bird? "
        "(c) Can a shark regulate its own body temperature independently? "
        "(d) Is it possible for an ocean-dwelling animal to have a four-chambered heart? "
        "(e) What can you NOT determine from these premises alone, and why?"
    ),
    # 9. complex multi-step instructional task
    (
        "You are helping a small business owner set up a basic financial tracking system in Python. "
        "Complete all of the following tasks in a single coherent Python script:\n"
        "1. Define a Transaction dataclass with fields: date, description, amount, category.\n"
        "2. Write add_transaction(ledger, transaction) that appends to a list.\n"
        "3. Write monthly_summary(ledger, month, year) returning total income, expenses, balance.\n"
        "4. Write top_expense_categories(ledger, n=3) returning top n spending categories.\n"
        "5. Write export_to_csv(ledger, filename) that saves all transactions.\n"
        "6. Demonstrate with 8 sample transactions across 3 categories and 2 months.\n"
        "Add docstrings to every function and handle the empty ledger case."
    ),
    # 10. extended cause-and-effect scenario
    (
        "A mid-sized city of 500,000 people decides to ban all private cars from its city centre "
        "starting next year. Analyse the consequences across five dimensions:\n\n"
        "(a) Transportation: How will residents and workers get around? What infrastructure is needed?\n\n"
        "(b) Local economy: Which businesses benefit or suffer? Consider short and long-term effects.\n\n"
        "(c) Environment and health: Effects on air quality, noise, and physical activity levels.\n\n"
        "(d) Social equity: Which socioeconomic groups are most affected and how?\n\n"
        "(e) Political feasibility: What opposition is likely and what compromises could help?\n\n"
        "Conclude: is this policy net beneficial over 10 years, and what are the critical success factors?"
    ),
]


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
)
def capture(model_name: str = MODEL_NAME, prompts: list = PROMPTS,
            gen_length: int = GEN_LENGTH, n_steps: int = N_STEPS):
    import os, torch
    os.environ["HF_HOME"] = CACHE
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    from transformers import AutoTokenizer, AutoModel

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    cfg = getattr(model.config, "text_config", model.config)
    n_layers = getattr(cfg, "num_hidden_layers", 32)
    print(f"model loaded: {n_layers} decoder layers")

    def run_one(prompt):
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            prompt_text = f"<|user|>\n{prompt}\n<|assistant|>\n"

        enc = tok(prompt_text, return_tensors="pt").to(model.device)
        prompt_ids = enc["input_ids"]
        prompt_len = prompt_ids.shape[1]
        prompt_tokens = tok.convert_ids_to_tokens(prompt_ids[0])

        # Start with all generation positions masked
        x = torch.cat([
            prompt_ids,
            torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=model.device)
        ], dim=1)

        locked_step        = [-1] * gen_length  # which step each gen token got locked in
        confidence_history = []                 # [n_steps] of tensor[gen_len]
        pred_history       = []                 # [n_steps] of tensor[gen_len]
        hidden_snapshots   = {}                 # {step: {layer_idx: tensor[seq, d]}}

        for step in range(n_steps):
            want_hidden = step in SNAPSHOT_STEPS
            with torch.no_grad():
                out = model(x, output_hidden_states=want_hidden, use_cache=False)

            if want_hidden and out.hidden_states is not None:
                snap = {}
                for li in SNAPSHOT_LAYERS:
                    if li < len(out.hidden_states):
                        snap[li] = out.hidden_states[li].detach().float().cpu().squeeze(0)
                hidden_snapshots[step] = snap

            # Confidence over generation positions only
            logits = out.logits[0, prompt_len:]          # [gen_len, vocab]
            probs  = torch.softmax(logits.float(), -1)
            conf   = probs.max(-1).values                # [gen_len]
            pred   = probs.argmax(-1)                    # [gen_len]

            confidence_history.append(conf.cpu())
            pred_history.append(pred.cpu())

            gen_ids     = x[0, prompt_len:]
            still_masked = (gen_ids == MASK_ID)
            n_masked    = still_masked.sum().item()
            if n_masked == 0:
                break

            # Uniform schedule: unlock ~gen_length/n_steps tokens per step
            n_to_unlock = min(max(1, round(gen_length / n_steps)), n_masked)

            conf_masked = conf.clone()
            conf_masked[~still_masked] = -1.0
            unlock_local = conf_masked.topk(n_to_unlock).indices

            for li in unlock_local.tolist():
                x[0, prompt_len + li] = pred[li]
                if locked_step[li] == -1:
                    locked_step[li] = step

        final_ids    = x[0, prompt_len:].cpu()
        final_text   = tok.decode(final_ids, skip_special_tokens=True)
        final_tokens = tok.convert_ids_to_tokens(final_ids)

        print(f"  prompt : {prompt[:55]!r}")
        print(f"  output : {final_text[:80]!r}")
        print(f"  locked steps: {sorted(set(s for s in locked_step if s >= 0))}")

        return {
            "prompt":             prompt,
            "prompt_tokens":      prompt_tokens,
            "final_output":       final_text,
            "final_tokens":       final_tokens,
            "locked_step":        locked_step,        # list[int], len=gen_length
            "confidence_history": confidence_history, # list of tensor[gen_len]
            "pred_history":       pred_history,        # list of tensor[gen_len]
            "hidden_snapshots":   hidden_snapshots,    # {step: {layer: tensor}}
            "n_steps":            n_steps,
            "gen_length":         gen_length,
            "prompt_len":         prompt_len,
        }

    captures = []
    for p in prompts:
        captures.append(run_one(p))

    safe = model_name.replace("/", "_")
    os.makedirs(OUT, exist_ok=True)
    out_path = f"{OUT}/signatures-{safe}.pt"
    torch.save({
        "model":           model_name,
        "prompts":         prompts,
        "captures":        captures,
        "n_steps":         n_steps,
        "gen_length":      gen_length,
        "snapshot_layers": SNAPSHOT_LAYERS,
        "snapshot_steps":  SNAPSHOT_STEPS,
    }, out_path)
    volume.commit()

    return {
        "model":      model_name,
        "file":       f"out/signatures-{safe}.pt",
        "n_prompts":  len(prompts),
        "n_layers":   n_layers,
        "n_steps":    n_steps,
        "gen_length": gen_length,
        "outputs":    [c["final_output"][:60] for c in captures],
    }


@app.local_entrypoint()
def main():
    s = capture.remote()
    print("\nCapture summary:")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"\nDownload with:")
    print(f"  modal volume get signatures-vol {s['file']} ./{s['file'].split('/')[-1]}")
