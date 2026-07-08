"""
modal_capture_qwen.py - Qwen 3.5 multi-prompt activation capture on Modal.
Same harness as modal_capture.py; uses the shared PROMPTS list so both
models are always compared on identical inputs.

Run:
  modal run modal_capture_qwen.py
  modal run modal_capture_qwen.py --model "Qwen/Qwen3.5-27B"
"""

import modal

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
        "and NADPH to fix carbon dioxide into glucose. Factors that limit the rate of "
        "photosynthesis include light intensity, carbon dioxide concentration, and temperature.\n\n"
        "PASSAGE B: Cellular respiration is the process by which organisms break down glucose "
        "to release energy in the form of ATP. In aerobic respiration, which requires oxygen, "
        "glucose is first broken down through glycolysis in the cytoplasm into pyruvate, yielding "
        "2 ATP and 2 NADH. Pyruvate then enters the mitochondria, where the Krebs cycle generates "
        "additional NADH, FADH2, and 2 ATP per glucose. Finally, the electron transport chain on "
        "the inner mitochondrial membrane uses those electron carriers to produce approximately "
        "32-34 ATP through oxidative phosphorylation. The overall reaction is the reverse of "
        "photosynthesis: C6H12O6 + 6O2 → 6CO2 + 6H2O + ~36-38 ATP.\n\n"
        "Questions: (1) Identify two ways in which photosynthesis and cellular respiration are "
        "complementary processes. (2) Where in the cell does each process primarily occur, and "
        "why is compartmentalisation important? (3) A student claims that photosynthesis and "
        "respiration are simply the reverse of each other. Is this fully accurate? Explain. "
        "(4) If a plant is placed in complete darkness for several days, predict what will happen "
        "to its glucose stores and its oxygen consumption, and justify your answer."
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
        "4. No cold-blooded animal can regulate its own body temperature independently of "
        "the environment.\n"
        "5. Sharks are not mammals and are not birds.\n"
        "6. Some animals with four-chambered hearts can live in the ocean.\n\n"
        "Using only these premises and strict logical inference, answer each question and "
        "show your reasoning chain: "
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
        "1. Define a Transaction dataclass with fields: date (str), description (str), "
        "amount (float), and category (str).\n"
        "2. Write a function add_transaction(ledger, transaction) that appends to a list.\n"
        "3. Write a function monthly_summary(ledger, month, year) that returns total income "
        "(positive amounts), total expenses (negative amounts), and net balance for that month.\n"
        "4. Write a function top_expense_categories(ledger, n=3) that returns the n categories "
        "with the highest total spending.\n"
        "5. Write a function export_to_csv(ledger, filename) that saves all transactions.\n"
        "6. Demonstrate the system with at least 8 sample transactions across at least 3 categories "
        "and 2 months, then call each function and print the results.\n"
        "Add docstrings to every function and handle the case where the ledger is empty."
    ),
    # 10. extended cause-and-effect scenario
    (
        "A mid-sized city of 500,000 people decides to ban all private cars from its city centre "
        "starting next year. Analyse the consequences of this policy across the following five "
        "dimensions, considering both first-order and second-order effects:\n\n"
        "(a) Transportation and mobility: How will residents, workers, and visitors get around? "
        "What infrastructure investments will be needed and over what timeline?\n\n"
        "(b) Local economy: Which types of businesses will benefit and which will be harmed? "
        "Consider both short-term disruption and long-term adaptation.\n\n"
        "(c) Environment and public health: Quantify the likely effects on air quality, "
        "noise pollution, and physical activity levels. What second-order health effects follow?\n\n"
        "(d) Social equity: Which socioeconomic groups will be most affected positively and "
        "negatively? How might the policy widen or narrow existing inequalities?\n\n"
        "(e) Political feasibility: What opposition is likely, from whom, and what compromises "
        "or phase-in strategies might make the policy more acceptable?\n\n"
        "Conclude with an overall assessment: is this policy likely to be net beneficial for the "
        "city over a 10-year horizon, and what are the two or three critical success factors?"
    ),
]

app = modal.App("signature-capture-qwen")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers>=4.50", "accelerate", "safetensors", "huggingface_hub")
)

volume = modal.Volume.from_name("signatures-vol", create_if_missing=True)
CACHE = "/cache"
OUT = "/cache/out"

MODEL_NAME = "Qwen/Qwen3.5-27B"
CAPTURE_ATTN = True


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
