# top-w-decoding

Implementation of **Top-W (Wasserstein-regularized truncation)** for LLM inference-time decoding.

Top-W is a **training-free, logits-processor decoding method** that truncates the next-token distribution using
**token-embedding geometry** (via a Wasserstein-inspired objective) while explicitly trading off:

- **Faithfulness** to the original distribution (geometry-aware distortion penalty)
- **Coherence / sharpness** (entropy control)
- **Retained probability mass** (mass reward / anti-over-truncation)

This repo contains:
- A `transformers` `LogitsProcessor` implementing Top-W (`logit_processor_w1.py`)
- Scripts to run **lm-eval-harness** experiments (GSM8K / GPQA)
- Scripts to generate + evaluate with **AlpacaEval**

---

## Method (high level)

At each decoding step, the model produces a distribution `p` over the vocabulary. Top-W selects a kept set `S`
and samples from the renormalized distribution `q_S`.

Top-W chooses `S` by minimizing an objective of the form:

\[
F_{\lambda,\beta}(S) = W_1(p, q_S) + \lambda H(q_S) - \beta \log \Gamma_S
\]

where:
- `W1(p, q_S)` is a geometry-aware transport penalty defined on a token metric derived from embeddings
- `H(q_S)` is entropy of the cropped distribution
- `Γ_S = \sum_{i \in S} p_i` is retained mass
- `λ` controls entropy (sharpness / coherence)
- `β` controls mass retention (avoid overly small crops)

**Efficient implementation idea (practical Top-W):**
- Restrict computation to a **candidate pool** `C = top_m` most probable tokens
- Alternate:
  - Build a feasible potential from distances to the current set (geometry)
  - Update `S` efficiently using a **prefix scan** after sorting by a combined score

In practice you run a small number of alternations per decoding step (e.g., 3–10).

---

## Repository structure (what each file does)

### Core decoding
- `logit_processor_w1.py`  
  **Top-W decoding logic** implemented as a Hugging Face `LogitsProcessor`.
  Typical responsibilities:
  - Reads the model’s input embedding matrix to define token geometry (distance metric)
  - Given per-step logits, computes a kept set `S` (using `top_m`, `λ`, `β`, and `Tsel`)
  - Returns modified logits that enforce truncation / renormalization behavior

### Harness glue
- `huggingface.py`  
  Glue code to ensure custom generation kwargs are passed through the `lm_eval` Hugging Face backend.
  This is the “plumbing layer” that makes `--gen_kwargs "top_m=...,lambda_geom=...,beta=..."` reach generation.


---

## Acknowledgements

Parts of the evaluation/harness plumbing were adapted from the reference implementation accompanying:

- Erfan Baghaei Potraghloo, Seyedarmin Azizi, Souvik Kundu, Massoud Pedram,  
  **“Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation”**, 2025.  
  arXiv:2509.02510.

### BibTeX

```bibtex
@misc{potraghloo2025toph,
  title        = {Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation},
  author       = {Erfan Baghaei Potraghloo and Seyedarmin Azizi and Souvik Kundu and Massoud Pedram},
  year         = {2025},
  eprint       = {2509.02510},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}


### lm-eval-harness scripts
- `run.sh`  
  Runs **GSM8K CoT** with `lm_eval`.
  - Expects environment vars `T` and `m` each run
  - Sweeps over a (small) grid of Top-W parameters
  - Parses the lm_eval JSON output and appends a compact record to a single summary file

- `run_gpqa.sh`  
  Runs **GPQA** with `lm_eval` over a list of models.
  - Same aggregation style as `run.sh`
  - Produces one summary JSON per model
  - **Important**: if `set -u` is enabled, ensure `LIMIT_ARGS=()` is defined in the script

### AlpacaEval generation + evaluation
- `alpaca_generate_w.py`  
  Generates completions for the AlpacaEval evaluation set using `transformers.pipeline(...)` and:
  - Injects `logits_processor=[TopW_LogitsProcessor(...)]`
  - Writes outputs in AlpacaEval JSON format (`output`, `instruction`, `generator`, etc.)

- `alpaca_evaluate_w.sh`  
  Convenience runner that:
  1) Calls `alpaca_generate_w.py` to produce a JSON output file  
  2) Runs `alpaca_eval` with an LLM-as-judge config  
  3) Stores evaluation artifacts under `./alpaca_results_w/...`

---

## Installation

### 1) Create an environment
```bash
conda create -n topw python=3.10 -y
conda activate topw
```

### 2) Install dependencies

#### Core
```bash
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # choose the right CUDA wheel for your system
pip install transformers accelerate safetensors sentencepiece
pip install numpy scipy tqdm datasets
```

#### lm-eval-harness
```bash
pip install lm-eval
# or (editable):
# git clone https://github.com/EleutherAI/lm-evaluation-harness.git
# cd lm-evaluation-harness && pip install -e .
```

#### AlpacaEval
```bash
pip install alpaca-eval
```

### 3) (Optional) API keys / auth

If you plan to run AlpacaEval with an OpenAI judge (e.g., `gpt4o`):
```bash
export OPENAI_API_KEY="..."
```

If you evaluate gated Hugging Face models (e.g., some Llama checkpoints):
```bash
export HF_TOKEN="..."
huggingface-cli login
```

### 4) Quick sanity check
```bash
python3 -c "from logit_processor_w1 import TopW_LogitsProcessor; print('OK')"
```

---

## Usage

### Top-W generation knobs

Top-W is controlled via generation kwargs:

- `top_m`: candidate pool size (Top-M by probability, where Top-W operates)
- `selection_temperature` (`Tsel`): temperature used for selection (kept-set decision)
- `temperature` (`T`): final sampling temperature for generation
- `lambda_geom` (`lam`): entropy/sharpness control
- `beta`: mass-retention reward (prevents overly small crops)
- `warm_p`: warm-up behavior (if implemented in your processor)

Optional debug toggles:
- `TOPW_PRINT_PARAMS=1`
- `TOPW_PRINT_KEPT=1`
- `TOPW_DEBUG_STEPS=1`

Optional geometry mode (if implemented):
- `TOPW_GEOM_MODE=...`

---

## Running lm-eval-harness (GSM8K / GPQA)

### GSM8K CoT

`run.sh` expects `T` and `m` as environment variables:

```bash
export DEVICE=cuda:0
export T=1.0
export m=400

bash run.sh
```

Outputs:
- Per-run lm_eval JSON is generated, parsed, then removed
- A single summary file is kept:
  - `temp/all_results_<timestamp>_T${T}_m${m}_geom${TOPW_GEOM_MODE}.json`

The summary contains one record per run with:
- `strict-match_value` / `strict-match_std`
- `flexible-extract_value` / `flexible-extract_std`

### GPQA (multi-model sweep)

```bash
export DEVICE=cuda:0
export T=1.0
export m=1200

bash run_gpqa.sh
```

Outputs:
- One aggregated summary JSON per model under `temp/`

---

## Running AlpacaEval

### 1) Generate model outputs with Top-W

`alpaca_generate_w.py` loads the AlpacaEval eval set and generates completions using:
`logits_processor=[TopW_LogitsProcessor(...)]`.

Example:

```bash
python3 -u alpaca_generate_w.py \
  --save_address "./outputs_llama_T1.0.json" \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --max_new_tokens 1024 \
  --temperature 1.0 \
  --do_sample \
  --lam 2.2 \
  --beta 2.8
```

### 2) Evaluate with AlpacaEval

```bash
alpaca_eval \
  --annotators_config "gpt4o" \
  --model_outputs "./outputs_llama_T1.0.json" \
  --output_path "./alpaca_results_w/meta-llama_Llama-3.1-8B-Instruct_T1.0" \
  --precomputed_leaderboard None
```

Or run the wrapper script (generate + evaluate):

```bash
bash alpaca_evaluate_w.sh
```
