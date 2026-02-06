# top-w-decoding

Implementation of **Top-W (Wasserstein-regularized truncation)** for LLM inference-time decoding.

Top-W is a **training-free, logits-processor decoding method** that truncates the next-token distribution using
**token-embedding geometry** (via a Wasserstein-inspired objective) while explicitly trading off:

- **Faithfulness** to the original distribution (geometry-aware distortion penalty)
- **Coherence / sharpness** (entropy control)
- **Retained probability mass** (mass reward / anti-over-truncation)

This repo contains:
- A `transformers` LogitsProcessor implementing Top-W (`logit_processor_w1.py`)
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
