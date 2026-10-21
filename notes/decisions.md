# Research log

Every choice, deviation and result of this reproduction, in order (newest first). The numbers match the files in `results/`.

## 2026-10-22: Website, live demo and DG inspector
- **Site:** a Hugging Face Space (Gradio SDK on ZeroGPU hardware) serves the results explorer, a live demo and a DG inspector (`space/`, built from `explorer/` by `scripts/build_space.py`).
- **Live demo backend:** SA runs on Modal GPUs behind a small token-checking gateway (`scripts/modal_live.py`); workers scale to zero, restore from memory snapshots (about 30 s cold start, 4–8 s per run), and fall back across GPU types when one is unavailable. `SA_BACKEND=local` runs the same code on the Space's CPU instead.
- **DG inspector** (`scripts/export_dg_inspector.py`): the 133 kept rules (119 traced to the exploration episode they came from; the rest were merged by the filter step), every task's calls with and without rules in all 3 runs, and the 80 exploration episodes.
- **Finding from the inspector:** DG's effect per task is stable across runs. DG solves 16 tasks in all 3 runs that no adaptation never solves, and loses 11 that no adaptation always solves; the +3.5-point gain is the net of steady wins and losses, not noise.

## 2026-10-21 (evening): SA measured by task success, and run-to-run noise
**How GTTA applies SA in an episode** (GTTA code, commit 24ba304): BFCL's `base_handler.inference_multi_turn_prompting` writes a reset flag on the first call of each user turn (`count == 0`); `vllm/model_runner.py` then zeroes δ and the Adam state, runs `PA_STEPS` Adam steps on **every prefill** (every model call, whole prompt, continuing δ within the turn), and adds δ during decoding. The released server keeps one δ for the whole batch; I keep one per task.

**Implementation** (`sa/agent_sa.py`, `scripts/modal_sa_agent.py`): a greedy loop in PyTorch (bf16 model, fp32 head). H and the KV cache don't depend on δ, so each episode reuses them and only new tokens are run (checked: incremental H = full-pass H to 1e-4). "No SA" uses the same loop.
- **Shortcut (exact):** one tracked no-SA run checks at every token whether each SA condition would pick a different token. Until the first change the runs are identical (same tokens → same prompts → same δ), so a task with no change has the same outcome. Validated on 1.5B: every "no change" task's full SA run was token-for-token identical. Changed tasks get their own full SA run (`--mode rerun`); lr 1.0×3 changed nearly everything, so it was run in full.

**Results (Qwen2.5-14B, all 200 tasks, `results/sa_agent/sa-agent-14b/summary.md`):**
| condition | tasks where SA changed ≥1 token | success | Δ vs no-SA | 95% CI | SA-only wins / losses |
|---|---|---|---|---|---|
| no SA (PyTorch loop) | – | 19.5% (39) | – | – | – |
| released code, lr 0.1 × 1 (paper default) | 53/200 | 19.5% | **+0.0** | [0.0, 0.0] | 0 / 0 |
| released code, lr 1.0 × 3 (paper's best 14B) | 182/200 | 20.0% | **+0.5** | [−2.5, +3.5] | 5 / 4 |
- At the default setting SA changes some tokens in a quarter of tasks but **never changes an outcome**. At the larger setting it changes almost every trajectory, with gains and losses that cancel.

**Run-to-run noise (vLLM, identical prompts, `results/dg/dg-14b-noise`):** no adaptation 17.5 / 18.5 / 18.5 / 18.0% (4 runs); DG 21.0 / 22.0 / 22.0% (3 runs). Range ≈ 1 point; DG's gain within each run is +3.5, +3.5, +4.0, and two repeats hit the paper's 18.5 → 22.0 exactly. The PyTorch loop's no-SA score (19.5%) is 1–2 points above vLLM's: engine numerics alone move the score this much.

**Correction (SA+DG medians):** the SA+DG entry below quotes test-split medians (5,696 → 8,016 tokens, ‖δ‖ 0.112 → 0.068). Over all 200 tasks, consistent with the other numbers: prompts 5,781 → 7,882 (+36%), released-code ‖δ‖ 0.108 → 0.070 (0.65×), exact 1.00×. The report and README use the all-200 numbers.

**Compute:** about $9 for this step (about 4 GPU-hours on L40S at batch size 1, 13 min on H100, small tests); about $13 for all experiments.

## 2026-10-21 (later): SA with DG rules in the prompt (`qwen2.5-14b-bf16-dg`)
- DG rules (133, from `results/dg/dg-14b/dynamics.json`) inserted with the released ENV_DYNAMICS_SYSTEM_PROMPT. Prompts grow from a median 5,696 → **8,016 tokens** (+41%, test split).
- **Prediction confirmed:** the released-code δ shrinks **0.112 → 0.068 (0.61×)** with the longer DG prompts, while the exact-gradient δ is unchanged (1.00×). SA's effect stays ≈ 0 with DG (−0.0002 [−0.0003, −0.0001]), so **SA + DG ≈ DG**, consistent with the paper's reported naive hybrid (21.0) ≈ DG (22.0) on BFCLv3.
- **Nuance:** DG *lowers* the first-turn correct-call likelihood (−0.124 [−0.204, −0.046]) while raising multi-turn success (+3.5 pts, not significant at n = 200). DG's benefit is about multi-step dynamics, not the first call's wording, so DG is reported by task success only.

## 2026-10-21 (later): DG on BFCLv3 (14B, vLLM 0.32 on H100)
- Vendored from GTTA (commit 24ba304) into `third_party/bfcl/`: API simulators, `multi_turn_utils`, `multi_turn_checker`, prompts; parsing helpers copied verbatim (Python branch). Exploration goals and extraction/filter prompts in `data/bfcl_v3/dg/`. Pipeline: `sa/dg.py`, `scripts/modal_dg.py` (vLLM, step-synchronous batching).
- Loop/checker sanity: an oracle answering with the correct calls scores 139/160 (the rest are BFCL quirks: correct calls with positional arguments that BFCL's keyword-only parser drops, plus a few state mismatches); a do-nothing model scores 0/160.
- vLLM on Modal's slim image needs FlashInfer off (no nvcc): `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`.
- **Run v1 (strict BFCL parsing during exploration), `results/dg/dg-14b-v1-strict-explore`:** baseline 37/200 = 18.5%, identical to the paper's Qwen2.5-14B BFCLv3 baseline; DG 40/200 = 20.0% (paper 22.0). But extraction failed on 55/80 logs: Qwen-14B as explorer wrote several "[call]" lines (or "[call]\n###STOP") per response, BFCL's decoder rejected them, nothing ran, and the logs were empty. Only 27 rules survived.
- **Deviation (exploration only):** execute the first decodable "[call]" line of an exploration response (the exploration prompt asks for one call at a time). Evaluation is unchanged, still strict BFCL.
- **Run v2 (`results/dg/dg-14b`):** exploration 456 steps; extraction 339 rules (5 unparsed); filtering → **133 rules** (GorillaFS 16, Math 9, Message 12, Ticket 12, TradingBot 39, Travel 21, Twitter 13, Vehicle 11).
  - **All 200: no adaptation 35/200 = 17.5%, DG 42/200 = 21.0%, +3.5 pts** (paper: 18.5 → 22.0, +3.5), 95% CI [−2.0, +9.0]; 19 tasks solved only with DG, 12 only without.
  - Test 160: 18.1% → 19.4%, +1.3 pts, CI [−4.4, +6.9].
  - The baseline gave 37/200 in v1 and 35/200 in v2 with identical prompts (temperature 0; vLLM batching nondeterminism), so the evaluation was later repeated to measure noise.

## 2026-10-21 (later): Fairness check at the paper's best 14B setting (lr 1.0; 1 and 3 steps)
- Modal H100, both settings in one container, 400 records. Multi-step PyTorch Adam validated against MLX (δ cosine 0.990–0.9998).
- **Test n = 160, correct-call mean log-prob Δ vs. no SA:**

  | setting | released code | released + user mask | exact gradient | median ‖δ‖ released / exact |
  |---|---|---|---|---|
  | lr 0.1, 1 step | −0.0002 [−0.0004, +0.0000] | −0.0003 [−0.0005, −0.0001] | −0.0108 [−0.0212, −0.0011] | 0.108 / 6.89 |
  | lr 1.0, 1 step | −0.0019 [−0.0046, +0.0004] | −0.0032 [−0.0057, −0.0010] | −0.4376 [−0.566, −0.319] | 1.08 / 68.9 |
  | **lr 1.0, 3 steps (paper's best)** | **−0.0027 [−0.0077, +0.0019]** | **−0.0060 [−0.0110, −0.0016]** | −0.4502 [−0.573, −0.331] | 2.98 / 83.4 |
- **Reading:** the larger learning rate and step count do enlarge the released-code δ (0.11 → 2.98, about 28×), so "lr 1.0 compensates for the small step" is partly true. But even at this setting SA gives **no gain** on correct-call likelihood, and the masked variant is slightly but reliably worse. The exact gradient at lr 1.0 overshoots (prompt loss rises by 0.8 to 1.4). The released-code δ still shrinks with prompt length (Spearman −0.99 in every setting).
- At lr 1.0 × 3 the released-code top tokens shift to identifiers (65%), but still only about 6% of them appear in the correct answer.

## 2026-10-21 (later): 14B full-precision results (the paper's model)
- Modal H100, `Qwen/Qwen2.5-14B-Instruct` bf16: model load 97 s, 200 examples at 0.59 s each.
- **Results (test n = 160)** in `results/summary/qwen2.5-14b-bf16/summary.md`:
  - **What δ boosts:** its top-20 tokens are about 70–75% prompt text (system part) and **only about 7–8% in the correct answer**. Most boosted: `.`, `\n`, space, `,`, `.\n`, `(`, plus **`Note`/` Note`/` note`** (exact gradient): BFCL appends "Note that the provided function is in Python 3 syntax." to every function, so it's the most repeated text in the prompt. The masked variant boosts schema words (`float`, `dict`, `current`). δ follows the prompt's most frequent text, not the vocabulary the answer needs.
  - **Correct-call likelihood:** released code −0.0002 [−0.0004, +0.0000] (≈ no effect); released + mask −0.0003 [−0.0005, −0.0001]; **exact −0.0108 [−0.0212, −0.0011] (significantly worse)**; baseline "boost every prompt token" (α = 0.25) −0.0007 [−0.0022, +0.0007]; random δ ≈ 0. Exact SA is worse than its same-size random control (−0.0101 [−0.0209, −0.0001]).
  - **Shrinking step:** median ‖δ‖ released 0.108 vs. exact 6.888 (~64×), 100% |g| < ε, Spearman(‖δ‖, prompt length) = −0.99.
- **Overall picture (both models):** (1) as released, SA's step shrinks with prompt length and is nearly a no-op on these ~6K-token prompts; (2) a full-size step doesn't help this metric either: learning δ from the prompt's own text pushes toward its most frequent tokens, and on 14B that slightly hurts the correct answer.
- **Caveats:** first-turn likelihood, not multi-turn task success (measured later, see above); BFCL only; prompts dominated by function documentation.
- **Pre-registration:** predictions were not written down before the results, so nothing here is pre-registered; the report says so.

## 2026-10-21 (later): Cloud GPUs; 1.5B full results
- Running 14B on the laptop (4-bit, MLX) would take 6–8 h for 200 examples, so experiments moved to cloud GPUs (Modal) in full bf16 precision, the paper's precision. An earlier partial MLX run (`qwen2.5-1.5b`, 41 examples) is superseded and not used.
- **Engine:** `sa/engine_torch.py` (PyTorch, CUDA). Speed-up: all variants' first step from δ = 0 share p = softmax(HWᵀ), and released = exact / T exactly, so one pass over the vocabulary gives every variant. Cross-checked against MLX on one example: δ norms equal to 3 digits, losses to 3 decimals, top-20 overlap 20/20, 20/20, 19/20.
- **Modal** (`scripts/modal_run.py`): one container with a hard timeout, results streamed back and saved per example. Full 1.5B on L4: 200 examples at about 1.4 s each.
- **1.5B results (test n = 160)** in `results/summary/qwen2.5-1.5b-bf16/summary.md`:
  - δ's top-20 tokens: about 72% in the prompt (mostly system), **about 11% in the correct answer**, **about 50% whitespace/punctuation**; lists are similar across prompts (Jaccard ≈ 0.38). A generic formatting bias, not environment vocabulary.
  - Correct-call log-prob Δ vs. no SA: released +0.0000 [−0.0003, +0.0003]; released + mask −0.0004 [−0.0007, −0.0002]; exact −0.0037 [−0.0119, +0.0043]; baseline (α = 0.25, chosen on val) +0.0007 [−0.0009, +0.0023]. **No reliable gain from any SA variant.** (A 3-example smoke test had suggested the exact gradient helps; it didn't hold up.)
  - **Shrinking step confirmed on all 200:** median ‖δ‖ released 0.10 vs. exact 3.82; 100% |g| < ε; Spearman(‖δ‖, prompt length) = −0.99.

## 2026-10-21: 14B memory test (laptop, 4-bit)
- `mlx-community/Qwen2.5-14B-Instruct-4bit`: loads in 6 s, 8.3 GB; on the longest prompt (T = 7,767): forward 81 s, peak 10.4 GB; each SA variant about 37 s. Untied head, d = 5120, V = 152064.
- The shrinking-step finding holds on 14B: released code (no mask) ‖δ‖ = 0.070 vs. exact 6.861 (~100×), 100% of |g| < ε.

## 2026-10-21: First sign that the released SA step vanishes on long prompts
- `tests/test_alignment.py`: 8/8 pass, including a line-by-line NumPy port of the released gradient (`model_runner.py` lines 1938–1982) matching this implementation's `mode="code"` (cosine > 0.999), and exact gradient == autograd.
- **Masking flag:** `mask_tokens` comes from a `MASK_TOKENS` environment variable posted to a reset server that isn't in the repository, so the value used for the reported numbers is unknown. The paper's Eq. 3 has no masking → three SA variants: **released** (code formula, mask off), **released + user mask**, **exact** (true gradient, the reference).
- **Dev-model measurement (Coder-1.5B, one val prompt):**

  | variant | T=700: Δloss / ‖δ‖ | T=4933 (full prompt): Δloss / ‖δ‖ | % coords with \|g\| < ε (T=4933) |
  |---|---|---|---|
  | released (no mask) | −0.0100 / 0.98 | **−0.00006 / 0.07** | **100%** |
  | released + mask | +0.00008 / 1.88 | −0.00001 / 0.09 | 100% |
  | exact | −0.0276 / 3.86 | −0.0073 / 3.77 | 0.7% |
- **Cause:** the released gradient is divided by T twice (`/num_valid`, then `.mean(0)`), so on realistic prompts |g| ≈ 1e-8 falls below Adam's ε = 1e-6, and the update becomes ≈ lr·g/ε instead of ≈ lr·sign(g): about 50× smaller δ. This may explain small SA gains and why the best 14B result used lr = 1.0.
- Preliminary at this point (one prompt, dev model); confirmed later on all prompts and on 14B (entries above).

## 2026-10-21: The released SA gradient differs from the gradient of its own loss
Reading `vllm/model_runner.py` lines 1952–1983:
- `one_hot` is scattered with the mask, but `probs` isn't masked, so `d_logits = probs − one_hot·mask`. **Masked (user-turn) positions still contribute +softmax**, pushing δ away from whatever the model predicts there.
- It's divided by `num_valid` **and** then `d_hidden.mean(dim=0)` averages over **all** T positions: an extra 1/T factor. With T ≈ 6K tokens, |g| gets close to Adam's ε = 1e-6, which can shrink the first update (normally ≈ lr·sign(g)).
- The logged `loss` includes log(1e-10) for masked positions, so the logged loss values are inflated (the gradient is computed analytically, so this doesn't affect it).
- **Decision:** implement both `mode="code"` (faithful, used for the main results) and `mode="exact"` (true gradient of the masked loss), and report whether results change.

## 2026-10-21: Data and backend checks
- Data: GTTA's own BFCLv3 copy and split (val 40 / test 160), commit 24ba304. Prompts 3.7K–7.7K tokens (median 5.8K), answers 3–159 tokens. The logits computation must be chunked.
- Backend check (Coder-1.5B, bf16): head(hidden) == model forward (max diff 0.0); vs. HF float32: mean absolute logit difference 0.027, argmax agreement 99.0% (bf16 rounding).

## 2026-10-21: Models
- **Main model: Qwen2.5-14B-Instruct**, the paper's headline model. First planned in 4-bit on the laptop (MLX); later run in full bf16 on cloud GPUs (entries above). Fallback: Qwen2.5-7B-Instruct (the paper's ablation model).
- **Helper model: Qwen2.5-1.5B-Instruct** (bf16), used for development, the live demo, and a second replication. Never used for the main reported results.
- **Engine:** MLX at first (direct access to final hidden states and the LM head, autograd available), then PyTorch on CUDA.

## 2026-10-21: Project set up
- Questions: what SA's vector δ boosts; whether it makes the correct first call more likely than simple baselines (a "boost every prompt token" rule and random δ of the same size); later, DG and task success.
- Dev model: Qwen2.5-Coder-1.5B-Instruct (local).
