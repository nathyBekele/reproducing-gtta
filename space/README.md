---
title: Reproducing GTTA
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Reproducing GTTA (ICLR 2026): results, live SA, DG rules
---

# Reproducing GTTA

A reproduction of *Test-Time Adaptation for LLM Agents via Environment Interaction* (GTTA, ICLR 2026) on Qwen2.5-14B-Instruct and BFCLv3.

- **Results** (`/`): what SA's vector δ learns on each of the 200 BFCLv3 tasks, task-success results for SA and DG, and the report.
- **SA Inspector** (`/try`, live): run SA's step yourself on a real model, with GTTA's released gradient or the exact gradient, on a BFCL task or your own text.
- **DG Inspector** (`/dg`, saved runs): every DG rule, where it came from, and each task's calls with and without the rules.

## Settings (Space → Settings → Variables and secrets)

| name | kind | value |
|---|---|---|
| `SA_BACKEND` | variable | `modal` (default, GPU) or `local` (this Space's free CPU) |
| `MODAL_URL` | secret | URL of the `gateway` endpoint printed by `modal deploy scripts/modal_live.py` |
| `MODAL_MODELS` | variable | `small,large` (default) or `small` to hide the 14B option |
| `SA_LIVE_TOKEN` | secret | the shared token stored in the Modal secret `sa-live-token` |
| `LOCAL_MODEL` | variable | model for `local` mode (default `Qwen/Qwen2.5-0.5B-Instruct`) |
| `CLARITY_ID` | variable | Microsoft Clarity project ID |
| `FORMSUBMIT_URL` | variable | FormSubmit AJAX URL for visitor emails |
| `RUNS_PER_HOUR` / `RUNS_PER_DAY` | variable | rate limits (default 20 per visitor per hour / 300 per day in total) |

Hardware: **ZeroGPU** (as for NathyB/ir-studio); the GPU work itself runs on Modal or, with `SA_BACKEND=local`, on CPU.
Changing a variable restarts the Space. Switch to the free CPU by setting `SA_BACKEND=local`, then stop the GPU side with `modal app stop sa-live`.
